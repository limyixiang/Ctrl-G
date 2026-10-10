import os
import argparse
import random
import json

import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from ctrlg import HMM


def apply_dropout(input_ids, dropout, vocab_size, eos_token_id):
    n, d = input_ids.shape
    observed = (input_ids >= 0) & (input_ids != eos_token_id)
    input_ids[(torch.rand(n, d, device=input_ids.device) < dropout) & observed] = -1
    return input_ids


def train_hmm(rank, world_size,
    model_path, checkpoint, save_per_step,
    data_path, dataset, total_chunks, batch_size, sample_length,
    em_schedule, log_file, dropout, pseudocount):

    device = f'cuda:{rank}'

    hmm_model = HMM.from_pretrained(f'{model_path}/checkpoint-{checkpoint}', map_location='cpu').to(device)
    hidden_states, vocab_size, eos_token_id = hmm_model.hidden_states, hmm_model.vocab_size, hmm_model.eos_token_id
    eps_cuda = torch.tensor([1e-7], device=device)

    dev_data = torch.load(f'{data_path}/{dataset}.dev')[:, :sample_length]
    dev_size = dev_data.shape[0]
    num_per_process = dev_data.shape[0] // world_size + 1
    dev_start, dev_end = rank * num_per_process, min(dev_data.shape[0], (rank+1) * num_per_process)
    dev_data = dev_data[dev_start:dev_end]
    span_path = f'{data_path}/{dataset}.dev.spans.json'
    context_lengths = None
    if os.path.isfile(span_path):
        with open(span_path) as source:
            spans = json.load(source)
        if len(spans) != dev_size:
            raise ValueError("held-out spans do not match dev tensor")
        context_lengths = [span["context_length"] for span in spans[dev_start:dev_end]]
        if sample_length is not None:
            raise ValueError("sample_length would censor versioned training data; use its observed lengths")
    def dev_score():
        return (hmm_model.conditional_loglikelihood(dev_data, context_lengths, batch_size)
                if context_lengths is not None else hmm_model.loglikelihood(dev_data, batch_size))

    for _, step_size in em_schedule:
        assert step_size <= total_chunks

    step_offset = checkpoint
    for step_count, step_size in em_schedule:
        for step_idx in range(0, step_count):
            # evaluate ll
            if step_offset == checkpoint:
                dev_ll = dev_score()
                torch.distributed.all_reduce(dev_ll, op=dist.ReduceOp.SUM)
                if rank == 0:
                    dev_ll = dev_ll.item() / dev_size
                    msg = f'{checkpoint}\t{-1.0}\t{dev_ll}'
                    print(msg)
                    with open(log_file, 'a+') as fout:
                        fout.write(msg + '\n')

            # get train_step for current step
            train_step = torch.cat([torch.load(f'{data_path}/{dataset}.train.{idx % total_chunks}')
                for idx in range(step_offset, step_offset+step_size)], dim=0)

            # get train_data for current process
            num_per_process = train_step.shape[0] // world_size + 1
            train_data = train_step[rank * num_per_process: min(train_step.shape[0], (rank+1) * num_per_process)]
            train_data_eval = train_data[:dev_data.shape[0]].clone()
            train_eval_count = torch.tensor(float(len(train_data_eval)), device=device)
            torch.distributed.all_reduce(train_eval_count, op=dist.ReduceOp.SUM)
            train_data = apply_dropout(train_data, dropout, vocab_size, eos_token_id)

            # compute flows for one em step
            alpha_flow = torch.zeros(hidden_states, hidden_states, device=device)
            beta_flow = torch.zeros(vocab_size + 1, hidden_states, device=device) # one extra token_id to account for MISSING token
            gamma_flow = torch.zeros(hidden_states, device=device)

            with torch.no_grad():
                for batch_idx in tqdm(range(0, train_data.shape[0], batch_size)):
                    batch_size_ = min(batch_size, train_data.shape[0] - batch_idx)
                    train_data_batch = train_data[batch_idx: batch_idx + batch_size_].to(device)

                    probs = hmm_model.forward(train_data_batch)
                    hmm_model.backward(train_data_batch, probs,
                        alpha_flow, beta_flow, gamma_flow)

                alpha_flow.mul_(hmm_model.alpha_exp)

                # distribute flow on the MISSING token
                missing_flow = beta_flow[vocab_size, :] / (vocab_size-1)
                beta_flow.add_(missing_flow[None, :])
                beta_flow[eos_token_id, :] -= missing_flow
                beta_flow[eos_token_id, :] = torch.maximum(beta_flow[eos_token_id, :], eps_cuda)

                beta_flow = torch.permute(beta_flow[:-1, :], (1, 0)).contiguous() # hidden_states * vocab_size

            torch.distributed.all_reduce(alpha_flow, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(beta_flow, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(gamma_flow, op=dist.ReduceOp.SUM)

            with torch.no_grad():
                # flow to params; in-place to reduce memory consumption
                alpha_flow += pseudocount / alpha_flow.shape[-1]
                beta_flow += pseudocount / beta_flow.shape[-1]
                gamma_flow += pseudocount / gamma_flow.shape[-1]
                alpha_flow.div_(torch.sum(alpha_flow, dim=-1, keepdim=True))
                beta_flow.div_(torch.sum(beta_flow, dim=-1, keepdim=True)).log_()
                gamma_flow.div_(torch.sum(gamma_flow, dim=0, keepdim=True)).log_()
                hmm_model.update_params(alpha_flow, beta_flow, gamma_flow)

                # evaluate ll
                train_ll = hmm_model.loglikelihood(train_data_eval, batch_size)
                dev_ll = dev_score()

            torch.distributed.all_reduce(train_ll, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(dev_ll, op=dist.ReduceOp.SUM)

            if rank == 0:
                train_ll = train_ll.item() / max(float(train_eval_count), 1)
                dev_ll = dev_ll.item() / dev_size
                ckpt = step_offset + step_size
                msg = f'{ckpt}\t{train_ll}\t{dev_ll}'
                print(msg)
                with open(log_file, 'a+') as fout:
                    fout.write(msg + '\n')

                if ckpt % save_per_step == 0 and ckpt != 0:
                    hmm_model.save_pretrained(f'{model_path}/checkpoint-{ckpt}')

            step_offset += step_size

            torch.cuda.empty_cache()

    # Always materialize the final step even when it is not a save interval.
    if rank == 0:
        hmm_model.save_pretrained(f'{model_path}/checkpoint-{step_offset}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--model_path', default='', type=str)
    arg_parser.add_argument('--checkpoint', default=50, type=int)
    arg_parser.add_argument('--save_per_step', default=150, type=int)

    arg_parser.add_argument('--data_path', default='', type=str)
    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--total_chunks', default=200, type=int)
    arg_parser.add_argument('--batch_size', default=32, type=int)
    arg_parser.add_argument('--sample_length', default=None, type=int)
    arg_parser.add_argument('--em_schedule', type=str)
    arg_parser.add_argument('--dropout', default=0.001, type=float)
    arg_parser.add_argument('--pseudocount', default=0.001, type=float)
    arg_parser.add_argument('--seed', default=42, type=int)

    arg_parser.add_argument('--log_file', default='', type=str)

    args = arg_parser.parse_args()

    dist.init_process_group('nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if rank == 0:
        with open(args.log_file, 'a+') as fout:
            fout.write(str(vars(args)) + '\n')
            if os.path.isfile(f'{args.data_path}/{args.dataset}.dev.spans.json'):
                fout.write('# dev_score=mean_output_conditional_log_likelihood_per_attempt\n')

    em_schedule = [tuple([int(y) for y in x.split(',')]) for x in args.em_schedule.split(';') if x != '']

    train_hmm(rank, world_size,
        args.model_path, args.checkpoint, args.save_per_step,
        args.data_path, args.dataset, args.total_chunks, args.batch_size, args.sample_length,
        em_schedule, args.log_file, args.dropout, args.pseudocount)
