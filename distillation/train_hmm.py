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


CONDITIONAL_SCORE = 'mean_output_conditional_log_likelihood_per_attempt'
JOINT_SCORE = 'mean_joint_log_likelihood_per_sequence'


def load_spans(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as source:
        spans = json.load(source)
    if not isinstance(spans, list):
        raise ValueError(f'{path}: expected a list of spans')
    return spans


def validate_spans(data, spans, name):
    """Check row alignment and observed lengths, including censored prefixes."""
    if len(spans) != len(data):
        raise ValueError(f'{name}: spans do not match tensor rows')
    lengths, contexts = [], []
    for span in spans:
        length, context = span.get('length'), span.get('context_length')
        if (type(length) is not int or type(context) is not int
                or not 0 <= context <= length <= data.shape[1]):
            raise ValueError(f'{name}: invalid observed/context length')
        lengths.append(length)
        contexts.append(context)
    padding = torch.arange(data.shape[1])[None, :] >= torch.tensor(lengths)[:, None]
    if not torch.equal(data == -2, padding):
        raise ValueError(f'{name}: observed lengths do not match right padding')
    return contexts


def load_evaluation_data(data_path, dataset, total_chunks, sample_length=None):
    """Use fixed dev rows and the first dev-sized subset in numeric chunk order.

    The builder writes train.spans.json in that same flattened chunk order.
    Read one chunk at a time so the full training set need not stay in memory.
    """
    if total_chunks < 1:
        raise ValueError('total_chunks must be positive')
    prefix = f'{data_path}/{dataset}'
    train_spans = load_spans(f'{prefix}.train.spans.json')
    dev_spans = load_spans(f'{prefix}.dev.spans.json')
    if (train_spans is None) != (dev_spans is None):
        raise ValueError('train and dev spans are both required for comparable conditional scores')
    conditional = dev_spans is not None
    if conditional and sample_length is not None:
        raise ValueError('sample_length would censor versioned training data; use its observed lengths')
    if sample_length is not None and sample_length < 1:
        raise ValueError('sample_length must be positive')

    dev_data = torch.load(f'{prefix}.dev', map_location='cpu')[:, :sample_length]
    if not len(dev_data):
        raise ValueError('development evaluation set is empty')
    dev_contexts = validate_spans(dev_data, dev_spans, 'dev') if conditional else None
    train_parts, train_contexts = [], [] if conditional else None
    offset, selected = 0, 0
    for index in range(total_chunks):
        chunk = torch.load(f'{prefix}.train.{index}', map_location='cpu')
        contexts = (validate_spans(chunk, train_spans[offset:offset + len(chunk)],
                                   f'train chunk {index}') if conditional else None)
        take = min(len(chunk), len(dev_data) - selected)
        if take:
            train_parts.append(chunk[:take, :sample_length].clone())
            if conditional:
                train_contexts.extend(contexts[:take])
            selected += take
        offset += len(chunk)
    if conditional and offset != len(train_spans):
        raise ValueError('train spans do not match total training rows')
    if not selected:
        raise ValueError('training evaluation set is empty')
    train_data = nn.utils.rnn.pad_sequence(
        [row for part in train_parts for row in part], batch_first=True, padding_value=-2)
    return train_data, dev_data, train_contexts, dev_contexts


def shard_evaluation_data(data, contexts, rank, world_size):
    """Partition global evaluation rows exactly once, even with empty ranks."""
    start, end = len(data) * rank // world_size, len(data) * (rank + 1) // world_size
    return data[start:end], None if contexts is None else contexts[start:end]


@torch.no_grad()
def evaluation_scores(model, train_data, dev_data, train_contexts, dev_contexts, batch_size):
    def score(data, contexts):
        return (model.conditional_loglikelihood(data, contexts, batch_size)
                if contexts is not None else model.loglikelihood(data, batch_size)).reshape(())

    # Reduce sums and actual row counts, never averages of unequal rank means.
    totals = torch.stack((score(train_data, train_contexts), score(dev_data, dev_contexts),
                          model.alpha_exp.new_tensor(len(train_data)),
                          model.alpha_exp.new_tensor(len(dev_data)))).to(torch.float64)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return (totals[:2] / totals[2:]).tolist()


def train_hmm(rank, world_size,
    model_path, checkpoint, save_per_step,
    data_path, dataset, total_chunks, batch_size, sample_length,
    em_schedule, log_file, dropout, pseudocount, device=None):

    device = device or f'cuda:{os.environ.get("LOCAL_RANK", rank)}'

    hmm_model = HMM.from_pretrained(f'{model_path}/checkpoint-{checkpoint}', map_location='cpu').to(device)
    hidden_states, vocab_size, eos_token_id = hmm_model.hidden_states, hmm_model.vocab_size, hmm_model.eos_token_id
    eps_cuda = torch.tensor([1e-7], device=device)

    train_eval, dev_data, train_contexts, dev_contexts = load_evaluation_data(
        data_path, dataset, total_chunks, sample_length)
    if rank == 0:
        score_name = CONDITIONAL_SCORE if dev_contexts is not None else JOINT_SCORE
        with open(log_file, 'a+', encoding='utf-8') as fout:
            fout.write(f'# train_score={score_name}\n# dev_score={score_name}\n')
            fout.write(f'# train_eval=first_rows_in_numeric_chunk_order; '
                       f'train_eval_count={len(train_eval)}; dev_eval_count={len(dev_data)}\n')
    train_eval, train_contexts = shard_evaluation_data(train_eval, train_contexts, rank, world_size)
    dev_data, dev_contexts = shard_evaluation_data(dev_data, dev_contexts, rank, world_size)

    def log_scores(checkpoint):
        train_ll, dev_ll = evaluation_scores(
            hmm_model, train_eval, dev_data, train_contexts, dev_contexts, batch_size)
        if rank == 0:
            msg = f'{checkpoint}\t{train_ll}\t{dev_ll}'
            print(msg)
            with open(log_file, 'a+', encoding='utf-8') as fout:
                fout.write(msg + '\n')

    for _, step_size in em_schedule:
        assert step_size <= total_chunks

    step_offset = checkpoint
    log_scores(checkpoint)
    for step_count, step_size in em_schedule:
        for step_idx in range(0, step_count):
            # get train_step for current step
            train_step = torch.cat([torch.load(f'{data_path}/{dataset}.train.{idx % total_chunks}')
                for idx in range(step_offset, step_offset+step_size)], dim=0)

            # get train_data for current process
            num_per_process = train_step.shape[0] // world_size + 1
            train_data = train_step[rank * num_per_process: min(train_step.shape[0], (rank+1) * num_per_process)]
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

            ckpt = step_offset + step_size
            log_scores(ckpt)

            if rank == 0:
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

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if rank == 0:
        with open(args.log_file, 'a+') as fout:
            fout.write(str(vars(args)) + '\n')

    em_schedule = [tuple([int(y) for y in x.split(',')]) for x in args.em_schedule.split(';') if x != '']

    train_hmm(rank, world_size,
        args.model_path, args.checkpoint, args.save_per_step,
        args.data_path, args.dataset, args.total_chunks, args.batch_size, args.sample_length,
        em_schedule, args.log_file, args.dropout, args.pseudocount)
