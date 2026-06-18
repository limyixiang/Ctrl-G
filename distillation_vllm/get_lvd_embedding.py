import torch
import torch.distributed as dist
from tqdm import tqdm
import os
import argparse
from safetensors.torch import save_file

def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for LVD data (Distributed)")
    parser.add_argument("--file_path", type=str, 
                        default="dapo/DAPO-DAPO-Baseline-V7-S60.lvd",
                        help="Path to the data files (without extension)")
    parser.add_argument("--model_path", type=str,
                        default="/path/to/your/model_or_checkpoint",
                        help="Path to the model directory")
    parser.add_argument("--debug", action="store_true",
                        help="Process only the first 20 samples for debugging")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Initialize distributed processing
    dist.init_process_group('gloo')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{rank}'
    
    file_path = args.file_path
    model_path = args.model_path

    # Load data on all processes (simpler approach)
    output_tokens = torch.load(f"{file_path}")
    input_tokens = torch.load(f"{file_path}.prompt_tokens")
    full_tokens = torch.load(f"{file_path}.full_tokens")
    
    # Limit data for debugging if requested
    if args.debug:
        if rank == 0:
            print("Debug mode: Processing only first 100 samples")
        output_tokens = output_tokens[:100]
        input_tokens = input_tokens[:100]
        full_tokens = full_tokens[:100]
        # output_tokens = output_tokens[:20]
        # input_tokens = input_tokens[:20]
        # full_tokens = full_tokens[:20]

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)

    total_samples = len(output_tokens)

    # Check that the output part matches (only on rank 0 for efficiency)
    if rank == 0:
        assert len(output_tokens) == len(full_tokens) == len(input_tokens), "Length mismatch between output tokens and full tokens"
        
        for full_token, input_token, output_token in zip(full_tokens, input_tokens, output_tokens):
            prompt_output = torch.cat([input_token, output_token], dim=0)
            assert torch.equal(prompt_output, full_token[:len(prompt_output)]), "input + output tokens do not match full tokens"
        
        print("All checks passed!")

    # Divide work among processes
    samples_per_process = total_samples // world_size
    start_idx = rank * samples_per_process
    if rank == world_size - 1:
        end_idx = total_samples  # Last process handles remainder
    else:
        end_idx = start_idx + samples_per_process

    if rank == 0:
        print(f"Processing {total_samples} sequences across {world_size} GPUs")
        print(f"Rank {rank} processing samples {start_idx}:{end_idx}")

    # Process assigned samples
    model.eval()
    output_len = output_tokens.size(1)
    
    with torch.no_grad():
        hidden_state_list = []
        local_samples = end_idx - start_idx
        
        if rank == 0:
            print(f"Rank {rank}: Processing {local_samples} samples in batches of {args.batch_size}...")
        
        for i in tqdm(range(start_idx, end_idx, args.batch_size), disable=rank != 0):
            batch_end = min(i + args.batch_size, end_idx)
            batch_token_ids = full_tokens[i:batch_end].to(device)
            
            # Forward pass for this batch
            outputs = model(input_ids=batch_token_ids, output_hidden_states=True)
            
            # Get the last layer hidden states for this batch
            last_hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
            for j, hidden_state in enumerate(last_hidden_states):
                global_idx = i + j
                prompt_len = input_tokens[global_idx].size(0)
                hidden_state_list.append(hidden_state[prompt_len - 1:prompt_len + output_len - 1, :].cpu())

    # Gather results from all processes
    if len(hidden_state_list) > 0:
        local_embeddings = torch.stack(hidden_state_list)
    else:
        # Handle case where process has no samples
        local_embeddings = torch.empty(0, output_len, model.config.hidden_size)
    
    if rank == 0:
        # Gather all embeddings
        embeddings_list = [torch.empty_like(local_embeddings) for _ in range(world_size)]
        dist.gather(local_embeddings, gather_list=embeddings_list)
        
        # Concatenate and save
        all_embeddings = torch.cat(embeddings_list, dim=0)
        
        if args.debug:
            embedding_file = f"{file_path}.embeddings.debug.safetensors"
        else:
            embedding_file = f"{file_path}.embeddings.safetensors"
        
        save_file({"embeddings": all_embeddings}, embedding_file)
        print(f"Embeddings saved to: {embedding_file}")
    else:
        dist.gather(local_embeddings)
