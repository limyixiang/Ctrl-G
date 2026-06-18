import json
import argparse
from openai import OpenAI
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

def pad_to_len(x, d, eos_token_id):
    if x.shape[1] < d:
        new_shape = x.shape[:1] + (d-x.shape[1],) + x.shape[2:]
        x = torch.cat((x, torch.full(new_shape, eos_token_id, dtype=x.dtype)), dim=1)

    return x

def init_client_and_tokenizer(args):
    """Initialize the tokenizer and the OpenAI client"""
    print(f"Initializing OpenAI client at {args.vllm_url} with model {args.model_name_or_path}")
    client = OpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key)
    print(f"Initializing Tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    print("Client and Tokenizer initialized successfully.")

    return client, tokenizer

def query_vllm(client, prompts, args, stop_token_list = [], eos_token_id=None):
    completion = client.completions.create(
        model=args.model_name_or_path,
        prompt=prompts,
        max_tokens=args.max_gen_tokens,
        n=1,
        stop=stop_token_list,
        timeout = 60*60*2,  # 2 hours timeout
        extra_body= {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "return_token_ids": True
        }
    )

    token_ids_list = []
    num_eos_token = 0
    for choice in completion.choices:
        # Check if token_ids attribute exists, fallback to text if not
        if hasattr(choice, 'token_ids'):
            # if len(choice.token_ids) > args.max_new_tokens:
            #     # Random window of the token_ids if longer than max_new_tokens. Currently disabled. (Conflict with the lvd code)
            #     # Random window of the token_ids if longer than max_new_tokens.
            #     end_offset = args.max_new_tokens // 2 # The truncation token_ids will at least contain half of max_new_tokens
            #     start_idx = torch.randint(0, len(choice.token_ids) - end_offset + 1, (1,)).item()
            #     choice.token_ids = choice.token_ids[start_idx:start_idx + args.max_new_tokens]
            token_ids_list.append(choice.token_ids)
            if eos_token_id is not None and eos_token_id in choice.token_ids:
                num_eos_token += 1
        else:
            # This is a fallback - you might need to tokenize the text manually
            raise AttributeError("token_ids not available in completion response. You may need to tokenize the text manually.")
    print("Number of sequences with eos tokens:", num_eos_token)
    return token_ids_list


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    # Model config
    arg_parser.add_argument('--model_name_or_path', default='', type=str)

    # Data config
    arg_parser.add_argument('--input_file',  default='', type=str)
    arg_parser.add_argument('--output_file',  default='', type=str)
    arg_parser.add_argument('--total_chunks',  default=1, type=int)
    arg_parser.add_argument('--n', default=1, type=int, help='Number of copies per prompt')
    arg_parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the generated sequences before saving')
    arg_parser.add_argument('--lvd', action='store_true', help='Whether to include also the prompt in the saved sequences')
    arg_parser.add_argument('--max_data', default=-1, type=int, help='If >0, limit the number of data samples to process')

    # Generation config
    arg_parser.add_argument('--max_new_tokens', type=int, default=128)
    arg_parser.add_argument('--max_gen_tokens', type=int, default=None, help='Max tokens to generate from vLLM. If not set, defaults to max_new_tokens.')
    arg_parser.add_argument('--top_k', type=int, default=-1)
    arg_parser.add_argument('--top_p', type=float, default=1.0)
    arg_parser.add_argument('--temperature', type=float, default=1.0)
    
    # Client config
    arg_parser.add_argument("--vllm_url", default="http://localhost:8000/v1", help="The URL for the VLLM client")
    arg_parser.add_argument("--vllm_api_key", default="EMPTY", help="The API key for VLLM client, if required")

    args = arg_parser.parse_args()

    if args.max_gen_tokens is None:
        args.max_gen_tokens = args.max_new_tokens
    # Init client and tokenizer
    client, tokenizer = init_client_and_tokenizer(args)

    # load input_data: a list of prompts for sampling data from the base model
    with open(args.input_file, 'r') as fin:
        input_data = json.load(fin)
    if args.max_data > 0:
        input_data = input_data[:args.max_data]
        print(f"Limiting input data to first {args.max_data} samples.")


    for chunk_idx in tqdm(range(0, args.total_chunks)):

        # Create prompts with n duplicates for each input prompt
        all_prompts = []
        for prompt in input_data:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,  # Return string instead of token IDs
            )
            # Add n copies of this prompt
            all_prompts.extend([formatted_prompt] * args.n)

        # Call vLLM server to generate responses
        print(f'Calling vLLM server with {len(all_prompts)} prompts...')
        generated_sequences = query_vllm(client, all_prompts, args, stop_token_list = [tokenizer.eos_token], eos_token_id=tokenizer.eos_token_id)

        # Convert token IDs to tensors and pad them
        sequences = []
        full_sequences = []
        prompt_sequences = []
        max_pad_len = args.max_new_tokens
        for i, token_ids in enumerate(generated_sequences):
            # Convert to tensor and ensure it's 2D (1, sequence_length)
            seq_tensor = torch.LongTensor(token_ids).unsqueeze(0)
            padded_tensor = pad_to_len(seq_tensor, args.max_new_tokens, tokenizer.eos_token_id)
            sequences.append(padded_tensor)  # Keep as 2D tensor (1, sequence_length)

            if args.lvd:
                # If lvd, we do not pad the sequence but include the prompt
                prompt_tokens = tokenizer([all_prompts[i]], return_tensors='pt').input_ids
                # We use padded tensor here to ensure the output token_ids is a subsequence of the full token_ids
                full_seq_tensor = torch.cat([prompt_tokens, padded_tensor], dim=1) 
                # Store the maximum length for padding later
                if full_seq_tensor.size(1) > max_pad_len:
                    max_pad_len = full_seq_tensor.size(1)
                prompt_sequences.append(prompt_tokens[0])
                full_sequences.append(full_seq_tensor)

        # Pad sequences to the maximum length
        if args.lvd:
            for i, full_seq_tensor in enumerate(full_sequences):
                full_sequences[i] = pad_to_len(full_seq_tensor, max_pad_len, tokenizer.eos_token_id)

        # Concatenate all sequences into a single tensor
        sequences = torch.cat(sequences, dim=0)
        if args.lvd:
            full_sequences = torch.cat(full_sequences, dim=0)

        # Shuffle the sequences
        if args.shuffle:
            perm = torch.randperm(len(sequences))
            sequences = sequences[perm, :]
            if args.lvd:
                prompt_sequences = [prompt_sequences[i] for i in perm] # Shuffle prompt sequences accordingly. Can't use tensor operation.
                full_sequences = full_sequences[perm, :]

        # Save output
        output_file = f'{args.output_file}.{chunk_idx}' if args.total_chunks > 1 else f'{args.output_file}'
        torch.save(sequences, output_file)
        print(f'Saved {len(sequences)} sequences to {output_file}')

        if args.lvd:
            torch.save(full_sequences, output_file + '.full_tokens')
            torch.save(prompt_sequences, output_file + '.prompt_tokens') # It's a list of tensors

 