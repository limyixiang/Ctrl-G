import json
import argparse
from openai import AsyncOpenAI
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import asyncio

def pad_to_len(x, d, eos_token_id):
    if x.shape[1] < d:
        new_shape = x.shape[:1] + (d-x.shape[1],) + x.shape[2:]
        x = torch.cat((x, torch.full(new_shape, eos_token_id, dtype=x.dtype)), dim=1)

    return x

def init_client_and_tokenizer(args):
    """Initialize the tokenizer and the OpenAI client"""
    print(f"Initializing OpenAI client at {args.vllm_url} with model {args.model_name_or_path}")
    client = AsyncOpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key)
    print(f"Initializing Tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    print("Client and Tokenizer initialized successfully.")

    return client, tokenizer

# async def process_all_messages(messages_list, client, args, stop_token_list):
#     tasks = [
#         query_vllm_msg(client, message, args, stop_token_list)
#         for message in messages_list
#     ]

#     results = await asyncio.gather(*tasks)
#     return results

async def query_vllm_msg(client, message, args, stop_token_list = []):
    response = await client.chat.completions.create(
        model=args.model_name_or_path,
        messages=message,
        max_tokens=args.max_gen_tokens,
        n=args.n,
        stop=stop_token_list,
        timeout = 60*60*2,  # 2 hours timeout
        extra_body= {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "return_token_ids": True
        }
    )
    # Return only the first choice's token_ids (since n=1)
    token_id_list = []
    for choice in response.choices:
        token_ids = getattr(choice, "token_ids", None)
        if token_ids is None and hasattr(choice, "message"):
            token_ids = getattr(choice.message, "token_ids", None)
        token_id_list.append(token_ids)
    return token_id_list


async def main():
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
    arg_parser.add_argument('--batch_size', default=50, type=int, help='batch size for querying vLLM')
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
        all_messages = []
        for d in input_data:
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{d['mime_type']};base64,{d['img_b64']}"
                            }
                        },
                        {
                            "type": "text", 
                            "text": d['prompt']
                        },
                    ],
                }
            ]

            # Add n copies of this prompt
            all_messages.append(messages)
            # all_messages.extend([messages] * args.n)

        # Create tasks
        stop_token_list = [tokenizer.eos_token]
        generated_sequences = []
        print(f'Calling vLLM server with {len(all_messages)} prompts with n = {args.n}...')

        for i in tqdm(range(0, len(all_messages), args.batch_size)):  # Batch size of 50
            batch_messages = all_messages[i:i+args.batch_size]

            tasks = [
                query_vllm_msg(client, message, args, stop_token_list)
                for message in batch_messages
            ]

            # Call vLLM server to generate responses
            gathered_lists = await asyncio.gather(*tasks)
            for token_id_list in gathered_lists:
                generated_sequences.extend(token_id_list)
        # generated_sequences = query_vllM_msg(client, all_messages, args, stop_token_list = [tokenizer.eos_token], eos_token_id=tokenizer.eos_token_id)

        # Convert token IDs to tensors and pad them
        sequences = []
        for i, token_ids in enumerate(generated_sequences):
            if token_ids is None:
                print(f"Warning: Got None token_ids for sequence {i}, skipping...")
                continue
            # Convert to tensor and ensure it's 2D (1, sequence_length)
            seq_tensor = torch.LongTensor(token_ids).unsqueeze(0)
            padded_tensor = pad_to_len(seq_tensor, args.max_new_tokens, tokenizer.eos_token_id)
            sequences.append(padded_tensor)  # Keep as 2D tensor (1, sequence_length)

        # Skip if no valid sequences
        if not sequences:
            print(f"Warning: No valid sequences generated for chunk {chunk_idx}")
            continue

        # Concatenate all sequences into a single tensor
        sequences = torch.cat(sequences, dim=0)

        # Initialize save_messages for LVD
        save_messages = [msg for msg in all_messages for _ in range(args.n)]  # Duplicate messages accordingly

        # Shuffle the sequences
        if args.shuffle:
            perm = torch.randperm(len(sequences))
            sequences = sequences[perm, :]
            if args.lvd:
                save_messages = [all_messages[i] for i in perm] # Shuffle prompt sequences accordingly. Can't use tensor operation.

        # Save output
        output_file = f'{args.output_file}.{chunk_idx}' if args.total_chunks > 1 else f'{args.output_file}'
        torch.save(sequences, output_file)
        print(f'Saved {len(sequences)} sequences to {output_file}')

        if args.lvd:
            torch.save(save_messages, output_file + '.messages')

    

if __name__ == '__main__':
    # Run the async function
    asyncio.run(main())