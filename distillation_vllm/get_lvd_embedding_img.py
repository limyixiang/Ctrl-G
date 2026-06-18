import torch
import torch.distributed as dist
from tqdm import tqdm
import os
import argparse
import re
import io
import base64
from safetensors.torch import save_file
from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence

# ----------------- Chat helpers -----------------
def _contains_image(chat_turns: List[Dict[str, Any]]) -> bool:
    for turn in chat_turns:
        content = turn.get("content")
        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "image_url":
                    return True
    return False


def _any_vl(messages: List[List[Dict[str, Any]]]) -> bool:
    return any(_contains_image(m) for m in messages)


def _extract_images_from_chat(chat: List[Dict[str, Any]]):
    """Decode images from data: URLs into PIL Images (RGB). Other URL types are skipped."""
    from PIL import Image
    images = []
    for turn in chat:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "image_url":
                url = (chunk.get("image_url") or {}).get("url", "")
                m = re.match(r"^data:(image/[^;]+);base64,(.+)$", url)
                if m:
                    _mime, b64 = m.groups()
                    img_bytes = base64.b64decode(b64)
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return images

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
    parser.add_argument("--min_pixels", type=int, default=None,
                        help="Minimum number of pixels for image processing. Need to align with the previous training step.")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Maximum number of pixels for image processing. Need to align with the previous training step.")
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
    messages = torch.load(f"{file_path}.messages")

    if rank == 0:
        print("Total number of sequences loaded:", len(output_tokens))
        print("Total number of messages loaded:", len(messages))
    
    # Limit data for debugging if requested
    if args.debug:
        if rank == 0:
            print("Debug mode: Processing only first 100 samples")
        output_tokens = output_tokens[:100]
        messages = messages[:100]
        # output_tokens = output_tokens[:20]
        # messages = messages[:20]

    # Load model
    from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.float32).to(device)

    total_samples = len(output_tokens)

    # # Check that the output part matches (only on rank 0 for efficiency)
    # if rank == 0:
    #     assert len(output_tokens) == len(full_tokens) == len(input_tokens), "Length mismatch between output tokens and full tokens"
        
    #     for full_token, input_token, output_token in zip(full_tokens, input_tokens, output_tokens):
    #         prompt_output = torch.cat([input_token, output_token], dim=0)
    #         assert torch.equal(prompt_output, full_token[:len(prompt_output)]), "input + output tokens do not match full tokens"
        
    #     print("All checks passed!")

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
    if args.min_pixels is not None or args.max_pixels is not None:
        print(f"Set min/max pixel values to {args.min_pixels} to {args.max_pixels}")
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    else:
        processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    output_len = output_tokens.size(1)
    
    with torch.no_grad():
        hidden_state_list = []
        local_samples = end_idx - start_idx
        
        if rank == 0:
            print(f"Rank {rank}: Processing {local_samples} samples in batches of {args.batch_size}...")
        
        for i in tqdm(range(start_idx, end_idx, args.batch_size), disable=rank != 0):
            batch_end = min(i + args.batch_size, end_idx)

            # Get input tokens
            batch_messages = messages[i:batch_end]
            batch_output_tokens = output_tokens[i:batch_end].to(device)

            full_ids_list, full_mask_list = [], []
            pixel_values_list, image_grid_thw_list = [], []
            original_prompt_lengths, original_lengths = [], []
            for chat, out_ids in zip(batch_messages, batch_output_tokens):
                # Get Image
                images = _extract_images_from_chat(chat)
                num_imgs = len(images)
                img_placeholders = [
                    {"type": "image", "image": f"image-{j+1}"} for j in range(num_imgs)
                ]
                # Get Prompt Text
                # chat item
                # [{
                #     'role': 'user', 
                #     'content': [
                #         {'type': 'text', 'text': 'Find x.\nChoices:\n(A) 120\n(B) 135\n(C) 145\n(D) 160 You FIRST think about the reasoning process as an internal monologue and then provide the final answer.'}
                #         {...}
                #     ]
                # }]


                text_item = None
                for content_item in chat[0]["content"]:
                    if content_item['type'] == 'text':
                        text_item = content_item
                        break
                assert text_item['type'] == 'text', "Got the wrong text item... Check your saved message file"
                
                chat_with_img_placeholder = [
                    {
                        "role": "user",
                        "content": img_placeholders + [text_item],
                    }
                ]

                prompt_text = processor.apply_chat_template(
                    chat_with_img_placeholder, add_generation_prompt=True, tokenize=False
                )
                
                
                # Log
                if i == 0 and full_ids_list == []: # Print the very first one
                    print("----------------- After applying chat template with image placeholder")
                    print(prompt_text)
                    print("----------------- Done")
                
                # Proc
                proc = processor(text=prompt_text,
                                 images=images if len(images) > 0 else None,
                                 return_tensors="pt")
                input_ids = proc["input_ids"][0]         # [P]
                attn_mask = proc["attention_mask"][0]    # [P]
                pixel_values = proc.get("pixel_values")     # [patch, dim], need to concat first dim
                image_grid_thw  = proc.get("image_grid_thw")      # [1, 3]  [batch, (Frame, height, width)]

                print("---------------------")
                print(input_ids.size())
                print(attn_mask)
                print(pixel_values.size())
                print(image_grid_thw)
                print(tokenizer.decode(input_ids.cpu().tolist(), skip_special_tokens=False))
                print("---------------------")
                
                # Trim prompt right-padding (usually none for single-sample, but safe)
                P = int(attn_mask.sum().item())
                input_ids = input_ids[:P]
                attn_mask = attn_mask[:P]
                
                # Append gold/generated tokens (no decode/re-tokenize!)
                out_ids = out_ids.to(input_ids.device)
                full_ids  = torch.cat([input_ids, out_ids], dim=0)
                full_mask = torch.cat([attn_mask, torch.ones_like(out_ids)], dim=0)
                
                
                print("full_ids: ", tokenizer.decode(full_ids.cpu().tolist(), skip_special_tokens=False))
                # print(tokenizer.decode(full_ids, skip_special_tokens=False))

                full_ids_list.append(full_ids)
                full_mask_list.append(full_mask)
                
                # Length
                original_prompt_lengths.append(len(input_ids))
                original_lengths.append(len(input_ids) + len(out_ids)) #TODO

                if pixel_values is not None:
                    pixel_values_list.append(pixel_values)  # [3, H, W]
                if image_grid_thw is not None:
                    image_grid_thw_list.append(image_grid_thw[0])    # [2]
                # input_ids = proc_inputs["input_ids"].to(device)  # [1, P]
                # print(input_ids.size(), output_ids.size())
                # full_ids = torch.cat([input_ids, output_ids.unsqueeze(0)], dim=1)  # [1, P+L]
                # batch_input_ids.append(input_ids[0])
                # batch_token_ids.append(full_ids[0])
            
            
            # Final pad (once) for the full sequences
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            batch_input_ids = pad_sequence(full_ids_list,  batch_first=True, padding_value=pad_id).to(device)
            batch_attn_mask = pad_sequence(full_mask_list, batch_first=True, padding_value=0).to(device)

            print(tokenizer.decode(batch_input_ids[0].cpu().tolist(), skip_special_tokens=False))

            model_kwargs = dict(
                input_ids=batch_input_ids,
                attention_mask=batch_attn_mask,
                output_hidden_states=True,
            )

            

            # If every sample has an image, stack and pass them
            if pixel_values_list:
                model_kwargs["pixel_values"] = torch.concat(pixel_values_list, dim=0).to(device)
            if image_grid_thw_list:
                model_kwargs["image_grid_thw"]  = torch.stack(image_grid_thw_list,  dim=0).to(device)

            print(input_ids.size(), attn_mask.size())
            print(model_kwargs["pixel_values"].size(), model_kwargs["image_grid_thw"])

            with torch.no_grad():
                outputs = model(**model_kwargs)
            # Store original lengths for proper indexing
            # original_lengths = [seq.size(0) for seq in batch_token_ids]
            # original_prompt_lengths = [seq.size(0) for seq in batch_input_ids]
            
            # Add padding for sequences of different lengths
            # Pad sequences to the same length (right padding with tokenizer.pad_token_id)
            # pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            # print(len(batch_token_ids), batch_token_ids[0].size())
            # batch_token_ids = pad_sequence(batch_token_ids, batch_first=True, padding_value=pad_token_id).to(device)
            
            # # Create attention mask for proper model processing
            # attention_mask = torch.zeros_like(batch_token_ids)
            # for j, orig_len in enumerate(original_lengths):
            #     attention_mask[j, :orig_len] = 1

            
            # Forward pass for this batch
            # outputs = model(input_ids=batch_token_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Get the last layer hidden states for this batch
                last_hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
                for j, hidden_state in enumerate(last_hidden_states):
                    prompt_len = original_prompt_lengths[j]
                    # Extract hidden states for the output tokens (right after prompt)
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
