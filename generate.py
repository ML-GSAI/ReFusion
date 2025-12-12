import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os
import random
import copy
import math
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from qwen3.modeling_qwen3_refusion import Qwen3ForCausalLM
from qwen3.diffusion_cache_utils import DiffusionDynamicCache

from typing import Optional, Dict, Any, Tuple, List

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


import torch
import torch.nn.functional as F


@ torch.no_grad()
def generate_refusion(model, tokenizer, prompt, gen_length=128, temperature=0., mask_id=151670, slot_size=8,
             model_path='', serial_num_blocks=2, slot_threshold=0.9, token_threshold=0.9):

    slot_threshold = slot_threshold
    token_threshold = token_threshold
    sum_TPF = 0.0
    forward_count = 0

    eos_token_id = tokenizer.eos_token_id
    batch_size = 1
    prompt_len = prompt.shape[1]
    device = model.device

    gen_pad_len = (slot_size - (gen_length % slot_size)) % slot_size
    gen_length = gen_length + gen_pad_len
    gen_x = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=device)

    prompt_pos_ids = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0)
    gen_pos_ids = torch.arange(prompt_len, prompt_len + gen_length, dtype=torch.long, device=device).unsqueeze(0)

    cur_x = prompt.clone()
    cur_pos = prompt_pos_ids.clone()

    cur_slot_size = slot_size

    eos_flag = False
    block_length = gen_length // serial_num_blocks

    past_key_values = None


    for serial_num_block in range(serial_num_blocks):

        # block level
        cur_gen_x = gen_x[:, serial_num_block*block_length:(serial_num_block+1)*block_length] # (batch_size, block_length)
        cur_gen_pos_ids = gen_pos_ids[:, serial_num_block*block_length:(serial_num_block+1)*block_length] # (batch_size, block_length)

        cur_gen_blocks_x = cur_gen_x.reshape(batch_size, -1, cur_slot_size)
        cur_gen_blocks_pos_ids = cur_gen_pos_ids.reshape(batch_size, -1, cur_slot_size)

        # slot level generation
        while cur_gen_blocks_x.numel() > 0:
            cur_gen_blocks_x = cur_gen_blocks_x.reshape(batch_size, -1, cur_slot_size)
            cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids.reshape(batch_size, -1, cur_slot_size)

            flat_gen_blocks_x = cur_gen_blocks_x.view(batch_size, -1)
            flat_gen_blocks_pos_ids = cur_gen_blocks_pos_ids.view(batch_size, -1)

            prefix_block_tag = False
            
            # MDM
            if past_key_values is None:
                input_x = torch.cat((cur_x, flat_gen_blocks_x), dim=1)
                input_pos_ids = torch.cat((cur_pos, flat_gen_blocks_pos_ids), dim=1)
                outputs = model(
                    input_ids=input_x,
                    position_ids=input_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                outputs = model(
                    input_ids=flat_gen_blocks_x,
                    position_ids=flat_gen_blocks_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            logits = outputs.logits

            gen_logits = logits[:, -flat_gen_blocks_x.shape[1]:, :]

            past_key_values = outputs.past_key_values
            past_key_values.crop(cur_x.shape[1])
            assert cur_x.shape[-1] == past_key_values[0][0].shape[-2]

            logits_with_noise = add_gumbel_noise(gen_logits, temperature=temperature)
            x0_gen = torch.argmax(logits_with_noise, dim=-1)
            x0_gen_blocks = x0_gen.view(batch_size, -1, cur_slot_size)

            p_softmax = F.softmax(gen_logits, dim=-1)
            x0_p_softmax = torch.gather(p_softmax, dim=-1, index=torch.unsqueeze(x0_gen, -1)).squeeze(-1)
            
            x0_p_softmax_blocks = x0_p_softmax.view(batch_size, -1, cur_slot_size)
            block_confidence_softmax = x0_p_softmax_blocks[:,:,0] # (bsz, num_slots)
            
            is_confident_block = block_confidence_softmax > slot_threshold
            counts_block = torch.sum(is_confident_block, dim=1).item()
            topk_indices_relative = is_confident_block[0].nonzero(as_tuple=True)[0]

            if counts_block <= 0:
                counts_block = 1
                _, topk_indices_relative = torch.topk(block_confidence_softmax.squeeze(0), k=1)

            # choose slot
            topk_indices_relative, _ = torch.sort(topk_indices_relative)

            chosen_gen_blocks = x0_gen_blocks[0, topk_indices_relative, :]
            chosen_position_ids = cur_gen_blocks_pos_ids[0, topk_indices_relative, :]
            chosen_p_softmax_blocks = x0_p_softmax_blocks[0, topk_indices_relative, :]


            # Global Verification            
            outputs = model(
                input_ids=chosen_gen_blocks.reshape(1, -1),
                position_ids=chosen_position_ids.reshape(1, -1),
                past_key_values=past_key_values,
                use_cache=True,
            )

            AR_logits = outputs.logits #[1, len, vocab_len]
            AR_logits = torch.cat([AR_logits[:,:1], AR_logits[:, :-1]], dim=1)
            AR_p_softmax = F.softmax(AR_logits, dim=-1) #[1, len, 1]
            AR_x0_p_softmax = torch.gather(AR_p_softmax, dim=-1, index=torch.unsqueeze(chosen_gen_blocks.reshape(1, -1), -1)).squeeze(-1) #[1, len]
            AR_x0_p_softmax_blocks = AR_x0_p_softmax.reshape(-1, cur_slot_size)
            chosen_p_softmax_blocks[:,1:] = AR_x0_p_softmax_blocks[:,1:]

            
            prob_mask = chosen_p_softmax_blocks > token_threshold
            prob_mask[:, 0] = 1
            tag_blocks = torch.cumprod(prob_mask.int(), dim=-1)

            tag_tokens = torch.cumprod(prob_mask.int().reshape(1, -1), dim=-1)
            prefix_len = torch.sum(tag_tokens, dim=-1)
            flat_chosen_gen_blocks = chosen_gen_blocks.reshape(1, -1)
            confident_prefix_tokens = flat_chosen_gen_blocks[:, :prefix_len]

            if prefix_len > 0:
                is_eos_in_prefix = (confident_prefix_tokens.squeeze(0) == eos_token_id)
                eos_found_flag = torch.any(is_eos_in_prefix)

                remain_indices = []

                indices_to_remove = set()

                if eos_found_flag:
                    first_eos_pos_tensor = torch.argmax(is_eos_in_prefix.int())

                    eos_block_pos = first_eos_pos_tensor // cur_slot_size + 1
                    eos_token_pos = first_eos_pos_tensor - (first_eos_pos_tensor // cur_slot_size) * cur_slot_size

                    eos_block = topk_indices_relative[eos_block_pos-1].item()

                    remain_indices.extend(topk_indices_relative[:eos_block_pos].tolist())
                    
                    topk_indices_relative = torch.tensor([], device=device)

                    eos_flag = True

                    indices_after_eos = list(range(eos_block, cur_gen_blocks_x.shape[1]))
                    indices_to_remove.update(indices_after_eos)

                elif (prefix_len // cur_slot_size) > 0:
                    num_prefix_blocks = prefix_len // cur_slot_size
                    remain_indices.extend(topk_indices_relative[:num_prefix_blocks].tolist())

                    topk_indices_relative = topk_indices_relative[num_prefix_blocks:]
                    tag_blocks = tag_blocks[num_prefix_blocks:]
            
                if len(remain_indices) > 0:

                    indices_to_remove.update(remain_indices)

                    token_indices = []

                    for i_idx, b_idx in enumerate(remain_indices):
                        start_index = b_idx * cur_slot_size
                        
                        current_block_len = cur_slot_size
                        # If EOS exists and this is the last slot, then adjust the length.
                        if eos_found_flag and i_idx == len(remain_indices) - 1:
                            current_block_len = eos_token_pos + 1
                            
                        
                        end_index = start_index + current_block_len
                        block_range = torch.arange(start_index, end_index, dtype=torch.long, device=device)
                        
                        token_indices.append(block_range)

                    full_token_indices = torch.cat(token_indices)

                    cur_x = torch.cat((cur_x, x0_gen[:, full_token_indices]), dim=1)
                    cur_pos = torch.cat((cur_pos, flat_gen_blocks_pos_ids[:, full_token_indices]), dim=1)

                    past_key_values = outputs.past_key_values
                    past_key_values.crop(cur_x.shape[1])
                    
                    assert cur_x.shape[-1] == past_key_values[0][0].shape[-2]

                    prefix_block_tag = True

                    sum_TPF += cur_slot_size * len(remain_indices) / 2
                    forward_count += 1

            if prefix_block_tag == True:
                keep_mask = torch.ones(cur_gen_blocks_x.shape[1], dtype=torch.bool, device=device)
                keep_mask[list(indices_to_remove)] = False
                cur_gen_blocks_x = cur_gen_blocks_x[:, keep_mask, :]
                cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids[:, keep_mask, :]

                continue

            elif prefix_block_tag == False:
                past_key_values = outputs.past_key_values
                past_key_values.crop(cur_x.shape[1])
                assert cur_x.shape[-1] == past_key_values[0][0].shape[-2]

            indices_to_remove = set(topk_indices_relative.tolist())

            current_speculative_blocks = chosen_gen_blocks.clone()
            accepted_prefix_len = 0
            eos_found_in_loop = False

            if past_key_values is not None and counts_block > 1:
                past_key_values.batch_repeat_interleave(counts_block)
            
            for loop_iter in range(cur_slot_size):
                if not torch.any(tag_blocks == 0):
                    break

                input_tokens = current_speculative_blocks[:, accepted_prefix_len:]
                input_pos = chosen_position_ids[:, accepted_prefix_len:]

                current_tags = tag_blocks[:, accepted_prefix_len:]
                masked_input_tokens = torch.where(current_tags.bool(), input_tokens, mask_id)

                # Prediction
                draft_len = past_key_values[0][0].shape[2]
                draft_outputs = model(
                    input_ids=masked_input_tokens,
                    position_ids=input_pos,
                    past_key_values=past_key_values,
                    use_cache=False,
                )
                past_key_values.crop(draft_len)
                draft_logits = draft_outputs.logits
                proposed_tokens = torch.argmax(draft_logits, dim=-1)

                input_tokens = torch.where(current_tags.bool(), input_tokens, proposed_tokens)
                current_speculative_blocks[:, accepted_prefix_len:] = input_tokens

                # Verification
                verify_outputs = model(
                    input_ids=input_tokens,
                    position_ids=input_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                verify_logits = verify_outputs.logits
                verify_logits = torch.cat([verify_logits[:,:1], verify_logits[:, :-1]], dim=1)
                
                verify_probs = F.softmax(verify_logits, dim=-1)
                gathered_probs = torch.gather(verify_probs, -1, input_tokens.unsqueeze(-1)).squeeze(-1)
                
                prob_mask = gathered_probs > token_threshold

                # Keep at least one token
                update_tag_blocks = F.pad(tag_blocks[:, accepted_prefix_len:], (1, 0), value=1)[:, :-1]

                prob_mask[update_tag_blocks == 1] = True

                new_tags = torch.cumprod(prob_mask.int(), dim=-1)
                tag_blocks[:, accepted_prefix_len:] = new_tags

                newly_verified_mask = (tag_blocks[:, accepted_prefix_len:] == 1)
                is_eos_in_new = (current_speculative_blocks[:, accepted_prefix_len:] == eos_token_id) & newly_verified_mask

                if torch.any(is_eos_in_new):
                    eos_found_in_loop = True
                    first_eos_block_idx = torch.where(torch.any(is_eos_in_new, dim=1))[0][0].item()
                    
                    current_speculative_blocks = current_speculative_blocks[:first_eos_block_idx+1]
                    tag_blocks = tag_blocks[:first_eos_block_idx+1]
                    tag_blocks[first_eos_block_idx] = 1
                    chosen_position_ids = chosen_position_ids[:first_eos_block_idx+1]
                    topk_indices_relative = topk_indices_relative[:first_eos_block_idx+1]
                    if verify_outputs.past_key_values is not None:
                         verify_outputs.past_key_values.batch_select_minibatch(first_eos_block_idx + 1)

                current_tags = tag_blocks[:, accepted_prefix_len:]
                len_per_block = torch.sum(current_tags, dim=1)
                newly_accepted_len = torch.min(len_per_block).item()
                if newly_accepted_len > 0:
                    if torch.any(tag_blocks == 0):
                        accepted_prefix_len = accepted_prefix_len + newly_accepted_len - 1 
                    else:
                        accepted_prefix_len = accepted_prefix_len + newly_accepted_len
                    past_key_values = verify_outputs.past_key_values
                    if past_key_values is not None:
                        past_key_values.crop(cur_x.shape[1] + accepted_prefix_len)
                    
            sum_TPF += (cur_slot_size * counts_block) / (loop_iter * 2 + 2)
            forward_count += 1

            ar_kv_cache = tuple(
                (
                    layer_past[0][:, :, -cur_slot_size:, :],  # key
                    layer_past[1][:, :, -cur_slot_size:, :]   # value
                )
                for layer_past in past_key_values
            )


            past_key_values.crop(cur_x.shape[1])
            past_key_values.batch_select_indices(torch.tensor([0]).to(device))

            eos_mask = (current_speculative_blocks == eos_token_id) # (k*cur_slot_size)
            keep_mask = (torch.cumsum(eos_mask.flatten().int(), dim=-1) - eos_mask.flatten().int()) == 0
            kept_tokens = current_speculative_blocks.flatten()[keep_mask].reshape(batch_size, -1)
            kept_pos_ids = chosen_position_ids.flatten()[keep_mask].reshape(batch_size, -1)

            # update KV cache
            if kept_tokens.numel() > 0 and ar_kv_cache is not None:
                new_past = []
                for i, (key, val) in enumerate(ar_kv_cache):
                    num_heads, _, head_dim = key.shape[1], key.shape[2], key.shape[3]
                    
                    flat_key = key.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)
                    flat_val = val.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)

                    kept_key = flat_key[:, :, keep_mask, :]
                    kept_val = flat_val[:, :, keep_mask, :]
                
                    new_past.append((kept_key, kept_val))

                kept_kv = tuple(new_past)

            past_key_values.full_update(kept_kv)

            cur_x = torch.cat((cur_x, kept_tokens), dim=1)
            cur_pos = torch.cat((cur_pos, kept_pos_ids), dim=1)

            assert cur_x.shape[-1] == past_key_values[0][0].shape[-2]

            if eos_found_in_loop:
                indices_after_eos = list(range(first_eos_block_idx, cur_gen_blocks_x.shape[1]))
                indices_to_remove.update(indices_after_eos)
                eos_flag = True

            keep_mask = torch.ones(cur_gen_blocks_x.shape[1], dtype=torch.bool, device=device)
            keep_mask[list(indices_to_remove)] = False
            cur_gen_blocks_x = cur_gen_blocks_x[:, keep_mask, :]
            cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids[:, keep_mask, :]

        if eos_flag:
            break

    _, re_mask_indices = torch.sort(cur_pos, dim=-1)
    x = torch.gather(cur_x, dim=-1, index=re_mask_indices)

    TPF = sum_TPF / forward_count

    return x, TPF



def main():
    device = 'cuda'

    model_path = "./output_checkpoints"
    model = Qwen3ForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "You are an expert Python programmer. Your task is to write a single Python function to solve the problem described below, and here is your task: Write a function to sum all amicable numbers from 1 to a specified number.\n\nDirectly after the '[BEGIN]' marker, you must write only the Python code for the function. Do not provide any explanations, comments, or introductory text. The function must include the 'def' line, its arguments, the function body, and a 'return' statement. Your code should pass these tests:\n\nassert amicable_numbers_sum(999)==504\nassert amicable_numbers_sum(9999)==31626\nassert amicable_numbers_sum(99)==0\n[BEGIN]\n"

    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=True)

    print(prompt)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out, TPF = generate_refusion(model, tokenizer, input_ids, gen_length=512, temperature=0., mask_id=151670, slot_size=4, model_path=model_path, serial_num_blocks=32, slot_threshold=0.6, token_threshold=0.3)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print("---------TPF:", TPF)


if __name__ == '__main__':
    main()