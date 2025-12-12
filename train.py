import traceback
import io
import os
import copy
import re
import sys
import random
import json
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from multiprocessing import cpu_count
from datasets import load_dataset
from tqdm import tqdm
import psutil
import pickle
import torch
import transformers
from torch.utils.data import Dataset, IterableDataset
from datasets.iterable_dataset import IterableDataset
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import torch.nn.functional as F

from transformers import AutoTokenizer, HfArgumentParser, set_seed

from qwen3.modeling_qwen3_refusion import Qwen3ForCausalLM


IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="model_name_or_path")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_size: int = field(default=None, metadata={"help": "for calculate max steps."})
    gpu_size: int = field(default=None, metadata={"help": "for calculate max steps and for logging for calcuated intervel."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    modules_to_save: str = field(
        default=None,
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root,f)
            yield fullname

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        data_paths = data_path.split(',')
        list_data_dict = load_dataset('json', data_files=data_paths, split='train', streaming=True)

        input_ids = []
        prompt_lengths = []
        attention_mask = []
        count = 0

        for idx, example in enumerate(list_data_dict):
            prompt = example["query"]
            target = example["response"]

            messages = [
                {"role": "user", "content": prompt}
            ]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )

            input_id = inputs

            end_id = tokenizer.encode(text=target) + [tokenizer.eos_token_id]

            if len(input_id) + len(end_id) > tokenizer.model_max_length:
                continue
            
            prompt_length = len(input_id)
            input_id.extend(end_id)
                
            input_ids.append(torch.tensor(input_id))
            prompt_lengths.append(torch.tensor(prompt_length))
            attention_mask.append(torch.tensor([1] * len(input_id)))

            count += 1
            if count % 1000 == 0:
                print(f"Count reached: {count}")
        
        print(f"Number of items in the dataset: {count}")

        self.input_ids = input_ids
        self.labels = input_ids.copy()
        self.prompt_lengths = prompt_lengths
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], prompt_lengths=self.prompt_lengths[i], attention_mask=self.attention_mask[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, prompt_lengths, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "prompt_lengths", "attention_mask"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        prompt_lengths = torch.stack(prompt_lengths).view(-1, 1)

        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    training_args.remove_unused_columns = False
    config.use_cache = False
    config._attn_implementation = "flash_attention_2"

    # qwen3
    config.bos_token_id = 151669
    config.mask_token_id = 151670

    model = Qwen3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
    )

    new_tokens = {
        "additional_special_tokens": ["<|beginoftext|>", "<|mask|>"]
    }

    tokenizer.add_special_tokens(new_tokens)

    model.resize_token_embeddings(len(tokenizer))

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.low_rank_training:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    logging.warning(f"data_module: {data_module}")
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()