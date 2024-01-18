import random
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.utils.data import IterableDataset
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import warnings
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
import os
import numpy as np

from transformers import Trainer
import os
import re
from typing import List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk, Dataset
from datasets.builder import DatasetGenerationError
import json


def get_dataset(path, splits):
    raw_datasets = DatasetDict()
    for split in splits:
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(path, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(path, split))
        if "train" in split:
            raw_datasets["train"] = dataset
        if "test" in split:
            raw_datasets["test"] = dataset
    return raw_datasets

def apply_chat_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "dpo"] = "sft", assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        # print(example.keys())
        # print("--------------------")
        # print("using dpo")
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
            example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example

def calculate_perplexity(model, tokenizer, dataset, eval_len, batch_size, model_path, chunk_size=4096):

    total_ppl = 0
    total_tokens = 0
    total_loss = 0.0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    from tqdm import tqdm
    print("Begin Prediction")
    count = 0
    for batch in tqdm(dataloader):
        count += 1
        past = None
        bs, seq_len = batch["input_ids"].size()
        if seq_len != eval_len: continue
        batch_loss = 0.0
        batch_tokens = 0
        d = chunk_size
        for i in range(0, seq_len, d):
            input_ids = batch["input_ids"][:, i:i+d]
            input_ids = input_ids.to(device="cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past, use_cache=True, return_dict=True, labels=input_ids.clone())


            loss = outputs.loss
            
            chunk_tokens = input_ids.ne(tokenizer.pad_token_id).sum().item()
            if i % 1024 == 0:
                print(f"Length {i + d} Loss: ", loss.item())
            batch_loss += loss.item() * chunk_tokens
            batch_tokens += chunk_tokens
            # Store the past key and value for the next iteration
            past = outputs.past_key_values
            # except:
            #     break
        total_loss += batch_loss
        total_tokens += batch_tokens
        print("Chunk Loss:", batch_loss/batch_tokens)
    average_ppl = np.exp(total_loss / total_tokens)
    with open(f"{model_path}/test_{batch_tokens}.json", "w") as f:
        json.dump({"ppl": average_ppl}, f)
    return average_ppl