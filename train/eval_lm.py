
import sys


import copy
from dataclasses import dataclass, field, asdict
import pathlib

import torch
import transformers
from transformers import Trainer, is_torch_tpu_available, AutoModelForCausalLM, AutoConfig
from transformers.trainer_pt_utils import LabelSmoother
from argument import *


import logging
import wandb
logger = logging.getLogger(__name__)
from utils import calculate_perplexity



IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import os
os.environ["WANDB_MODE"] = "disabled"



local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def build_clex_args(config, model_args):
    config.log_scale = model_args.log_scale
    config.use_flashattn = model_args.use_flashattn
    config.rope_scaling = {
        "type": model_args.scaling_type,
        "max_factor": model_args.max_factor,
        "param_factor": model_args.param_factor,
        "act": model_args.clex_act,
        "factor": 1
    }
    


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        trust_remote_code=True,
        use_flash_attention_2=model_args.use_flashattn,
        torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    from dataset import create_datasets
    _, valid_dataset = create_datasets(tokenizer, training_args, data_args.data_path)

    perplexity = calculate_perplexity(model, tokenizer, valid_dataset, training_args.model_max_length, 1, model_args.model_name_or_path)
    print("Perplexity:", perplexity)


if __name__ == "__main__":
    train()
