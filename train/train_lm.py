
import sys


import copy
from dataclasses import dataclass, field, asdict
import pathlib

import torch
import transformers
from transformers import Trainer, is_torch_tpu_available, AutoModelForCausalLM, AutoConfig
from transformers.trainer_pt_utils import LabelSmoother
from argument import *
from CLEX import LlamaForCausalLM, CLEXLlamaConfig, MixtralForCausalLM, CLEXMixtralConfig, PhiForCausalLM, CLEXPhiConfig


import logging
import wandb
logger = logging.getLogger(__name__)




IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import os
os.environ["WANDB_MODE"] = "disabled"



local_rank = None


def get_model_config_class(model_name_or_path):
    if "llama" in model_name_or_path.lower():
        return CLEXLlamaConfig, LlamaForCausalLM
    elif "mixtral" in model_name_or_path.lower():
        return CLEXMixtralConfig, MixtralForCausalLM
    elif "phi" in model_name_or_path.lower():
        return CLEXPhiConfig, PhiForCausalLM
    else:
        raise ValueError(f"Unknown model class from the path: {model_name_or_path}")
    


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
        "factor": 1,
        "time_dt": model_args.time_dt
    }
    


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank


    CONFIG_CLASS, MODEL_CLASS = get_model_config_class(model_args.model_name_or_path)

    config = CONFIG_CLASS.from_pretrained(
        model_args.model_name_or_path
    )
    if training_args.do_train:
        build_clex_args(config, model_args)

    model = MODEL_CLASS.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        trust_remote_code=True,
        use_flash_attention_2=model_args.use_flashattn,
        torch_dtype=torch.bfloat16,
        _fast_init=False
    )
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    from dataset import create_datasets
    train_dataset, valid_dataset = create_datasets(tokenizer, training_args, data_args.data_path)

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
