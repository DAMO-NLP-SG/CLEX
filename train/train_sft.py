
import sys


import copy
from dataclasses import dataclass, field, asdict
import pathlib

import torch
import transformers
from transformers import Trainer, is_torch_tpu_available, AutoModelForCausalLM, AutoConfig
from transformers.trainer_pt_utils import LabelSmoother
from argument import *
from accelerate import Accelerator

import logging
import wandb
logger = logging.getLogger(__name__)
from utils import apply_chat_template, get_dataset
from trl import SFTTrainer


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
    if training_args.do_train:
        build_clex_args(config, model_args)

    model = AutoModelForCausalLM.from_pretrained(
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
    tokenizer.pad_token = tokenizer.eos_token
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    raw_datasets = get_dataset(data_args.data_path, splits=["train_sft", "test_sft"])

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"}, num_proc=40)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    # model.eval()

    # perplexity = calculate_perplexity(model, tokenizer, dataset.predict_dataset, 1, config)
    # print("Perplexity:", perplexity)

    logger.info("*** Model loaded! ***")

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.model_max_length,
        tokenizer=tokenizer,
        packing=True,
        num_of_sequences=64
    )


    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    accelerator = Accelerator()
    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tags": ["sft"],
        }
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
