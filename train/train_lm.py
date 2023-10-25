
import sys


import copy
from dataclasses import dataclass, field, asdict
import pathlib

import torch
import transformers
from transformers import Trainer, is_torch_tpu_available, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from argument import *


import logging
import wandb
logger = logging.getLogger(__name__)




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

def build_clex_args(config, model_args):
    config.log_scale = model_args.log_scale
    config.use_flashattn = model_args.use_flashattn
    config.rope_scaling = {
        "type": model_args.scaling_type,
        "max_factor": model_args.max_factor,
        "param_factor": model_args.param_factor,
        "factor": 1
    }
    


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank



    from CLEX import LlamaForCausalLM, CLEXLlamaConfig


    config = CLEXLlamaConfig.from_pretrained(
        model_args.model_name_or_path
    )
    build_clex_args(config, model_args)

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    from dataset import WikiDataset
    dataset = WikiDataset(tokenizer=tokenizer, model_args=model_args, data_args=data_args, training_args=training_args)
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)


    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics if training_args.do_predict and not is_torch_tpu_available() else None,
        data_collator=dataset.data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_predict and not is_torch_tpu_available()
        else None,
    )
    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_dataset = dataset.predict_dataset
        predictions= trainer.predict(predict_dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predictions.metrics)
        trainer.save_metrics("predict", predictions.metrics)


if __name__ == "__main__":
    train()
