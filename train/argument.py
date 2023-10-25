from dataclasses import dataclass, field, asdict
import transformers
from typing import Dict, Optional, Sequence



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    log_scale: Optional[bool] = field(default=False)
    use_flashattn: Optional[bool] = field(default=True)
    scaling_type: Optional[str] = field(default="clex")
    max_factor: int = field(
        default=16,
        metadata={
            "help": "The maximum sampling value of scaling factor."
        },
    )
    param_factor: int = field(
        default=1,
        metadata={
            "help": "The ODE up projection factor."
        },
    )



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="/mnt/workspace/tmp")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )