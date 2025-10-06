from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class DataConfig:
    data_dir: str
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    max_candidates: int = 40
    max_length: int = 512
    num_workers: int = 2
    sample_frac: Optional[float] = None
    relevant_top_k: int = 3


@dataclass
class OptimConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 1


@dataclass
class SupervisedConfig:
    model_id: str = "nomic-ai/modernbert-base"
    output_dir: str = "outputs/sft"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Sequence[str] = ("query", "key", "value")
    batch_size: int = 8
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 400
    mixed_precision: Optional[str] = "bf16"
    gradient_checkpointing: bool = False


@dataclass
class PPOConfig:
    sft_checkpoint: str
    output_dir: str = "outputs/ppo"
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_loss_coef: float = 0.1
    kl_penalty: float = 0.02
    target_kl: Optional[float] = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    rollout_samples: int = 512
    temperature: float = 0.7


@dataclass
class EvalConfig:
    checkpoint: str
    split: str = "val"
    batch_size: int = 8
    max_eval_samples: Optional[int] = None


@dataclass
class TeacherConfig:
    model_id: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 1
    system_prompt: str = (
        "You are an expert support agent. Rank the candidate passages from most to least relevant "
        "to answer the user query. Return the ranking as a list of indices."
    )
