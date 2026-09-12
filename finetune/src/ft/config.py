"""Fine-tuning configuration.

One schema for every scale: the 135M CPU run on this laptop and a 7B LoRA run on
a rented A100 differ only in values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

import yaml


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32              # scaling = alpha / r; alpha = 2r is the usual default
    dropout: float = 0.05
    # Llama-family projection names — covers SmolLM2, Qwen2.5/3, Llama, Mistral, Gemma.
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    # Training the embedding/head is what you need for genuinely NEW vocabulary
    # (new domain jargon, new special tokens). Off by default: it is most of the
    # trainable-parameter budget and usually unnecessary.
    train_embeddings: bool = False


@dataclass
class DataConfig:
    train_file: str = "data/train.jsonl"
    val_file: str = "data/val.jsonl"
    max_len: int = 1024
    # Examples longer than max_len are DROPPED, not truncated. Truncating mid-answer
    # teaches the model to stop mid-sentence, which is worse than losing the example.
    on_overflow: Literal["drop", "truncate"] = "drop"
    mask_prompt: bool = True     # loss on assistant turns only


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    epochs: float = 3.0
    micro_batch: int = 1
    grad_accum: int = 8
    lr: float = 2.0e-4           # LoRA wants ~10-50x the LR of a full fine-tune
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.03
    weight_decay: float = 0.0    # LoRA adapters are small; decay mostly just hurts
    grad_clip: float = 1.0
    dtype: Literal["float32", "bfloat16"] = "float32"
    device: str = "cpu"
    num_threads: int = 0
    seed: int = 1337
    log_every: int = 5
    eval_every: int = 50
    eval_batches: int = 20
    save_every: int = 100


@dataclass
class Config:
    name: str = "default"
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def from_dict(raw: dict) -> "Config":
        """Rebuild from asdict() output. Checkpoints store the config this way, so
        every sub-config must be reconstructed — leaving one as a plain dict fails
        only later, at the point of use."""
        return Config(
            name=raw.get("name", "default"),
            base_model=raw.get("base_model", "HuggingFaceTB/SmolLM2-135M-Instruct"),
            lora=LoRAConfig(**(raw.get("lora") or {})),
            data=DataConfig(**(raw.get("data") or {})),
            train=TrainConfig(**(raw.get("train") or {})),
        )

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Config.from_dict(raw)

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
