"""Configuration schema for the multimodal LLM.

One dataclass tree, loaded from YAML. The same schema drives the 6M-param CPU
smoke run and the 3B-param 300k-context target run; only the values change.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional

import yaml

# Layer mixer kinds. The 300k context budget is only affordable because most
# layers are O(L) SSD rather than O(L^2) attention.
LayerKind = Literal["ssd", "swa", "global"]


@dataclass
class VisionConfig:
    """3D patch ViT shared by images (T=1) and video (T>1)."""
    enabled: bool = True
    image_size: int = 224          # training crop; inference is native-res
    patch_size: int = 14           # spatial patch edge in pixels
    temporal_patch: int = 2        # frames per patch (video); images are padded to this
    dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    merge_factor: int = 2          # pixel-shuffle: merges merge_factor^2 patches -> 1 token
    max_frames: int = 64           # per clip, before the LM sees it
    rope_theta: float = 10000.0

    @property
    def tokens_per_image(self) -> int:
        g = self.image_size // self.patch_size
        return (g // self.merge_factor) ** 2

    @property
    def tokens_per_frame(self) -> int:
        # temporal_patch frames collapse into one patch-row
        return self.tokens_per_image // self.temporal_patch


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    dim: int = 512
    n_layers: int = 12

    # --- attention (used by "swa" and "global" layers) ---
    n_heads: int = 8
    n_kv_heads: int = 2            # GQA. Drives the 300k KV-cache bill directly.
    head_dim: Optional[int] = None  # defaults to dim // n_heads
    window: int = 4096             # sliding-window span for "swa" layers
    rope_theta: float = 500000.0   # large base -> extrapolates to long context
    rope_scaling: Optional[float] = None  # NTK/YaRN factor applied at extension time

    # --- SSD / Mamba-2 mixer (used by "ssd" layers) ---
    ssd_heads: int = 8
    ssd_head_dim: int = 64
    ssd_state: int = 64            # N, state dimension
    ssd_groups: int = 1            # B/C sharing groups
    ssd_chunk: int = 128           # chunked-scan chunk length
    ssd_expand: int = 2
    ssd_conv: int = 4              # short depthwise conv before the scan

    # --- pattern: which mixer per layer ---
    # Repeated to cover n_layers. Default = Jamba/Nemotron-H style hybrid.
    layer_pattern: List[LayerKind] = field(
        default_factory=lambda: ["ssd", "ssd", "swa", "ssd", "ssd", "global"]
    )

    mlp_ratio: float = 4.0         # SwiGLU hidden = mlp_ratio * dim * 2/3, rounded
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    z_loss: float = 1e-4           # logit stabiliser; cheap divergence insurance

    max_seq_len: int = 4096        # training length. Extension phase raises this.
    vision: VisionConfig = field(default_factory=VisionConfig)

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.ssd_heads % self.ssd_groups == 0, "ssd_heads must be divisible by ssd_groups"

    def kinds(self) -> List[LayerKind]:
        p = self.layer_pattern
        return [p[i % len(p)] for i in range(self.n_layers)]


@dataclass
class DataConfig:
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    tokenizer: str = "data/tokenizer.json"
    dtype: str = "uint16"          # uint16 up to 65535 vocab, else uint32
    # Sampling weights over source shards, applied at pack time.
    mixture: dict = field(default_factory=lambda: {"text": 0.5, "code": 0.5})


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    seq_len: int = 1024
    micro_batch: int = 2
    grad_accum: int = 8            # global tokens/step = seq_len*micro_batch*grad_accum
    max_steps: int = 2000

    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.02
    schedule: Literal["wsd", "cosine"] = "wsd"
    decay_frac: float = 0.15       # WSD: final fraction spent decaying

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    dtype: Literal["float32", "bfloat16"] = "float32"
    compile: bool = False
    grad_checkpoint: bool = False

    log_every: int = 10
    eval_every: int = 200
    eval_iters: int = 20
    ckpt_every: int = 500
    seed: int = 1337
    device: str = "cpu"
    num_threads: int = 0           # 0 = leave torch default


@dataclass
class Config:
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        vis = VisionConfig(**(raw.get("model", {}).pop("vision", {}) or {}))
        model = ModelConfig(**{**raw.get("model", {}), "vision": vis})
        return Config(
            name=raw.get("name", "default"),
            model=model,
            data=DataConfig(**(raw.get("data", {}) or {})),
            train=TrainConfig(**(raw.get("train", {}) or {})),
        )

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
