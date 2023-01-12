from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import diskcache

from .utils import Config as ConfigBase


@dataclass(frozen=True)
class Config(ConfigBase):
    data_root: Path = Path("data")
    data_dirs: list[Path] = field(default_factory=lambda: [])
    test_data_dirs: list[Path] = field(default_factory=lambda: [])

    @property
    def sample_rate(self):
        return 24_000

    p_additional_prompt: float = 0.5

    token_dim: int = 256
    num_tokens: int = 1024

    nj: int = 8
    batch_size: int = 128
    eval_batch_size: int = 512
    warmup_min_lr: float = 1e-6
    warmup_max_lr: float = 2e-4
    dis_warmup_max_lr: float = 4e-4
    warmup_num_steps: int = 1_000
    max_iter: int = 10_000
    gradient_clipping: float = 100
    eval_every: int = 2_000
    save_ckpt_every: int = 10_000

    model: str = "ar-quarter"
    spkr_name_getter: str = "lambda p: p.parts[-2]"

    @cached_property
    def get_spkr(self):
        return eval(self.spkr_name_getter)

    @property
    def ds_cfg(self):
        return {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "Adam"},
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": self.warmup_min_lr,
                    "warmup_max_lr": self.warmup_max_lr,
                    "warmup_num_steps": self.warmup_num_steps,
                    "total_num_steps": self.max_iter,
                    "warmup_type": "linear",
                },
            },
            "gradient_clipping": self.gradient_clipping,
        }

    @property
    def cache_dir(self):
        return ".cache" / self.relpath

    @cached_property
    def diskcache(self):
        return diskcache.Cache(self.cache_dir).memoize


cfg = Config.from_cli()

if __name__ == "__main__":
    print(cfg)
