"""Typed configuration for SMART-LLM.

Everything is a plain ``dataclass`` so the config is importable without any heavy
dependency (``yaml`` is imported lazily only when loading a file). A YAML file
overrides defaults field-by-field; CLI ``--set a.b.c=value`` overrides YAML.

Design contract used across the whole codebase:

* ``Config.paths`` holds every directory. Nothing else hard-codes paths.
* Feature extraction (Stage 1) reads ``Config.llm`` / ``Config.embedding`` /
  ``Config.retrieval`` / ``Config.data``.
* CDKA training (Stage 2) reads ``Config.probe`` / ``Config.rbe`` /
  ``Config.router`` and the cached features written by Stage 1.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    root: str = "runs"
    #: sub-directories are derived from ``root`` unless explicitly set.
    cache_dir: Optional[str] = None      # Stage-1 cached features
    index_dir: Optional[str] = None      # FAISS indexes
    results_dir: Optional[str] = None    # master CSVs / metrics
    figures_dir: Optional[str] = None
    tables_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    checkpoints_dir: Optional[str] = None  # probe / RBE / LoRA weights
    paper_dir: Optional[str] = None

    def resolve(self) -> "PathsConfig":
        """Fill unset sub-directories from ``root`` and create them."""
        root = Path(self.root)
        defaults = {
            "cache_dir": root / "cache",
            "index_dir": root / "index",
            "results_dir": root / "results",
            "figures_dir": root / "figures",
            "tables_dir": root / "tables",
            "logs_dir": root / "logs",
            "checkpoints_dir": root / "checkpoints",
            "paper_dir": root / "paper",
        }
        for name, default in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, str(default))
        return self

    def mkdirs(self) -> None:
        self.resolve()
        for name in ("cache_dir", "index_dir", "results_dir", "figures_dir",
                     "tables_dir", "logs_dir", "checkpoints_dir", "paper_dir"):
            Path(getattr(self, name)).mkdir(parents=True, exist_ok=True)


@dataclass
class LLMConfig:
    """Frozen backbone used for hidden extraction and verbalizer scoring."""
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    secondary_name: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"                 # bfloat16 | float16 | float32
    device_map: str = "auto"
    attn_implementation: str = "sdpa"       # sdpa | flash_attention_2 | eager
    trust_remote_code: bool = True
    max_input_tokens: int = 2048            # prompt truncation budget
    #: how many *last* tokens of hidden state to cache for attention pooling.
    #: last/mean pooling are computed on the fly; attention pooling needs tokens.
    cache_hidden_tokens: int = 64
    hidden_layer: int = -1                  # which layer's hidden state to pool
    batch_size: int = 1                     # keep 1 for long, ragged prompts
    load_in_4bit: bool = False              # optional bitsandbytes quantization


@dataclass
class EmbeddingConfig:
    name: str = "BAAI/bge-large-en-v1.5"
    alternative_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: Optional[int] = None               # auto-detected if None
    batch_size: int = 64
    normalize: bool = True
    #: bge-style retrieval instruction prepended to *queries* (not documents).
    query_instruction: str = "Represent this sentence for retrieving relevant documents: "
    device: str = "cuda"
    max_seq_length: int = 512


@dataclass
class RetrievalConfig:
    k: int = 8
    index_type: str = "flat_ip"             # flat_ip | ivf_flat | hnsw
    exclude_self: bool = True
    #: conditions used by Experiment 2 (retrieval robustness).
    conditions: List[str] = field(default_factory=lambda: ["clean", "random", "adversarial"])
    #: adversarial = nearest neighbours drawn from *other* classes (hard negatives).
    adversarial_strategy: str = "other_class_nn"   # other_class_nn | label_flip | shuffle_labels
    ivf_nlist: int = 256
    hnsw_m: int = 32


@dataclass
class DataConfig:
    dataset: str = "20newsgroups"
    seed: int = 20240517
    max_train: int = 4000                   # cap corpus / retrieval pool size
    max_eval: int = 1500                    # cap evaluation set
    val_fraction: float = 0.2               # of the eval pool, for calibration
    text_max_chars: int = 2000              # truncate raw text before tokenizing
    remove_headers: bool = True             # 20NG: strip headers/footers/quotes
    #: number of in-context demonstrations shown in the RETRIEVAL prompt.
    #: (== retrieval.k unless overridden.)
    n_demos: Optional[int] = None
    label_subset: Optional[List[str]] = None  # optionally restrict classes


@dataclass
class PoolingConfig:
    #: Experiment 1 compares these three.
    types: List[str] = field(default_factory=lambda: ["last", "mean", "attention"])
    default: str = "mean"
    attention_hidden: int = 256             # attn-pool bottleneck (learned)


@dataclass
class ProbeConfig:
    """Lightweight confidence probe  C_i = max(softmax(W_p h_L))."""
    lr: float = 1e-3
    epochs: int = 60
    weight_decay: float = 1e-4
    dropout: float = 0.1
    batch_size: int = 128
    #: temperature scaling of the probe logits (fit on val) for calibrated C_i.
    temperature_scale: bool = True
    hidden_dims: List[int] = field(default_factory=list)  # [] => linear probe


@dataclass
class RBEConfig:
    """Retrieval Benefit Estimator  B_pred = RBE([h_L || mu_K])."""
    hidden_dims: List[int] = field(default_factory=lambda: [512, 128])
    lr: float = 1e-3
    epochs: int = 120
    weight_decay: float = 1e-4
    dropout: float = 0.1
    batch_size: int = 128
    #: clip/normalise B_true target; huber loss is robust to outliers.
    loss: str = "huber"                     # huber | mse
    target_clip: float = 3.0


@dataclass
class RouterConfig:
    """RUS = alpha*sim(x,K) + beta*B_pred ;  route if calibrated(RUS) - C_i > 0."""
    alpha: float = 0.5
    beta: float = 0.5
    calibration: str = "platt"              # platt | isotonic | temperature
    #: uncertainty  U(x) = lam*norm_entropy + (1-lam)*(1-C_i)
    lam_uncertainty: float = 0.5
    #: search alpha/beta on the validation split before freezing them.
    tune_alpha_beta: bool = True
    tune_grid: int = 11                     # grid points in [0,1] for alpha (beta=1-alpha)
    decision_threshold: float = 0.0         # route if delta_C > threshold


@dataclass
class UAASConfig:
    """Contribution 2 — Uncertainty-Aware Adapter Scaling."""
    r_min: int = 4
    r_max: int = 32
    lam: float = 0.5                        # same lambda symbol as U(x)
    #: static LoRA baselines to compare against.
    static_ranks: List[int] = field(default_factory=lambda: [4, 16, 32])
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    epochs: int = 3
    lr: float = 1e-4
    batch_size: int = 8
    #: number of discrete rank buckets for the adaptive scheduler.
    rank_buckets: List[int] = field(default_factory=lambda: [4, 8, 16, 24, 32])


@dataclass
class ExplainConfig:
    """Contribution 3 — attribution-guided explanation verification."""
    method: str = "integrated_gradients"    # integrated_gradients | input_x_gradient
    ig_steps: int = 32
    ig_internal_batch: int = 8              # IG interpolation batch (caps GPU memory)
    dtype: str = "bfloat16"                 # attribution dtype; fp32 7B won't fit 24GB
    top_k_tokens: int = 10                  # salient tokens considered "important"
    n_samples: int = 200                    # samples to run attribution on (cost)


@dataclass
class Config:
    experiment_name: str = "phase1a_cdka"
    seed: int = 20240517
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    rbe: RBEConfig = field(default_factory=RBEConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    uaas: UAASConfig = field(default_factory=UAASConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)

    # ---- convenience ----
    @property
    def n_demos(self) -> int:
        return self.data.n_demos if self.data.n_demos is not None else self.retrieval.k

    def prepare(self) -> "Config":
        self.paths.mkdirs()
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
def _merge_into_dataclass(obj: Any, overrides: dict) -> Any:
    """Recursively apply a (possibly nested) dict onto a dataclass instance."""
    if not is_dataclass(obj):
        return overrides
    valid = {f.name: f for f in fields(obj)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}. "
                           f"Valid keys: {sorted(valid)}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI ``--set`` overrides."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [_coerce(v.strip()) for v in value.split(",")]
    return value


def _apply_dotted(cfg: Config, dotted: str) -> None:
    """Apply a single ``a.b.c=value`` override in place."""
    key, _, raw = dotted.partition("=")
    if not _:
        raise ValueError(f"Bad --set override '{dotted}', expected key=value")
    parts = key.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"Unknown config path '{key}'")
    setattr(obj, leaf, _coerce(raw))


def load_config(path: Optional[str] = None,
                overrides: Optional[List[str]] = None) -> Config:
    """Build a :class:`Config` from an optional YAML file and CLI overrides.

    Parameters
    ----------
    path:      YAML file whose (nested) keys override the dataclass defaults.
    overrides: list of ``a.b.c=value`` strings (from ``--set``).
    """
    cfg = Config()
    if path:
        import yaml  # lazy import
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge_into_dataclass(cfg, data)
    for dotted in overrides or []:
        _apply_dotted(cfg, dotted)
    return cfg.prepare()


def add_config_args(parser) -> None:
    """Attach ``--config`` and repeatable ``--set`` to an argparse parser."""
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="a.b.c=value",
                        help="Override a config value (repeatable).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Shortcut for --set data.dataset=<name>.")


def config_from_args(args) -> Config:
    overrides = list(getattr(args, "overrides", []) or [])
    if getattr(args, "dataset", None):
        overrides.append(f"data.dataset={args.dataset}")
    return load_config(getattr(args, "config", None), overrides)
