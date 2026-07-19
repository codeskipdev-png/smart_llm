"""Frozen open-weight LLM used as a verbalizer classifier + hidden extractor.

One forward pass yields, for a given input:
  * a class distribution  P(y|x)  (softmax over option-letter logits),
  * the classification loss  -log P(y*|x),
  * (optionally) pooled/token hidden states for CDKA.

The backbone weights are never updated. On the GPU box this loads
``Qwen/Qwen2.5-7B-Instruct`` in bf16 (~16 GB); ``load_in_4bit`` shrinks it if
needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import Config
from ..utils.device import resolve_dtype
from ..utils.logging import get_logger
from .prompts import VerbalizerSpec, build_classification_messages

_log = get_logger("smart_llm.llm")


@dataclass
class LLMOutput:
    """Per-sample result (numpy so it can be cached/serialised directly)."""
    probs: np.ndarray            # [n_classes] verbalizer distribution
    loss: float                  # cross-entropy of the true label
    pred: int
    confidence: float            # max softmax (LLM verbalizer confidence)
    entropy: float               # normalised entropy in [0, 1]
    # hidden features (only when want_hidden=True; else None)
    h_last: Optional[np.ndarray] = None      # [dim]
    h_mean: Optional[np.ndarray] = None      # [dim]
    h_tokens: Optional[np.ndarray] = None    # [N, dim] last-N window (front-padded)
    token_mask: Optional[np.ndarray] = None  # [N] 1 for real tokens
    n_prompt_tokens: int = 0


class FrozenLLM:
    def __init__(self, cfg: Config, model_name: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        name = model_name or cfg.llm.name
        self.name = name
        dtype = resolve_dtype(cfg.llm.dtype)

        _log.info("Loading frozen LLM %s (dtype=%s)…", name, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, trust_remote_code=cfg.llm.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(
            torch_dtype=dtype,
            device_map=cfg.llm.device_map,
            trust_remote_code=cfg.llm.trust_remote_code,
            attn_implementation=cfg.llm.attn_implementation,
        )
        if cfg.llm.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kwargs.pop("torch_dtype", None)

        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.hidden_size = int(self.model.config.hidden_size)
        cfg.llm_hidden_size = self.hidden_size  # type: ignore[attr-defined]
        self.input_device = self.model.get_input_embeddings().weight.device
        _log.info("LLM ready. hidden_size=%d device=%s", self.hidden_size,
                  self.input_device)

    # ------------------------------------------------------------------ #
    def option_token_ids(self, letters: List[str]) -> List[List[int]]:
        """Variant token ids per option letter (with & without leading space).

        We marginalise over these variants when scoring, so template whitespace
        does not bias the class distribution.
        """
        groups = []
        for letter in letters:
            ids = set()
            for variant in (letter, " " + letter):
                enc = self.tokenizer.encode(variant, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            groups.append(sorted(ids))
        return groups

    # ------------------------------------------------------------------ #
    def _tokenize(self, messages) -> "tuple":
        torch = self.torch
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt")
        # left-truncate to keep the answer scaffold (options + request) at the end
        max_t = self.cfg.llm.max_input_tokens
        if input_ids.shape[1] > max_t:
            input_ids = input_ids[:, -max_t:]
        input_ids = input_ids.to(self.input_device)
        attn = torch.ones_like(input_ids)
        return input_ids, attn

    def classify(self, text: str, verbalizer: VerbalizerSpec,
                 demo_block: Optional[str] = None,
                 true_label: Optional[int] = None,
                 want_hidden: bool = False,
                 task_hint: Optional[str] = None) -> LLMOutput:
        torch = self.torch
        messages = build_classification_messages(
            text, verbalizer, demo_block=demo_block, task_hint=task_hint)
        input_ids, attn = self._tokenize(messages)
        groups = self.option_token_ids(verbalizer.letters)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn,
                             output_hidden_states=want_hidden, use_cache=False)
        # answer-position logits = last position predicts the first answer token
        logits = out.logits[0, -1, :].float()                       # [vocab]

        # per-class score = logsumexp over that class's variant token logits
        class_scores = torch.stack([
            torch.logsumexp(logits[torch.tensor(g, device=logits.device)], dim=0)
            if g else torch.tensor(float("-inf"), device=logits.device)
            for g in groups
        ])
        log_probs = torch.log_softmax(class_scores, dim=0)
        probs = log_probs.exp()
        n = len(verbalizer.letters)

        pred = int(torch.argmax(probs).item())
        conf = float(probs.max().item())
        ent = float((-(probs * log_probs).sum()).item())
        norm_ent = ent / float(np.log(n)) if n > 1 else 0.0
        loss = (float(-log_probs[true_label].item())
                if true_label is not None else float("nan"))

        result = LLMOutput(
            probs=probs.detach().cpu().numpy().astype(np.float32),
            loss=loss, pred=pred, confidence=conf, entropy=norm_ent,
            n_prompt_tokens=int(input_ids.shape[1]))

        if want_hidden:
            self._attach_hidden(result, out, attn)
        return result

    # ------------------------------------------------------------------ #
    def _attach_hidden(self, result: LLMOutput, out, attn) -> None:
        torch = self.torch
        layer = self.cfg.llm.hidden_layer
        hidden = out.hidden_states[layer][0]          # [T, D]
        mask = attn[0].bool()                         # [T]
        real = hidden[mask]                           # [T_real, D]

        result.h_last = real[-1].float().cpu().numpy().astype(np.float32)
        result.h_mean = real.mean(dim=0).float().cpu().numpy().astype(np.float32)

        n = self.cfg.llm.cache_hidden_tokens
        window = real[-n:]                            # last N real tokens
        d = window.shape[1]
        pad = n - window.shape[0]
        tok = torch.zeros(n, d, dtype=window.dtype, device=window.device)
        tmask = torch.zeros(n, dtype=torch.int64, device=window.device)
        if window.shape[0] > 0:                       # front-pad (right-align)
            tok[pad:] = window
            tmask[pad:] = 1
        result.h_tokens = tok.float().cpu().numpy().astype(np.float32)
        result.token_mask = tmask.cpu().numpy().astype(np.int64)
