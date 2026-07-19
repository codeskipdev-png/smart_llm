"""LoRA adapters for UAAS: one frozen base model hosting several adapters of
different rank. Static baselines train a single adapter at r in {4,16,32}; the
adaptive scheme trains a bucket of ranks and, at inference, routes each input to
the adapter whose rank matches its uncertainty-derived r*(x).

Classification uses the same verbalizer target as CDKA (predict the option
letter), so accuracies are directly comparable to Phase-1A.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..llm.classifier import (option_token_ids, primary_letter_ids,
                              score_from_logits)
from ..llm.prompts import VerbalizerSpec, build_classification_messages
from ..utils.device import resolve_dtype
from ..utils.logging import get_logger

_log = get_logger("smart_llm.uaas")


class UAASLoRA:
    def __init__(self, cfg: Config, model_name: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.torch = torch
        name = model_name or cfg.llm.name
        dtype = resolve_dtype(cfg.llm.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, trust_remote_code=cfg.llm.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, device_map=cfg.llm.device_map,
            trust_remote_code=cfg.llm.trust_remote_code,
            attn_implementation=cfg.llm.attn_implementation)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.device = self.base.get_input_embeddings().weight.device
        self.peft = None
        self.adapters: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def _config(self, rank: int):
        from peft import LoraConfig
        return LoraConfig(
            r=rank, lora_alpha=self.cfg.uaas.lora_alpha,
            lora_dropout=self.cfg.uaas.lora_dropout,
            target_modules=self.cfg.uaas.target_modules,
            bias="none", task_type="CAUSAL_LM")

    def add_adapter(self, name: str, rank: int) -> None:
        from peft import get_peft_model
        cfg = self._config(rank)
        if self.peft is None:
            self.peft = get_peft_model(self.base, cfg, adapter_name=name)
        else:
            self.peft.add_adapter(name, cfg)
        self.adapters[name] = rank

    def _set_trainable(self, active: str) -> None:
        for n, p in self.peft.named_parameters():
            p.requires_grad_(("lora_" in n) and (f".{active}." in n))

    # ------------------------------------------------------------------ #
    def build_examples(self, texts: List[str], labels, verbalizer: VerbalizerSpec):
        torch = self.torch
        letter_ids = primary_letter_ids(self.tokenizer, verbalizer.letters)
        max_t = self.cfg.llm.max_input_tokens
        examples = []
        for text, y in zip(texts, labels):
            msg = build_classification_messages(text, verbalizer)
            pids = self.tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True)
            if len(pids) > max_t - 1:
                pids = pids[-(max_t - 1):]
            tgt = letter_ids[int(y)]
            input_ids = pids + [tgt]
            lbl = [-100] * len(pids) + [tgt]
            examples.append((torch.tensor(input_ids), torch.tensor(lbl)))
        return examples

    def _collate(self, batch):
        torch = self.torch
        pad = self.tokenizer.pad_token_id
        maxlen = max(len(x[0]) for x in batch)
        ids, lbls, attn = [], [], []
        for input_ids, lbl in batch:
            padn = maxlen - len(input_ids)
            ids.append(torch.cat([input_ids, torch.full((padn,), pad)]))
            lbls.append(torch.cat([lbl, torch.full((padn,), -100)]))
            attn.append(torch.cat([torch.ones(len(input_ids), dtype=torch.long),
                                   torch.zeros(padn, dtype=torch.long)]))
        return (torch.stack(ids).to(self.device),
                torch.stack(attn).to(self.device),
                torch.stack(lbls).to(self.device))

    def train_adapter(self, name: str, examples) -> dict:
        torch = self.torch
        self.peft.set_adapter(name)
        self._set_trainable(name)
        params = [p for p in self.peft.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.cfg.uaas.lr)
        bs = self.cfg.uaas.batch_size
        history = {"loss": []}
        self.peft.train()
        for epoch in range(self.cfg.uaas.epochs):
            order = np.random.default_rng(self.cfg.seed + epoch).permutation(len(examples))
            ep = 0.0
            for start in range(0, len(order), bs):
                batch = [examples[i] for i in order[start:start + bs]]
                ids, attn, lbls = self._collate(batch)
                out = self.peft(input_ids=ids, attention_mask=attn, labels=lbls)
                out.loss.backward(); opt.step(); opt.zero_grad()
                ep += float(out.loss.item()) * len(batch)
            history["loss"].append(ep / len(examples))
            _log.info("[adapter %s r=%d] epoch %d loss=%.4f",
                      name, self.adapters[name], epoch, history["loss"][-1])
        return history

    # ------------------------------------------------------------------ #
    @property
    def _groups(self):
        return getattr(self, "_cached_groups", None)

    def score(self, text: str, verbalizer: VerbalizerSpec, adapter: str,
              true_label: Optional[int] = None) -> dict:
        torch = self.torch
        self.peft.set_adapter(adapter)
        self.peft.eval()
        groups = option_token_ids(self.tokenizer, verbalizer.letters)
        msg = build_classification_messages(text, verbalizer)
        ids = self.tokenizer.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        max_t = self.cfg.llm.max_input_tokens
        if ids.shape[1] > max_t:
            ids = ids[:, -max_t:]
        ids = ids.to(self.device)
        with torch.no_grad():
            out = self.peft(input_ids=ids, attention_mask=torch.ones_like(ids))
        return score_from_logits(out.logits[0, -1, :], groups, true_label)

    def evaluate(self, texts, labels, verbalizer, adapter_of) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """adapter_of: str (fixed adapter) OR list[str] (per-sample)."""
        preds, losses, ranks = [], [], []
        for i, (text, y) in enumerate(zip(texts, labels)):
            ad = adapter_of if isinstance(adapter_of, str) else adapter_of[i]
            r = self.score(text, verbalizer, ad, int(y))
            preds.append(r["pred"]); losses.append(r["loss"])
            ranks.append(self.adapters[ad])
        return (np.asarray(preds, dtype=np.int64),
                np.asarray(losses, dtype=np.float32),
                np.asarray(ranks, dtype=np.int64))
