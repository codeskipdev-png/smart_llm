"""Contribution 3 — attribution-guided explanation verification.

We do NOT assume a generated explanation is faithful. Instead:

  1. Attribute the model's classification decision to input tokens with Integrated
     Gradients (Sundararajan et al., 2017) over the embedding layer (falls back to
     input x gradient if captum is unavailable).
  2. Ask the model to explain its decision in natural language.
  3. Measure whether the tokens that actually drove the prediction (top-k by
     attribution) are reflected in the explanation -> a *faithfulness* score.

A high score means the explanation talks about the evidence the model relied on;
a low score flags an explanation that is decoupled from the decision.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import Config
from ..llm.classifier import (option_token_ids, primary_letter_ids,
                              score_from_logits)
from ..llm.prompts import (SYSTEM_PROMPT, VerbalizerSpec,
                           build_classification_messages, encode_chat)
from ..utils.logging import get_logger

_log = get_logger("smart_llm.explain")

_STOP = set("""a an the and or but of to in on for with is are was were be been being
this that these those it its as at by from into over under then than so such not no
i you he she they we them his her their our your my me""".split())


def _clean_token(tok: str) -> str:
    # strip common BPE markers (Ġ, ▁) and surrounding punctuation
    tok = tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", " ")
    return tok.strip().strip(string.punctuation).lower()


@dataclass
class Attribution:
    text: str
    pred_class: int
    tokens: List[str]
    scores: np.ndarray
    top_tokens: List[str]


class AttributionExplainer:
    def __init__(self, cfg: Config, model_name: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.torch = torch
        name = model_name or cfg.llm.name
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, trust_remote_code=cfg.llm.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # bf16 keeps a 7B within a 24 GB card (RTX 4090); IG interpolation is
        # batched (ig_internal_batch) to bound activation memory. Gradients flow
        # to embedding activations, not to the frozen weights.
        from ..utils.device import resolve_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=resolve_dtype(cfg.explain.dtype),
            device_map=cfg.llm.device_map,
            trust_remote_code=cfg.llm.trust_remote_code)
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device

    # ------------------------------------------------------------------ #
    def _prompt_ids(self, text, verbalizer):
        torch = self.torch
        msg = build_classification_messages(text, verbalizer)
        id_list = encode_chat(self.tokenizer, msg, add_generation_prompt=True)
        max_t = self.cfg.llm.max_input_tokens
        if len(id_list) > max_t:
            id_list = id_list[-max_t:]
        return torch.tensor([id_list], dtype=torch.long, device=self.device)

    def predict(self, text, verbalizer, true_label=None) -> dict:
        torch = self.torch
        ids = self._prompt_ids(text, verbalizer)
        groups = option_token_ids(self.tokenizer, verbalizer.letters)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
        return score_from_logits(out.logits[0, -1, :], groups, true_label)

    # ------------------------------------------------------------------ #
    def attribute(self, text, verbalizer) -> Attribution:
        torch = self.torch
        ids = self._prompt_ids(text, verbalizer)
        pred = self.predict(text, verbalizer)["pred"]
        letter_id = primary_letter_ids(self.tokenizer, verbalizer.letters)[pred]
        emb_layer = self.model.get_input_embeddings()

        def forward_letter(input_ids):
            out = self.model(input_ids=input_ids,
                             attention_mask=torch.ones_like(input_ids))
            return out.logits[:, -1, letter_id]

        try:
            from captum.attr import LayerIntegratedGradients
            lig = LayerIntegratedGradients(forward_letter, emb_layer)
            baseline = torch.full_like(ids, self.tokenizer.pad_token_id)
            attr = lig.attribute(inputs=ids, baselines=baseline,
                                 n_steps=self.cfg.explain.ig_steps,
                                 internal_batch_size=self.cfg.explain.ig_internal_batch)
            scores = attr.sum(dim=-1).squeeze(0).abs().float().detach().cpu().numpy()
        except ImportError:
            scores = self._input_x_grad(ids, forward_letter, emb_layer)

        tokens = self.tokenizer.convert_ids_to_tokens(ids[0].tolist())
        top = self._top_content_tokens(tokens, scores, self.cfg.explain.top_k_tokens)
        return Attribution(text, pred, tokens, scores.astype(np.float32), top)

    def _input_x_grad(self, ids, forward_letter, emb_layer):
        """Fallback attribution: |embedding · d(logit)/d(embedding)|."""
        torch = self.torch
        emb = emb_layer(ids).detach().requires_grad_(True)

        def fwd(inputs_embeds):
            out = self.model(inputs_embeds=inputs_embeds,
                             attention_mask=torch.ones(inputs_embeds.shape[:2],
                                                       device=inputs_embeds.device))
            return out.logits[:, -1, :]

        logit = fwd(emb)
        # gradient of the max class logit at the answer position
        target = logit[0].argmax()
        grad = torch.autograd.grad(logit[0, target], emb)[0]
        return (grad * emb).sum(dim=-1).squeeze(0).abs().detach().cpu().numpy()

    def _top_content_tokens(self, tokens, scores, k) -> List[str]:
        order = np.argsort(-scores)
        out, seen = [], set()
        for j in order:
            w = _clean_token(tokens[j])
            if len(w) < 3 or w in _STOP or w in seen:
                continue
            seen.add(w)
            out.append(w)
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ #
    def generate_explanation(self, text, verbalizer, pred_class, max_new=80) -> str:
        torch = self.torch
        label = verbalizer.label_names[pred_class]
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Text: {text}\n\nYou classified this as '{label}'. In one or two "
                f"sentences, explain which words or phrases in the text justify that "
                f"category."},
        ]
        id_list = encode_chat(self.tokenizer, msg, add_generation_prompt=True)
        if len(id_list) > self.cfg.llm.max_input_tokens:
            id_list = id_list[-self.cfg.llm.max_input_tokens:]
        ids = torch.tensor([id_list], dtype=torch.long, device=self.device)
        with torch.no_grad():
            gen = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)


def faithfulness_score(top_tokens: List[str], explanation: str) -> float:
    """Fraction of the top-attribution content tokens that appear in the
    explanation (word-boundary match). In [0, 1]; higher = more faithful."""
    if not top_tokens:
        return float("nan")
    expl = explanation.lower()
    hits = sum(1 for w in top_tokens
               if re.search(r"\b" + re.escape(w) + r"\b", expl))
    return hits / len(top_tokens)
