"""Reusable verbalizer scoring helpers (shared by UAAS eval and any model that
exposes next-token logits). Given the answer-position logits, marginalise over
each option letter's token variants to get a class distribution + loss."""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def option_token_ids(tokenizer, letters: List[str]) -> List[List[int]]:
    groups = []
    for letter in letters:
        ids = set()
        for variant in (letter, " " + letter):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        groups.append(sorted(ids))
    return groups


def primary_letter_ids(tokenizer, letters: List[str]) -> List[int]:
    """One target token id per letter (no-space variant), for LoRA training."""
    out = []
    for letter in letters:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        out.append(enc[0])
    return out


def score_from_logits(logits_last, groups: List[List[int]],
                      true_label: Optional[int] = None) -> dict:
    """logits_last: torch tensor [vocab]. Returns numpy class distribution + loss."""
    import torch
    logits = logits_last.float()
    scores = []
    for g in groups:
        if g:
            idx = torch.tensor(g, device=logits.device)
            scores.append(torch.logsumexp(logits[idx], dim=0))
        else:
            scores.append(torch.tensor(float("-inf"), device=logits.device))
    class_scores = torch.stack(scores)
    log_probs = torch.log_softmax(class_scores, dim=0)
    probs = log_probs.exp()
    n = len(groups)
    pred = int(torch.argmax(probs).item())
    conf = float(probs.max().item())
    ent = float((-(probs * log_probs).sum()).item())
    norm_ent = ent / float(np.log(n)) if n > 1 else 0.0
    loss = float(-log_probs[true_label].item()) if true_label is not None else float("nan")
    return {"probs": probs.detach().cpu().numpy().astype(np.float32),
            "pred": pred, "confidence": conf, "entropy": norm_ent, "loss": loss}
