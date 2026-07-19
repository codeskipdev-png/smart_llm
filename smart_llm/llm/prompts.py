"""Prompt construction for MMLU-style verbalizer classification.

The LLM is turned into a classifier by asking it to answer with a single option
letter (A, B, C, ...). This lets us read a proper probability distribution over
classes from a *single* forward pass (gather the answer-position logits over the
option-letter tokens) and define a well-posed cross-entropy loss — which is what
the ground-truth retrieval-benefit signal needs.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import List, Optional

SYSTEM_PROMPT = (
    "You are a careful text classification assistant. Read the text and choose "
    "the single most appropriate category. Respond with only the letter of the "
    "correct option."
)


def option_letters(n: int) -> List[str]:
    """A, B, ..., for up to 26 classes."""
    if n > len(string.ascii_uppercase):
        raise ValueError(f"verbalizer supports <=26 classes, got {n}")
    return list(string.ascii_uppercase[:n])


@dataclass
class VerbalizerSpec:
    """Maps class indices <-> option letters for one dataset."""
    label_names: List[str]

    @property
    def letters(self) -> List[str]:
        return option_letters(len(self.label_names))

    def options_block(self) -> str:
        return "\n".join(f"{L}) {name}"
                         for L, name in zip(self.letters, self.label_names))


def build_classification_messages(text: str,
                                  verbalizer: VerbalizerSpec,
                                  demo_block: Optional[str] = None,
                                  task_hint: Optional[str] = None) -> List[dict]:
    """Return chat ``messages`` for tokenizer.apply_chat_template.

    ``demo_block`` (retrieved in-context examples) is placed *before* the query so
    that the answer scaffold (options + letter request) stays at the very end,
    where left-truncation cannot remove it.
    """
    parts: List[str] = []
    if task_hint:
        parts.append(task_hint)
    if demo_block:
        parts.append(demo_block)
    text = text.strip()
    parts.append(f"Classify the following text.\n\nText: {text}")
    parts.append("Categories:\n" + verbalizer.options_block())
    parts.append("Answer with the single letter of the correct category.")
    user = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
