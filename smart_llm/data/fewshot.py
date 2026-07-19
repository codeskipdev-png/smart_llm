"""Formatting of retrieved in-context demonstrations.

Kept separate from :mod:`smart_llm.llm.prompts` because the *demo block* is data,
whereas the surrounding instruction/verbalizer is model-facing prompt logic.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def format_example(text: str, label_name: str, max_chars: int = 600) -> str:
    """One labelled demonstration."""
    text = text.strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return f"Text: {text}\nCategory: {label_name}"


def build_demo_block(demos: Sequence[Tuple[str, str]], max_chars: int = 600) -> str:
    """Render a list of ``(text, label_name)`` demonstrations into one block."""
    if not demos:
        return ""
    lines = ["Here are related labelled examples:"]
    for i, (text, label_name) in enumerate(demos, 1):
        lines.append(f"[Example {i}]\n{format_example(text, label_name, max_chars)}")
    return "\n\n".join(lines)


def demos_from_indices(indices: Sequence[int],
                       pool_texts: List[str],
                       pool_labels,
                       label_names: List[str],
                       max_chars: int = 600) -> List[Tuple[str, str]]:
    return [(pool_texts[i], label_names[int(pool_labels[i])]) for i in indices]
