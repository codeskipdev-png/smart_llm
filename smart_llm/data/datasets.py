"""Dataset loaders with a single, uniform interface.

Every loader returns a :class:`Corpus`:

* ``pool_*``  — the retrieval pool + in-context demonstration source (train).
* ``eval_*``  — the samples we classify / route / log (test).
* ``label_names`` — human-readable class names (index-aligned to integer labels).

20 Newsgroups is loaded via scikit-learn (no download token needed). The other
datasets use HuggingFace ``datasets`` through a small preset table so an unknown
HF id fails loudly instead of silently mislabeling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..config import Config
from ..utils.logging import get_logger

_log = get_logger("smart_llm.data")


# --------------------------------------------------------------------------- #
@dataclass
class Corpus:
    name: str
    label_names: List[str]
    # retrieval pool / demo source
    pool_ids: List[str]
    pool_texts: List[str]
    pool_labels: np.ndarray          # int64 [n_pool]
    # evaluation set
    eval_ids: List[str]
    eval_texts: List[str]
    eval_labels: np.ndarray          # int64 [n_eval]

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def summary(self) -> str:
        return (f"[{self.name}] classes={self.n_classes} "
                f"pool={len(self.pool_texts)} eval={len(self.eval_texts)}")


# --------------------------------------------------------------------------- #
# HuggingFace presets: key -> (path, name, text_field, label_field, splits)
# --------------------------------------------------------------------------- #
_HF_PRESETS = {
    "agnews": dict(
        path="ag_news", name=None, text="text", label="label",
        train_split="train", eval_split="test",
        label_names=["World", "Sports", "Business", "Sci/Tech"]),
    "tweeteval": dict(
        path="tweet_eval", name="sentiment", text="text", label="label",
        train_split="train", eval_split="test",
        label_names=["negative", "neutral", "positive"]),
    "tweeteval_emotion": dict(
        path="tweet_eval", name="emotion", text="text", label="label",
        train_split="train", eval_split="test",
        label_names=["anger", "joy", "optimism", "sadness"]),
    "financial_phrasebank": dict(
        path="financial_phrasebank", name="sentences_50agree",
        text="sentence", label="label",
        train_split="train", eval_split=None,   # single split -> we carve one
        label_names=["negative", "neutral", "positive"]),
    # parquet-native financial sentiment (loads on datasets>=3.0; no script)
    "twitter_financial": dict(
        path="zeroshot/twitter-financial-news-sentiment", name=None,
        text="text", label="label", train_split="train", eval_split="validation",
        label_names=["Bearish", "Bullish", "Neutral"]),
    # canonical parquet-native sentiment fallback (guaranteed to load)
    "rotten_tomatoes": dict(
        path="rotten_tomatoes", name=None, text="text", label="label",
        train_split="train", eval_split="test",
        label_names=["negative", "positive"]),
    "pubmed": dict(  # PubMed 20k RCT sentence-role classification
        path="armanc/pubmed-rct20k", name=None, text="text", label="label",
        train_split="train", eval_split="test",
        label_names=None),  # derived from the ClassLabel feature
}


def list_datasets() -> List[str]:
    return ["20newsgroups", *sorted(_HF_PRESETS)]


# --------------------------------------------------------------------------- #
def _clean_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:max_chars]


def _subsample(ids, texts, labels, cap: int, seed: int, stratify: bool = True
               ) -> Tuple[List[str], List[str], np.ndarray]:
    """Deterministically cap a split, stratified by label when possible."""
    n = len(texts)
    if cap is None or cap >= n:
        return list(ids), list(texts), np.asarray(labels)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    if stratify:
        keep = []
        classes = np.unique(labels)
        per = max(1, cap // len(classes))
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            keep.extend(idx[:per].tolist())
        keep = np.array(keep)
        if len(keep) > cap:
            rng.shuffle(keep)
            keep = keep[:cap]
    else:
        keep = rng.permutation(n)[:cap]
    keep = np.sort(keep)
    return ([ids[i] for i in keep], [texts[i] for i in keep], labels[keep])


# --------------------------------------------------------------------------- #
def _load_20newsgroups(cfg: Config) -> Corpus:
    from sklearn.datasets import fetch_20newsgroups
    remove = ("headers", "footers", "quotes") if cfg.data.remove_headers else ()
    train = fetch_20newsgroups(subset="train", remove=remove,
                               shuffle=True, random_state=cfg.data.seed)
    test = fetch_20newsgroups(subset="test", remove=remove,
                              shuffle=True, random_state=cfg.data.seed)
    label_names = list(train.target_names)
    mc = cfg.data.text_max_chars

    pool_texts = [_clean_text(t, mc) for t in train.data]
    eval_texts = [_clean_text(t, mc) for t in test.data]
    pool_ids = [f"train-{i}" for i in range(len(pool_texts))]
    eval_ids = [f"test-{i}" for i in range(len(eval_texts))]

    # drop empties created by header/footer stripping
    pool = [(i, t, y) for i, t, y in zip(pool_ids, pool_texts, train.target) if t]
    ev = [(i, t, y) for i, t, y in zip(eval_ids, eval_texts, test.target) if t]
    pool_ids, pool_texts, pool_labels = zip(*pool)
    eval_ids, eval_texts, eval_labels = zip(*ev)

    pool_ids, pool_texts, pool_labels = _subsample(
        list(pool_ids), list(pool_texts), list(pool_labels),
        cfg.data.max_train, cfg.data.seed)
    eval_ids, eval_texts, eval_labels = _subsample(
        list(eval_ids), list(eval_texts), list(eval_labels),
        cfg.data.max_eval, cfg.data.seed + 1)
    return Corpus("20newsgroups", label_names,
                  pool_ids, pool_texts, np.asarray(pool_labels, dtype=np.int64),
                  eval_ids, eval_texts, np.asarray(eval_labels, dtype=np.int64))


def _load_parquet_fallback(path, name, split):
    """Load a dataset from the Hub's auto-converted Parquet branch.

    Modern ``datasets`` (>=3.0) refuses script-based datasets (e.g.
    financial_phrasebank). Almost every public dataset also has an auto-generated
    Parquet copy on the ``refs/convert/parquet`` git ref; we read that directly
    with huggingface_hub + pandas, which is independent of the datasets version.
    """
    import datasets as hfds
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(path, repo_type="dataset",
                                revision="refs/convert/parquet")

    def _match(f):
        if not f.endswith(".parquet"):
            return False
        if name and name not in f:            # restrict to the requested config
            return False
        return split in f                     # and the requested split

    cands = [f for f in files if _match(f)]
    if not cands:                             # relax: split token only
        cands = [f for f in files if f.endswith(".parquet") and split in f]
    if not cands:
        raise RuntimeError(
            f"No Parquet files for {path}/{name}/{split} on refs/convert/parquet; "
            f"available (first 20): {files[:20]}")
    local = [hf_hub_download(path, f, repo_type="dataset",
                             revision="refs/convert/parquet") for f in sorted(cands)]
    df = pd.concat([pd.read_parquet(p) for p in local], ignore_index=True)
    _log.info("Loaded %s/%s:%s via Parquet fallback (%d rows)", path, name, split, len(df))
    return hfds.Dataset.from_pandas(df, preserve_index=False)


def _load_via_datasets_server(path, name, split):
    """Fetch Parquet via the HF datasets-server API.

    This is more general than the repo's git ref: the server can hold a Parquet
    copy even when the dataset repo has no ``refs/convert/parquet`` branch (as with
    the old script-based financial_phrasebank).
    """
    import json
    import os
    import tempfile
    import urllib.request

    import datasets as hfds
    import pandas as pd

    url = f"https://datasets-server.huggingface.co/parquet?dataset={path}"
    with urllib.request.urlopen(url, timeout=60) as r:  # nosec - public API
        meta = json.load(r)
    pfiles = meta.get("parquet_files", [])
    sel = [f for f in pfiles if f.get("split") == split
           and (name is None or f.get("config") == name)]
    if not sel:
        avail = sorted({(f.get("config"), f.get("split")) for f in pfiles})
        raise RuntimeError(f"datasets-server has no parquet for {path}/{name}/{split}; "
                           f"available configs/splits: {avail[:20]}")
    frames = []
    for f in sel:
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        tmp.close()
        urllib.request.urlretrieve(f["url"], tmp.name)  # nosec - public file
        frames.append(pd.read_parquet(tmp.name))
        os.unlink(tmp.name)
    df = pd.concat(frames, ignore_index=True)
    _log.info("Loaded %s/%s:%s via datasets-server (%d rows)", path, name, split, len(df))
    return hfds.Dataset.from_pandas(df, preserve_index=False)


def _hf_load_split(path, name, split):
    """Load an HF split, tolerant of datasets-library version differences.

    Order: (1) plain load_dataset, (2) legacy trust_remote_code, (3) the repo's
    auto-converted Parquet branch, (4) the datasets-server Parquet API. Scripts
    are unsupported on datasets>=3.0, so (3)/(4) are the fallbacks that matter.
    """
    from datasets import load_dataset
    errors = []
    for attempt in ("plain", "trc", "convert", "server"):
        try:
            if attempt == "plain":
                return load_dataset(path, name, split=split)
            if attempt == "trc":
                return load_dataset(path, name, split=split, trust_remote_code=True)
            if attempt == "convert":
                return _load_parquet_fallback(path, name, split)
            return _load_via_datasets_server(path, name, split)
        except Exception as exc:  # try the next strategy
            errors.append(f"{attempt}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All load strategies failed for "
                       f"{path}/{name}:{split}\n  " + "\n  ".join(errors))


def _load_hf(cfg: Config, key: str) -> Corpus:
    spec = _HF_PRESETS[key]
    mc = cfg.data.text_max_chars

    def _split_to_arrays(ds, prefix):
        texts = [_clean_text(x, mc) for x in ds[spec["text"]]]
        labels = list(ds[spec["label"]])
        ids = [f"{prefix}-{i}" for i in range(len(texts))]
        keep = [(i, t, y) for i, t, y in zip(ids, texts, labels) if t]
        i2, t2, y2 = zip(*keep)
        return list(i2), list(t2), list(y2)

    train_ds = _hf_load_split(spec["path"], spec["name"], spec["train_split"])
    # label names: preset override, else the ClassLabel feature names.
    label_names = spec["label_names"]
    if label_names is None:
        feat = train_ds.features[spec["label"]]
        label_names = list(getattr(feat, "names", None) or
                           sorted({str(x) for x in train_ds[spec["label"]]}))

    if spec["eval_split"] is not None:
        eval_ds = _hf_load_split(spec["path"], spec["name"], spec["eval_split"])
        p_ids, p_txt, p_lab = _split_to_arrays(train_ds, "train")
        e_ids, e_txt, e_lab = _split_to_arrays(eval_ds, "test")
    else:  # single split -> deterministic carve
        i, t, y = _split_to_arrays(train_ds, "all")
        rng = np.random.default_rng(cfg.data.seed)
        perm = rng.permutation(len(t))
        cut = int(0.8 * len(t))
        tr, ev = perm[:cut], perm[cut:]
        p_ids = [i[k] for k in tr]; p_txt = [t[k] for k in tr]; p_lab = [y[k] for k in tr]
        e_ids = [i[k] for k in ev]; e_txt = [t[k] for k in ev]; e_lab = [y[k] for k in ev]

    p_ids, p_txt, p_lab = _subsample(p_ids, p_txt, p_lab, cfg.data.max_train, cfg.data.seed)
    e_ids, e_txt, e_lab = _subsample(e_ids, e_txt, e_lab, cfg.data.max_eval, cfg.data.seed + 1)
    return Corpus(key, list(label_names),
                  p_ids, p_txt, np.asarray(p_lab, dtype=np.int64),
                  e_ids, e_txt, np.asarray(e_lab, dtype=np.int64))


# --------------------------------------------------------------------------- #
def load_corpus(cfg: Config) -> Corpus:
    name = cfg.data.dataset.lower()
    if name in ("20newsgroups", "20ng", "newsgroups"):
        corpus = _load_20newsgroups(cfg)
    elif name in _HF_PRESETS:
        corpus = _load_hf(cfg, name)
    else:
        raise ValueError(
            f"Unknown dataset '{cfg.data.dataset}'. Known: {list_datasets()}")

    # optional class-subset restriction (re-indexes labels contiguously)
    if cfg.data.label_subset:
        corpus = _restrict_labels(corpus, cfg.data.label_subset)
    _log.info(corpus.summary())
    return corpus


def _restrict_labels(corpus: Corpus, subset: List[str]) -> Corpus:
    keep_idx = [corpus.label_names.index(s) for s in subset]
    remap = {old: new for new, old in enumerate(keep_idx)}

    def _filt(ids, texts, labels):
        out = [(i, t, remap[int(y)]) for i, t, y in zip(ids, texts, labels)
               if int(y) in remap]
        if not out:
            return [], [], np.array([], dtype=np.int64)
        i2, t2, y2 = zip(*out)
        return list(i2), list(t2), np.asarray(y2, dtype=np.int64)

    p = _filt(corpus.pool_ids, corpus.pool_texts, corpus.pool_labels)
    e = _filt(corpus.eval_ids, corpus.eval_texts, corpus.eval_labels)
    return Corpus(corpus.name, list(subset), *p, *e)
