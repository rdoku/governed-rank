import math
from typing import List, Optional, Dict
import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def router_share_from_context(
    ctx: Dict,
    *,
    tau: float = 0.4162,  # Optuna-optimized (was 0.45)
    temp: float = 10.38,  # Optuna-optimized (was 12.0)
    floor: float = 0.15,
    ceil: float = 0.85,
    crush: float = 0.0,
    reorder_min: float = 0.15,
) -> float:
    npv = float(ctx.get("novelty_prior", [ctx.get("novelty_prior", 0.5)])[0] if isinstance(ctx.get("novelty_prior"), list) else ctx.get("novelty_prior", 0.5))
    reorder = float(ctx.get("reorder_exact", [ctx.get("reorder_exact", 0.0)])[0] if isinstance(ctx.get("reorder_exact"), list) else ctx.get("reorder_exact", 0.0))
    s_log = 1.0 / (1.0 + math.exp(-temp * (npv - tau)))
    s = (1.0 - crush * reorder) * s_log + reorder * reorder_min
    return float(max(floor, min(ceil, s)))


def mmr_select(item_ids: List[int], scores: np.ndarray, emb: np.ndarray, k: int, lamb: float = 0.25) -> List[int]:
    if not item_ids: return []
    pool = np.array(item_ids, dtype=np.int32)
    vecs = emb[pool]
    chosen = []
    rem = list(range(len(pool)))
    rel = scores[pool]
    while rem and len(chosen) < k:
        if not chosen:
            j = rem[int(np.argmax(rel[rem]))]
            chosen.append(j); rem.remove(j); continue
        sims = vecs[rem] @ vecs[chosen].T
        max_sim = sims.max(axis=1) if sims.ndim == 2 else sims
        mmr = lamb * rel[rem] - (1.0 - lamb) * max_sim
        j = rem[int(np.argmax(mmr))]
        chosen.append(j); rem.remove(j)
    return pool[chosen].tolist()


def topk_from_scores(scores: np.ndarray, k: int, exclude: Optional[set] = None) -> List[int]:
    if exclude:
        mask = np.ones_like(scores, dtype=bool)
        for e in exclude:
            if 0 <= e < scores.shape[0]: mask[e] = False
        idx = np.flatnonzero(mask)
        sub = scores[idx]
        kk = min(k, sub.size)
        sel = idx[np.argpartition(sub, -kk)[-kk:]]
        return sel[np.argsort(scores[sel])[::-1]].tolist()
    kk = min(k, scores.size)
    sel = np.argpartition(scores, -kk)[-kk:]
    return sel[np.argsort(scores[sel])[::-1]].tolist()


def history_recall_item2vec(item2vec: np.ndarray, history: List[int], num_items: int, *, L: int = 10, topk_per_seed: int = 20, centroid_topk: int = 20, persona_start: Optional[int] = None) -> List[int]:
    if not history: return []
    limit = persona_start if persona_start is not None else num_items
    seeds = [i for i in history[-L:] if 0 <= i < min(item2vec.shape[0], limit)]
    if not seeds: return []
    cand = set()
    for a in seeds:
        sims = item2vec @ item2vec[a]
        sims[a] = -1e9
        top = topk_from_scores(sims, topk_per_seed)
        cand.update(top)
    v = np.mean(item2vec[seeds], axis=0, keepdims=True).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    sims = (item2vec @ v.squeeze(0))
    for a in seeds: sims[a] = -1e9
    topc = topk_from_scores(sims, centroid_topk, exclude=set(seeds))
    cand.update(topc)
    return [i for i in cand if 0 <= i < num_items and (persona_start is None or i < persona_start)]


def score_candidates_item2vec(
    item2vec: np.ndarray, history: List[int], candidates: List[int]
) -> np.ndarray:
    if item2vec is None or not candidates:
        return np.zeros(len(candidates), dtype=np.float32)
    seeds = [i for i in history if 0 <= i < item2vec.shape[0]]
    if not seeds:
        return np.zeros(len(candidates), dtype=np.float32)
    v = np.mean(item2vec[seeds], axis=0, keepdims=True).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)

    scores = np.zeros(len(candidates), dtype=np.float32)
    for idx, cid in enumerate(candidates):
        if 0 <= cid < item2vec.shape[0]:
            scores[idx] = float(item2vec[cid] @ v.squeeze(0))
    return scores
