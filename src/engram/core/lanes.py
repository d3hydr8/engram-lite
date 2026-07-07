from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..storage import repository


def active(embedder) -> bool:
    return bool(getattr(embedder, "calibrated", True))


def _unit(vec) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return None
    return arr / norm


def _from_blob(blob) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(blob, dtype=np.float32)
    except (TypeError, ValueError, BufferError):
        return None
    return _unit(arr)


def build(conn, embedder, scope_tags, origin) -> Optional[Tuple[List[float], float]]:
    refs: List[np.ndarray] = []
    for tag in scope_tags or []:
        text = str(tag).replace("-", " ").strip()
        if not text:
            continue
        vec = _unit(embedder.embed(text))
        if vec is not None:
            refs.append(vec)
    if origin:
        rows = repository.facts_by_origin(conn, origin, config.LANE_FETCH_LIMIT)
        for _, blob in repository.vectors_for(conn, [r["id"] for r in rows]):
            vec = _from_blob(blob)
            if vec is not None:
                refs.append(vec)
    if not refs:
        return None
    centre = _unit(np.mean(np.vstack(refs), axis=0))
    if centre is None:
        return None
    floor = min(float(np.dot(centre, r)) for r in refs)
    return centre.tolist(), floor


def scores(conn, centre: List[float], fids: List[str]) -> Dict[str, float]:
    if centre is None or not fids:
        return {}
    c = np.asarray(centre, dtype=np.float64).reshape(-1)
    out: Dict[str, float] = {}
    for fid, blob in repository.vectors_for(conn, fids):
        vec = _from_blob(blob)
        if vec is not None and vec.size == c.size:
            out[fid] = float(np.dot(c, vec))
    return out
