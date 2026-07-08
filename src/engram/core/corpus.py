from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..storage import repository

_TOKEN = re.compile(r"[a-z0-9]+")
_MIN_CORPUS = 150
_STOP_FRAC = 0.5
_GENERIC_FRAC = 0.5
_BAND_SAMPLE = 400
_MIN_BAND_HITS = 50
_UPDATE_CLAMP = (0.82, 0.90)
_REINFORCE_CLAMP = (0.95, 0.99)


def _toks(text) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def _vec(blob) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
    except (TypeError, ValueError, BufferError):
        return None
    n = float(np.linalg.norm(arr))
    return arr / n if n else None


def _stop_generic(rows) -> Tuple[List[str], List[str]]:
    n = len(rows)
    df: Dict[str, int] = {}
    tdf: Dict[str, int] = {}
    for r in rows:
        for t in set(_toks(r["key"]) + _toks(r["value"])):
            df[t] = df.get(t, 0) + 1
        raw = r["tags"]
        try:
            tags = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (TypeError, ValueError):
            tags = []
        for t in set(tags):
            tdf[t] = tdf.get(t, 0) + 1
    stop = sorted(t for t, c in df.items() if c >= _STOP_FRAC * n)
    generic = sorted(t for t, c in tdf.items() if c >= _GENERIC_FRAC * n)
    return stop, generic


def _bands(conn, rows) -> Optional[Tuple[float, float]]:
    cos: List[float] = []
    for r in rows[:_BAND_SAMPLE]:
        pair = repository.vectors_for(conn, [r["id"]])
        if not pair:
            continue
        v = _vec(pair[0][1])
        if v is None:
            continue
        for fid, dist in repository.nearest(conn, v.tolist(), 6):
            if fid == r["id"]:
                continue
            other = repository.get(conn, fid)
            if other is None or other["superseded_by"] is not None:
                continue
            if other["validation_status"] != "fresh":
                continue
            if other["block_id"] != r["block_id"]:
                continue
            cos.append(1.0 - dist * dist / 2.0)
            break
    if len(cos) < _MIN_BAND_HITS:
        return None
    upd = float(np.clip(np.percentile(cos, 55), *_UPDATE_CLAMP))
    rei = float(np.clip(np.percentile(cos, 88), *_REINFORCE_CLAMP))
    if rei <= upd:
        rei = min(_REINFORCE_CLAMP[1], upd + 0.03)
    return round(rei, 4), round(upd, 4)


def _fallback() -> dict:
    return {"n": 0, "stoplist": sorted(config.STOPWORDS),
            "generic": sorted(config.GENERIC_TASK_TAGS),
            "reinforce": config.REINFORCE_SIM, "update": config.UPDATE_SIM}


def get(conn) -> dict:
    n = repository.count_current(conn)
    if n < _MIN_CORPUS:
        return _fallback()
    marker = repository.get_meta(conn, "corpus_n")
    cached = repository.get_meta(conn, "corpus")
    if marker is not None and cached is not None:
        prev = int(marker)
        if abs(n - prev) < max(64, int(0.25 * prev)):
            try:
                return json.loads(cached)
            except (TypeError, ValueError):
                pass
    rows = repository.all_current(conn)
    stop, generic = _stop_generic(rows)
    bands = _bands(conn, rows)
    stats = {"n": n, "stoplist": stop, "generic": generic,
             "reinforce": bands[0] if bands else config.REINFORCE_SIM,
             "update": bands[1] if bands else config.UPDATE_SIM}
    repository.set_meta(conn, "corpus", json.dumps(stats))
    repository.set_meta(conn, "corpus_n", str(n))
    return stats


def stoplist(conn) -> set:
    return set(get(conn)["stoplist"])


def generic_tags(conn) -> set:
    return set(get(conn)["generic"])


def bands(conn) -> Tuple[float, float]:
    s = get(conn)
    return s["reinforce"], s["update"]
