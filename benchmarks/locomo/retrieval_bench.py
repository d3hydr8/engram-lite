from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, ".."))

import board
import engram.config as cfg

cfg.RECENCY_WEIGHT = 0.0

from engram.core import corpus, retrieval
from engram.core.memory import Memory
from engram.embeddings import get_embedder
from engram.storage import repository

LOCOMO = os.environ.get("LOCOMO_JSON", r"C:\Users\thesh\AppData\Local\Temp\locomo10.json")
K = 30
SLICES = [10, 30]


def date_of(dt):
    return dt.split(" on ")[-1].strip() if " on " in dt else dt


def new_fact_ids(res):
    out = []
    if not isinstance(res, dict):
        return out
    items = res.get("results") if res.get("decision") == "MULTI" else [res]
    for r in items or []:
        if r.get("decision") in ("ADD", "UPDATE", "DELETE") and r.get("fact_id"):
            out.append(r["fact_id"])
    return out


def ingest(conv, emb):
    db = os.path.join(tempfile.mkdtemp(prefix="locomo-"), "m.db")
    mem = Memory(db, embedder=emb, origin_tool="locomo")
    fid2dia = {}
    c = conv["conversation"]
    i = 1
    while f"session_{i}" in c:
        when = date_of(c.get(f"session_{i}_date_time", ""))
        for turn in c[f"session_{i}"]:
            res = mem.remember(turn["text"], speaker=turn["speaker"], when=when)
            for fid in new_fact_ids(res):
                fid2dia[fid] = turn["dia_id"]
        i += 1
    return mem, fid2dia


def hybrid_rows(mem, emb, q):
    return retrieval.search(mem.conn, emb, q, k=K, validate=False, touch=False)


def keyword_ids(mem, q):
    stop = corpus.stoplist(mem.conn)
    return repository.keyword_search(mem.conn, retrieval._fts_query(q, stop), K)


def dias_from_rows(rows, fid2dia):
    return [fid2dia[r["id"]] for r in rows if r["id"] in fid2dia]


def dias_from_ids(ids, fid2dia):
    return [fid2dia[i] for i in ids if i in fid2dia]


def run(data, emb):
    hyb = {s: 0 for s in SLICES}
    kw = {s: 0 for s in SLICES}
    total = 0
    facts = 0
    for conv in data:
        mem, fid2dia = ingest(conv, emb)
        facts += len(mem.all_current())
        for qa in conv["qa"]:
            if qa.get("category") == 5:
                continue
            ev = set(qa.get("evidence") or [])
            if not ev:
                continue
            total += 1
            rows = hybrid_rows(mem, emb, qa["question"])
            kids = keyword_ids(mem, qa["question"])
            for s in SLICES:
                if ev & set(dias_from_rows(rows[:s], fid2dia)):
                    hyb[s] += 1
                if ev & set(dias_from_ids(kids[:s], fid2dia)):
                    kw[s] += 1
        mem.close()
    return hyb, kw, total, facts


def digest(conv, emb):
    mem, fid2dia = ingest(conv, emb)
    parts = []
    for qa in conv["qa"]:
        if qa.get("category") == 5:
            continue
        rows = hybrid_rows(mem, emb, qa["question"])
        parts.append("|".join(sorted(dias_from_rows(rows, fid2dia))))
    mem.close()
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def render(hyb, kw, total, facts, engine, det):
    lines = [board.banner("LOCOMO RETRIEVAL  —  MEASURED HERE (no LLM)"), ""]
    lines.append(board.section(1, f"evidence recall — cats 1-4, {total} questions, "
                                  f"{facts} facts stored ({engine})"))
    lines.append("")
    header = ["method", "recall@10", "recall@30"]
    rows = [
        ["keyword-only (grep-like)"] + [f"{100.0*kw[s]/total:.1f}%" for s in SLICES],
        ["engram hybrid ◄"] + [f"{100.0*hyb[s]/total:.1f}%" for s in SLICES],
    ]
    lines += board.grid(header, rows)
    lines.append("")
    lines.append(board.section(2, "determinism"))
    lines.append("")
    lines += board.footnotes([
        f"two from-scratch runs of conversation 1 → digests {det[0]} / {det[1]}   "
        f"{'IDENTICAL' if det[0] == det[1] else 'DIFFER'}",
    ])
    lines.append("")
    lines.append(board.rule("-"))
    lines += board.footnotes([
        "a question is a hit when any of its evidence turns is served in the top-K.",
        "keyword-only = FTS5 over the same store (the grep-like baseline).",
        "engram hybrid = keyword + vector + entity, RRF-fused, corpus-derived stoplist.",
        "recency weighting off for replay (the only nondeterminism source).",
        "LoCoMo: Maharana et al. (arXiv:2402.17753); no LLM in this measurement.",
    ])
    return "\n".join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not os.path.exists(LOCOMO):
        print(f"LoCoMo dataset not found at {LOCOMO}. Set LOCOMO_JSON to the path of "
              "locomo10.json (github.com/snap-research/locomo, data/locomo10.json).")
        return
    data = json.load(open(LOCOMO, encoding="utf-8"))
    emb = get_embedder()
    engine = getattr(emb, "model_name", type(emb).__name__)
    hyb, kw, total, facts = run(data, emb)
    det = (digest(data[0], emb), digest(data[0], emb))
    print(render(hyb, kw, total, facts, engine, det))


if __name__ == "__main__":
    main()
