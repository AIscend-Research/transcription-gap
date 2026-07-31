"""Cross-run analysis.

A single run tells you that a text drifted. The claim about an *attractor*
needs more than one run: if several different scores, put through the same
transcriber, end up closer to each other than they started, that shared
destination is the machine's house style — the speech-domain analogue of the
published image-generation convergence result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import metrics as M


def load_run(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary["_dir"] = str(out)
    summary["_name"] = out.name
    summary["_texts"] = [
        p.read_text(encoding="utf-8").strip()
        for p in sorted((out / "iterations").glob("*.txt"))
    ]
    return summary


def find_runs(root: str | Path) -> list[dict]:
    root = Path(root)
    dirs = sorted(p.parent for p in root.glob("**/summary.json"))
    return [load_run(d) for d in dirs]


def convergence_table(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        conv = r.get("convergence", {})
        auth = r.get("authorship", {})
        rows.append({
            "run": r["_name"],
            "iterations": r.get("iterations_run"),
            "status": conv.get("status"),
            "fixed_point_at": conv.get("fixed_point_at"),
            "early_delta": conv.get("early_mean_delta"),
            "late_delta": conv.get("late_mean_delta"),
            "contracting": conv.get("contracting"),
            "drift_from_seed": r.get("total_drift_from_seed"),
            "machine_share": auth.get("machine_share"),
            "voice": r.get("voice"),
            "listen_model": r.get("listen_model"),
            "smart_format": r.get("smart_format"),
        })
    return rows


def mutual_convergence(runs: list[dict]) -> dict:
    """Do distinct scores end up more alike than they began?

    ``seed_pairwise`` is the mean similarity between the runs' starting texts,
    ``final_pairwise`` between their ending texts. If the finals are closer,
    the loop is pulling everything towards a common attractor rather than just
    degrading each text independently.
    """
    if len(runs) < 2:
        return {"note": "need at least two runs to measure mutual convergence"}

    seeds = [r["seed_text"] for r in runs]
    finals = [r["final_text"] for r in runs]

    def mean_pairwise(texts: list[str], fn) -> float:
        vals = [
            fn(texts[i], texts[j])
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    seed_j = mean_pairwise(seeds, M.jaccard)
    final_j = mean_pairwise(finals, M.jaccard)
    seed_c = mean_pairwise(seeds, M.cosine)
    final_c = mean_pairwise(finals, M.cosine)

    # Vocabulary every run's final text shares but no seed contained: the
    # attractor's own words.
    seed_vocab = set().union(*[set(M.words(t)) for t in seeds])
    final_vocabs = [set(M.words(t)) for t in finals]
    shared_final = set.intersection(*final_vocabs) if final_vocabs else set()
    attractor_vocab = sorted(shared_final - seed_vocab)

    return {
        "runs": [r["_name"] for r in runs],
        "seed_pairwise_jaccard": seed_j,
        "final_pairwise_jaccard": final_j,
        "jaccard_gain": final_j - seed_j,
        "seed_pairwise_cosine": seed_c,
        "final_pairwise_cosine": final_c,
        "cosine_gain": final_c - seed_c,
        "converging_toward_each_other": bool(final_j > seed_j),
        "attractor_vocab": attractor_vocab[:100],
        "attractor_vocab_size": len(attractor_vocab),
    }


def _fmt(v, d=3) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "–" if v != v else f"{v:.{d}f}"
    return str(v)


def markdown_report(runs: list[dict]) -> str:
    lines: list[str] = ["# Transcription gap — cross-run analysis", ""]
    lines.append(f"{len(runs)} run(s).")
    lines.append("")

    lines.append("## Per-run convergence")
    lines.append("")
    cols = ["run", "iters", "status", "fixed at", "early Δ", "late Δ",
            "contracting", "drift", "machine share", "smart_format"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for row in convergence_table(runs):
        lines.append("| " + " | ".join([
            row["run"], _fmt(row["iterations"]), _fmt(row["status"]),
            _fmt(row["fixed_point_at"]), _fmt(row["early_delta"]),
            _fmt(row["late_delta"]), _fmt(row["contracting"]),
            _fmt(row["drift_from_seed"]), _fmt(row["machine_share"]),
            _fmt(row["smart_format"]),
        ]) + " |")
    lines.append("")

    mc = mutual_convergence(runs)
    lines.append("## Mutual convergence (is there a shared attractor?)")
    lines.append("")
    if "note" in mc:
        lines.append(f"_{mc['note']}_")
    else:
        lines.append(f"- mean pairwise vocabulary overlap, seeds: **{_fmt(mc['seed_pairwise_jaccard'])}**")
        lines.append(f"- mean pairwise vocabulary overlap, finals: **{_fmt(mc['final_pairwise_jaccard'])}**")
        lines.append(f"- change: **{_fmt(mc['jaccard_gain'])}** "
                     f"({'converging' if mc['converging_toward_each_other'] else 'diverging'})")
        lines.append(f"- mean pairwise cosine, seeds → finals: "
                     f"{_fmt(mc['seed_pairwise_cosine'])} → {_fmt(mc['final_pairwise_cosine'])}")
        lines.append(f"- words shared by every final text but present in no seed "
                     f"({mc['attractor_vocab_size']}): "
                     f"{', '.join(mc['attractor_vocab'][:40]) or '—'}")
    lines.append("")

    lines.append("## Most persistent rewrites")
    lines.append("")
    from collections import Counter
    pooled: Counter = Counter()
    for r in runs:
        for e in r.get("substitution_ledger", []):
            pooled[(e["instead_of"], e["heard_as"])] += e["count"]
    if pooled:
        lines.append("| heard as | instead of | count |")
        lines.append("|---|---|---|")
        for (ref, hyp), c in pooled.most_common(30):
            lines.append(f"| {hyp} | {ref} | {c} |")
    else:
        lines.append("_none recorded_")
    lines.append("")

    lines.append("## Final texts")
    lines.append("")
    for r in runs:
        lines.append(f"### {r['_name']}")
        lines.append("")
        lines.append("> " + r["final_text"].replace("\n", "\n> "))
        lines.append("")

    return "\n".join(lines)


def analyze(root: str | Path, out_path: str | Path | None = None) -> str:
    runs = find_runs(root)
    if not runs:
        raise SystemExit(f"no finished runs found under {root}")
    md = markdown_report(runs)
    if out_path:
        Path(out_path).write_text(md, encoding="utf-8")
    return md
