"""Render the paper's figures from a completed run or sweep.

Reads the same summary.json / metrics.jsonl the loop already writes and emits
vector PDFs, one per figure, ready for \\includegraphics. Nothing here recomputes
a metric: if a number appears in a figure it was measured by the loop.

    python figures.py                      # outputs/sweep -> figures/
    python figures.py --run outputs/run    # a single run
    python figures.py --out paper/figs

Runs that were produced offline are refused by default -- the simulated
transcriber is a caricature and its numbers must not reach a figure. Pass
--allow-offline to override for pipeline testing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One restrained house style, applied once. Thin strokes, no chartjunk, type
# sized to survive a two-column reduction.
plt.rcParams.update({
    "figure.figsize": (5.0, 3.1),
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

INK = "#1a1a1a"
MACHINE = "#c2410c"
HUMAN = "#94a3b8"
ACCENT = "#0f766e"

# Real API iterations take roughly half their audio duration; an offline run
# returns near-instantly. The gap is three orders of magnitude, so a loose
# threshold separates them safely.
OFFLINE_RATIO = 0.05


def load_summary(d: pathlib.Path) -> dict:
    return json.loads((d / "summary.json").read_text())


def is_offline(s: dict) -> bool:
    audio = s.get("cost", {}).get("stt_seconds", 0.0)
    if not audio:
        return False
    return (s.get("elapsed_seconds", 0.0) / audio) < OFFLINE_RATIO


def finish(fig, ax, out: pathlib.Path, name: str, legend: bool = False) -> pathlib.Path:
    if legend:
        ax.legend(loc="best")
    path = out / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_drift(s: dict, out: pathlib.Path) -> pathlib.Path:
    """How fast it is still moving, against how far it has travelled."""
    prev = s["series"]["wer_vs_prev"]
    seed = s["series"]["wer_vs_seed"]
    x = range(1, len(prev) + 1)
    fig, ax = plt.subplots()
    ax.plot(x, seed, color=MACHINE, label="vs. original score")
    ax.plot(x, prev, color=ACCENT, label="vs. previous take")
    fp = s["convergence"].get("fixed_point_at")
    if fp:
        ax.axvline(fp, color=INK, linestyle=":", linewidth=0.8)
        ax.annotate("fixed point", xy=(fp, ax.get_ylim()[1]), xytext=(3, -8),
                    textcoords="offset points", va="top", fontsize=7.5, color=INK)
    ax.set_xlabel("iteration")
    ax.set_ylabel("word error rate")
    ax.set_title("Drift")
    return finish(fig, ax, out, "drift", legend=True)


def fig_confidence(s: dict, out: pathlib.Path) -> pathlib.Path:
    """The crossover: certainty rising as the text stops resembling the score."""
    conf = s["series"]["mean_confidence"]
    overlap = s["series"]["jaccard_vs_seed"]
    x = range(1, len(conf) + 1)
    fig, ax = plt.subplots()
    ax.plot(x, conf, color=MACHINE, label="transcriber mean confidence")
    ax.plot(x, overlap, color=HUMAN, label="vocabulary overlap with score")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_ylim(0, 1)
    # The measured behaviour is flat certainty, not rising certainty -- the
    # title states what the axes show rather than what we expected to see.
    ax.set_title("Certainty is constant while the score recedes")
    return finish(fig, ax, out, "confidence", legend=True)


def fig_discarded(s: dict, out: pathlib.Path) -> pathlib.Path:
    """Punctuation and lexical variety: the performance being deleted."""
    punct = s["series"]["punct_per_100_tokens"]
    ttr = s["series"]["type_token_ratio"]
    x = range(1, len(punct) + 1)
    fig, ax = plt.subplots()
    ax.plot(x, punct, color=MACHINE, label="punctuation per 100 tokens")
    ax.set_xlabel("iteration")
    ax.set_ylabel("punctuation per 100 tokens", color=MACHINE)
    ax.tick_params(axis="y", labelcolor=MACHINE)
    twin = ax.twinx()
    twin.plot(x, ttr, color=HUMAN, label="type-token ratio")
    twin.set_ylabel("type-token ratio", color=HUMAN)
    twin.tick_params(axis="y", labelcolor=HUMAN)
    twin.grid(False)
    twin.spines["top"].set_visible(False)
    ax.set_title("Punctuation and lexical variety")
    return finish(fig, ax, out, "discarded")


def fig_authorship(runs: dict[str, dict], out: pathlib.Path) -> pathlib.Path:
    """Whose words the final text is made of, per seed."""
    names = sorted(runs)
    machine = [runs[n]["authorship"]["machine_share"] for n in names]
    fig, ax = plt.subplots(figsize=(5.0, 0.42 * len(names) + 1.1))
    ax.barh(names, machine, color=MACHINE, height=0.62)
    for i, v in enumerate(machine):
        ax.text(v, i, f" {v:.1%}", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("share of final text entering through mishearing")
    ax.set_xlim(0, max(machine + [0.01]) * 1.28)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_title("Uncredited authorship")
    return finish(fig, ax, out, "authorship")


def fig_convergence(runs: dict[str, dict], out: pathlib.Path) -> pathlib.Path:
    """Every seed's distance from its own score, on shared axes."""
    fig, ax = plt.subplots()
    for name in sorted(runs):
        series = runs[name]["series"]["wer_vs_seed"]
        ax.plot(range(1, len(series) + 1), series, linewidth=1.1, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("word error rate vs. own score")
    ax.set_title("All seeds")
    return finish(fig, ax, out, "convergence", legend=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="outputs/sweep", help="sweep directory")
    ap.add_argument("--run", help="a single run directory instead of a sweep")
    ap.add_argument("--out", default="figures", help="where to write the PDFs")
    ap.add_argument("--allow-offline", action="store_true",
                    help="do not refuse runs produced by the offline simulator")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.run:
        runs = {pathlib.Path(args.run).name: load_summary(pathlib.Path(args.run))}
    else:
        sweep = pathlib.Path(args.sweep)
        if not sweep.is_dir():
            print(f"no sweep at {sweep} -- run the sweep first, or pass --run",
                  file=sys.stderr)
            return 1
        runs = {d.name: load_summary(d) for d in sorted(sweep.iterdir())
                if (d / "summary.json").is_file()}

    if not runs:
        print("no summary.json found", file=sys.stderr)
        return 1

    simulated = [n for n, s in runs.items() if is_offline(s)]
    if simulated and not args.allow_offline:
        print(f"refusing: {', '.join(simulated)} came from the offline simulator.\n"
              f"Re-run those against the API, or pass --allow-offline to plot anyway.",
              file=sys.stderr)
        return 1
    if simulated:
        print(f"warning: plotting simulated runs: {', '.join(simulated)}")

    # The per-iteration figures describe one run; use the longest real one.
    lead = max(runs.items(), key=lambda kv: kv[1]["iterations_run"])
    written = [
        fig_drift(lead[1], out),
        fig_confidence(lead[1], out),
        fig_discarded(lead[1], out),
        fig_authorship(runs, out),
    ]
    if len(runs) > 1:
        written.append(fig_convergence(runs, out))

    print(f"per-iteration figures from '{lead[0]}' ({lead[1]['iterations_run']} iterations)")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
