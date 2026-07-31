"""Command line entry point.

    python -m transcription_gap run     --seed seeds/lucier.txt -n 12
    python -m transcription_gap sweep   --seeds seeds/*.txt -n 8
    python -m transcription_gap report  outputs/run
    python -m transcription_gap analyze outputs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .analyze import analyze
from .config import (
    DEFAULT_LISTEN_MODEL,
    DEFAULT_SPEAK_MODEL,
    ListenConfig,
    PerformanceConfig,
    Rates,
    RoomConfig,
    RunConfig,
    api_key,
)
from .loop import TranscriptionGapLoop
from .report import write_report


def _expand_seeds(patterns: list[str]) -> list[Path]:
    """Accept literal paths and glob patterns, whether or not the shell already
    expanded them."""
    out: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            out.extend(sorted(Path().glob(pattern)))
        else:
            out.append(Path(pattern))
    seen: set[Path] = set()
    return [p for p in out if p.exists() and not (p in seen or seen.add(p))]


def _cfg_from_args(a: argparse.Namespace, seed: str, out_dir: str) -> RunConfig:
    return RunConfig(
        seed_path=seed,
        out_dir=out_dir,
        iterations=a.iterations,
        keep_audio=not a.no_audio,
        stop_on_fixed_point=not a.run_to_end,
        seed_audio=getattr(a, "seed_audio", None),
        performance=PerformanceConfig(
            voice=a.voice,
            breath_groups=not a.no_breath_groups,
        ),
        listen=ListenConfig(
            model=a.listen_model,
            smart_format=not a.no_smart_format,
            punctuate=not a.no_smart_format,
            keyterms=a.keyterm or [],
        ),
        room=RoomConfig(
            noise_db=a.noise_db,
            gain_db=a.gain_db,
            lowpass_hz=a.lowpass,
            room_decay=a.room_decay,
        ),
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", "--iterations", type=int, default=12,
                   help="how many times to go round the loop (default 12)")
    p.add_argument("--voice", default=DEFAULT_SPEAK_MODEL, help="Deepgram Aura voice")
    p.add_argument("--listen-model", default=DEFAULT_LISTEN_MODEL, help="Deepgram STT model")
    p.add_argument("--no-smart-format", action="store_true",
                   help="turn off the silent editor (punctuation/capitalisation/'corrections')")
    p.add_argument("--no-breath-groups", action="store_true",
                   help="synthesise the whole text in one pass, no performed phrasing")
    p.add_argument("--no-audio", action="store_true", help="do not keep the wav files")
    p.add_argument("--run-to-end", action="store_true",
                   help="keep iterating even after a fixed point is reached")
    p.add_argument("--keyterm", action="append", help="bias the transcriber (repeatable)")
    p.add_argument("--noise-db", type=float, default=-90.0, help="noise floor in dBFS")
    p.add_argument("--gain-db", type=float, default=0.0)
    p.add_argument("--lowpass", type=float, default=0.0,
                   help="lowpass cutoff in Hz, e.g. 3400 for a telephone channel")
    p.add_argument("--room-decay", type=float, default=0.0,
                   help="0..0.95 crude standing-wave feedback, Lucier's actual room")
    p.add_argument("--offline", action="store_true",
                   help="run with a local stand-in transcriber (no API, no key, for testing)")


def _client(offline: bool, verbose: bool = True):
    if offline:
        from .offline import OfflineClient
        return OfflineClient(verbose=verbose)
    from .deepgram import DeepgramClient
    return DeepgramClient(api_key(), verbose=verbose)


def cmd_run(a: argparse.Namespace) -> int:
    cfg = _cfg_from_args(a, a.seed, a.out)
    loop = TranscriptionGapLoop(cfg, client=_client(a.offline))
    summary = loop.run()
    path = write_report(cfg.out_dir, title=a.title)
    print()
    print(f"drift from score : {summary.get('total_drift_from_seed', float('nan')):.3f} WER")
    print(f"outcome          : {summary['convergence']['status']}")
    share = summary["authorship"].get("machine_share")
    if share == share:
        print(f"transcriber wrote: {100 * share:.1f}% of the final text")
    c = summary["cost"]
    print(f"cost             : ${c['total_usd']:.4f}  "
          f"({c['tts_chars']:,} chars + {c['stt_seconds'] / 60:.1f} min)"
          + ("  [offline, nothing billed]" if a.offline else ""))
    print(f"report           : {path}")
    return 0


def cmd_sweep(a: argparse.Namespace) -> int:
    """One loop per seed, same transcriber, then the cross-run analysis.

    This is the experiment that actually tests the attractor claim: different
    starting texts, identical machine, do they end up in the same place.
    """
    seeds = _expand_seeds(a.seeds)
    if not seeds:
        print("no seed files matched", file=sys.stderr)
        return 1

    root = Path(a.out)
    client = _client(a.offline)
    billed = 0.0
    for seed in seeds:
        out = root / seed.stem
        print(f"\n=== {seed.name} -> {out} ===")
        cfg = _cfg_from_args(a, str(seed), str(out))
        summary = TranscriptionGapLoop(cfg, client=client).run()
        billed += summary["cost"]["total_usd"]
        write_report(str(out))

    md = analyze(root, root / "analysis.md")
    print(f"\ntotal cost across {len(seeds)} runs: ${billed:.3f}"
          + ("  [offline, nothing billed]" if a.offline else ""))
    print()
    print(md.split("## Final texts")[0])
    print(f"written: {root / 'analysis.md'}")
    return 0


def cmd_estimate(a: argparse.Namespace) -> int:
    """Price a plan before spending anything.

    Both billable quantities scale with the length of the text, and the text
    stays roughly the same length across iterations (it loses a few percent of
    its tokens per pass), so seed length x iterations is a good estimate.
    """
    rates = Rates(tts_per_1k_chars=a.rate_tts, stt_per_minute=a.rate_stt)
    seeds = _expand_seeds(a.seeds)
    if not seeds:
        print("no seed files matched", file=sys.stderr)
        return 1

    total_chars = 0
    total_secs = 0.0
    rows = []
    for seed in seeds:
        text = seed.read_text(encoding="utf-8").strip()
        chars = len(text) * a.iterations
        # Aura speaks at roughly 2.6 words/second; performed pauses add to that.
        n_words = len(text.split())
        secs_per_take = n_words / 2.6 + len(re.findall(r"[.!?;:,\n]", text)) * 0.25
        secs = secs_per_take * a.iterations
        c = rates.cost(chars, secs)
        rows.append((seed.name, n_words, c["total_usd"]))
        total_chars += chars
        total_secs += secs

    print(f"{len(seeds)} seed(s) x {a.iterations} iterations, "
          f"at ${a.rate_tts}/1k chars TTS and ${a.rate_stt}/min STT\n")
    for name, n_words, usd in rows:
        print(f"  {name:<20} {n_words:>4} words   ${usd:>6.3f}")
    total = rates.cost(total_chars, total_secs)
    print(f"\n  {'TTS':<20} {total['tts_chars']:>7,} chars   ${total['tts_usd']:>6.3f}")
    print(f"  {'STT':<20} {total['stt_seconds'] / 60:>7.1f} min     ${total['stt_usd']:>6.3f}")
    print(f"  {'TOTAL':<20} {'':>7}         ${total['total_usd']:>6.3f}")
    print("\nEstimate only. Runs that hit a fixed point stop early and cost less; "
          "rates are as published on 2026-07-31 and may have changed.")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    print(write_report(a.out_dir, title=a.title))
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    md = analyze(a.root, a.out)
    print(md if not a.out else f"written: {a.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcription_gap",
        description="Perform a text, transcribe it, make the transcript the next score. "
                    "Measure where it lands.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one loop, start to finish")
    r.add_argument("--seed", default="seeds/lucier.txt")
    r.add_argument("--out", default="outputs/run")
    r.add_argument("--title", default=None)
    r.add_argument("--seed-audio", default=None,
                   help="16-bit wav of a human reading the score, used for iteration 1")
    _add_common(r)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="one loop per seed, then cross-run attractor analysis")
    s.add_argument("--seeds", nargs="+", default=["seeds/*.txt"])
    s.add_argument("--out", default="outputs/sweep")
    _add_common(s)
    s.set_defaults(func=cmd_sweep)

    e = sub.add_parser("estimate", help="price a plan before spending anything")
    e.add_argument("--seeds", nargs="+", default=["seeds/*.txt"])
    e.add_argument("-n", "--iterations", type=int, default=25)
    e.add_argument("--rate-tts", type=float, default=Rates().tts_per_1k_chars,
                   help="USD per 1000 TTS characters")
    e.add_argument("--rate-stt", type=float, default=Rates().stt_per_minute,
                   help="USD per minute of STT audio")
    e.set_defaults(func=cmd_estimate)

    rp = sub.add_parser("report", help="re-render report.html for a finished run")
    rp.add_argument("out_dir")
    rp.add_argument("--title", default=None)
    rp.set_defaults(func=cmd_report)

    an = sub.add_parser("analyze", help="markdown analysis across every run under a directory")
    an.add_argument("root", nargs="?", default="outputs")
    an.add_argument("--out", default=None)
    an.set_defaults(func=cmd_analyze)

    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
