"""The loop.

    score -> performance -> mishearing -> score -> ...

Lucier re-recorded a tape in a room until the room's resonant frequencies ate
the speech. Here the room is a speech-recognition model, and what it resonates
at is its own language prior.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics as M
from .audio import apply_room, duration_s, encode_wav, read_wav_file, write_wav_file
from .config import SAMPLE_RATE, RunConfig, api_key
from .deepgram import DeepgramClient


@dataclass
class Iteration:
    index: int
    text: str
    audio_path: str | None
    metrics: dict | None
    words: list[dict]


class TranscriptionGapLoop:
    def __init__(self, cfg: RunConfig, client: DeepgramClient | None = None,
                 verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.client = client or DeepgramClient(api_key(), verbose=verbose)

        self.out = Path(cfg.out_dir)
        self.text_dir = self.out / "iterations"
        self.audio_dir = self.out / "audio"
        self.out.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(exist_ok=True)
        if cfg.keep_audio:
            self.audio_dir.mkdir(exist_ok=True)

        self.iterations: list[Iteration] = []
        self.all_substitutions: list[tuple[str, str]] = []

    # -- helpers ----------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _seed_text(self) -> str:
        text = Path(self.cfg.seed_path).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"seed {self.cfg.seed_path} is empty")
        return text

    # -- run --------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg
        seed = self._seed_text()
        started = time.time()

        cfg.save(self.out / "config.json")
        (self.text_dir / "00.txt").write_text(seed + "\n", encoding="utf-8")
        self.iterations.append(Iteration(0, seed, None, None, []))

        self._log(f"seed: {cfg.seed_path}  ({len(M.words(seed))} words)")
        self._log(f"voice: {cfg.performance.voice}   ear: {cfg.listen.model}"
                  f"   smart_format={cfg.listen.smart_format}")
        if cfg.room.active:
            self._log(f"room: {cfg.room}")
        self._log("")

        metrics_file = (self.out / "metrics.jsonl").open("w", encoding="utf-8")
        seen_vocab: set[str] = set(M.words(seed))
        identical_streak = 0

        try:
            for i in range(1, cfg.iterations + 1):
                prev = self.iterations[-1].text
                t0 = time.time()

                # --- perform -------------------------------------------
                if i == 1 and cfg.seed_audio:
                    self._log(f"[{i:02d}] performance: {cfg.seed_audio} (human reading)")
                    samples, rate = read_wav_file(cfg.seed_audio)
                    if rate != SAMPLE_RATE:
                        samples = _resample(samples, rate, SAMPLE_RATE)
                else:
                    samples = self.client.perform(prev, cfg.performance)

                samples = apply_room(samples, cfg.room)
                secs = duration_s(samples)
                wav = encode_wav(samples)

                audio_path = None
                if cfg.keep_audio:
                    audio_path = str(self.audio_dir / f"{i:02d}.wav")
                    write_wav_file(audio_path, samples)

                # --- mishear -------------------------------------------
                heard = self.client.hear(wav, cfg.listen)
                cur = heard.transcript
                if not cur:
                    self._log(f"[{i:02d}] heard nothing -- the loop has gone silent. stopping.")
                    break

                # --- measure -------------------------------------------
                sm = M.step_metrics(
                    i, seed, prev, cur,
                    seen_vocab=seen_vocab,
                    mean_conf=heard.mean_confidence,
                    min_conf=heard.min_confidence,
                    audio_seconds=secs,
                )
                self.all_substitutions.extend(M.align(M.words(prev), M.words(cur)).substitutions)
                seen_vocab |= set(M.words(cur))

                (self.text_dir / f"{i:02d}.txt").write_text(cur + "\n", encoding="utf-8")
                metrics_file.write(json.dumps(sm.to_dict()) + "\n")
                metrics_file.flush()
                self.iterations.append(Iteration(i, cur, audio_path, sm.to_dict(), heard.words))

                self._log(
                    f"[{i:02d}] {secs:5.1f}s audio  "
                    f"WER/prev {sm.wer_vs_prev:6.3f}  WER/seed {sm.wer_vs_seed:6.3f}  "
                    f"{sm.surface['tokens']:4d} tok  conf {sm.mean_confidence:.3f}  "
                    f"+{sm.new_words} new  ({time.time() - t0:.1f}s)"
                )
                subs = M.align(M.words(prev), M.words(cur)).substitutions[:4]
                if subs:
                    self._log("     " + "; ".join(f"{a} -> {b}" for a, b in subs))

                # --- has it stopped moving? ----------------------------
                if M.normalize(cur) == M.normalize(prev):
                    identical_streak += 1
                    if cfg.stop_on_fixed_point and identical_streak >= cfg.fixed_point_patience:
                        self._log(f"\nfixed point: {identical_streak} identical iterations. "
                                  f"the loop has found its attractor.")
                        break
                else:
                    identical_streak = 0
        finally:
            metrics_file.close()

        report = self.summarize(seed, elapsed=time.time() - started)
        (self.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._write_manifest()
        return report

    # -- summary ----------------------------------------------------------

    def summarize(self, seed: str | None = None, elapsed: float | None = None) -> dict:
        seed = seed if seed is not None else self.iterations[0].text
        texts = [it.text for it in self.iterations]
        final = texts[-1]

        conv = M.detect_convergence(texts)
        auth = M.authorship(seed, final, texts)
        steps = [it.metrics for it in self.iterations if it.metrics]

        def series(key: str) -> list[float]:
            return [float(s[key]) for s in steps]

        summary = {
            "seed_path": self.cfg.seed_path,
            "iterations_run": len(self.iterations) - 1,
            "iterations_requested": self.cfg.iterations,
            "elapsed_seconds": elapsed,
            "voice": self.cfg.performance.voice,
            "listen_model": self.cfg.listen.model,
            "smart_format": self.cfg.listen.smart_format,
            "room_active": self.cfg.room.active,
            "convergence": conv,
            "authorship": auth,
            "series": {
                "wer_vs_prev": series("wer_vs_prev") if steps else [],
                "wer_vs_seed": series("wer_vs_seed") if steps else [],
                "cer_vs_seed": series("cer_vs_seed") if steps else [],
                "jaccard_vs_seed": series("jaccard_vs_seed") if steps else [],
                "cosine_vs_seed": series("cosine_vs_seed") if steps else [],
                "mean_confidence": series("mean_confidence") if steps else [],
                "tokens": [s["surface"]["tokens"] for s in steps],
                "type_token_ratio": [s["surface"]["type_token_ratio"] for s in steps],
                "punct_per_100_tokens": [s["surface"]["punct_per_100_tokens"] for s in steps],
                "audio_seconds": series("audio_seconds") if steps else [],
            },
            "substitution_ledger": M.substitution_ledger(self.all_substitutions),
            "seed_text": seed,
            "final_text": final,
        }
        if steps:
            summary["total_drift_from_seed"] = steps[-1]["wer_vs_seed"]
            summary["final_delta"] = steps[-1]["wer_vs_prev"]
        return summary

    def _write_manifest(self) -> None:
        manifest = {
            "config": self.cfg.to_dict(),
            "iterations": [
                {
                    "index": it.index,
                    "text_path": f"iterations/{it.index:02d}.txt",
                    "audio_path": (Path(it.audio_path).name if it.audio_path else None),
                    "tokens": len(M.words(it.text)),
                }
                for it in self.iterations
            ],
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # -- reload a finished run -------------------------------------------

    @classmethod
    def from_output_dir(cls, out_dir: str | Path) -> "TranscriptionGapLoop":
        """Rebuild a loop object from disk so analysis/reporting can run
        without touching the API again."""
        out = Path(out_dir)
        cfg = RunConfig.load(out / "config.json")
        cfg.out_dir = str(out)
        loop = cls.__new__(cls)
        loop.cfg = cfg
        loop.verbose = False
        loop.client = None  # type: ignore[assignment]
        loop.out = out
        loop.text_dir = out / "iterations"
        loop.audio_dir = out / "audio"
        loop.iterations = []
        loop.all_substitutions = []

        metrics_by_index: dict[int, dict] = {}
        mpath = out / "metrics.jsonl"
        if mpath.exists():
            for line in mpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    m = json.loads(line)
                    metrics_by_index[m["iteration"]] = m

        for p in sorted(loop.text_dir.glob("*.txt")):
            idx = int(p.stem)
            text = p.read_text(encoding="utf-8").strip()
            audio = loop.audio_dir / f"{idx:02d}.wav"
            loop.iterations.append(Iteration(
                idx, text, str(audio) if audio.exists() else None,
                metrics_by_index.get(idx), [],
            ))
        loop.iterations.sort(key=lambda it: it.index)
        for a, b in zip(loop.iterations, loop.iterations[1:]):
            loop.all_substitutions.extend(
                M.align(M.words(a.text), M.words(b.text)).substitutions
            )
        return loop


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample. Good enough: we are about to hand this to a model that
    is going to mishear it on purpose."""
    if src == dst or len(x) == 0:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(
        np.linspace(0, len(x) - 1, n, dtype=np.float64),
        np.arange(len(x), dtype=np.float64),
        x.astype(np.float64),
    ).astype(np.float32)
