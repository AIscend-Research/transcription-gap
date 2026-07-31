"""A stand-in transcriber, for testing the loop without spending API calls.

This is **not** a speech model and it does not pretend to be. It is a scripted
caricature of the three things a real recogniser does to a performance:

  1. hears a word as a different, more probable word (phonetic confusion),
  2. drops unstressed function words entirely,
  3. silently rewrites punctuation and capitalisation to its own house style.

It exists so `--offline` can exercise the whole pipeline -- metrics,
convergence detection, reporting -- deterministically and for free. Every
number produced this way is a simulation; only a real Deepgram run says
anything about a real transcriber. The report header records which was used.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .audio import silence
from .config import ListenConfig, PerformanceConfig
from .deepgram import Hearing

# Words a language model likes better than what was actually said.
_CONFUSIONS: dict[str, list[str]] = {
    "sitting": ["setting"], "setting": ["sitting"],
    "room": ["rheum", "room"], "different": ["deferent"],
    "voice": ["voiced", "vice"], "resonant": ["resident"],
    "resident": ["president"], "frequencies": ["frequency's"],
    "speech": ["speeches", "each"], "smooth": ["smoothed"],
    "except": ["accept"], "accept": ["except"],
    "hear": ["here"], "here": ["hear"],
    "heard": ["herd"], "herd": ["heard"],
    "their": ["there"], "there": ["their"],
    "its": ["it's"], "it's": ["its"],
    "whose": ["who's"], "who's": ["whose"],
    "than": ["then"], "then": ["than"],
    "machine": ["machines"], "machines": ["machine"],
    "listen": ["listened"], "listening": ["listing"],
    "recording": ["reporting"], "score": ["scored", "store"],
    "performance": ["performances"], "author": ["other"],
    "other": ["author"], "words": ["word"],
    "again": ["a gain"], "into": ["in to"],
    "one": ["won"], "won": ["one"],
    "for": ["four"], "four": ["for"],
    "loud": ["allowed"], "aloud": ["allowed"],
    "presence": ["presents"], "credit": ["credits"],
    "silence": ["silent"], "silent": ["silence"],
    "gap": ["cap"], "cap": ["gap"],
    "drift": ["drifts"], "faithful": ["faithfully"],
}

_DROPPABLE = {"a", "the", "of", "to", "that", "and", "in", "is", "it", "as", "so", "any"}


class OfflineClient:
    """Duck-types :class:`~transcription_gap.deepgram.DeepgramClient`.

    ``perform`` returns real silence of a plausible length -- the audio is a
    placeholder -- and stashes the text so ``hear`` has something to corrupt.
    The loop only ever calls them in that order.
    """

    def __init__(self, verbose: bool = True, drop_rate: float = 0.035,
                 confuse_rate: float = 0.16):
        self.verbose = verbose
        self.drop_rate = drop_rate
        self.confuse_rate = confuse_rate
        self._pending = ""
        if verbose:
            print("    (offline: simulated transcriber, no API calls, results are not "
                  "measurements of a real speech model)")

    # -- perform ----------------------------------------------------------

    def perform(self, text: str, perf: PerformanceConfig) -> np.ndarray:
        self._pending = text
        # ~2.6 words/second plus the performed pauses, so audio_seconds in the
        # metrics stays a meaningful number.
        n_words = len(text.split())
        secs = max(0.5, n_words / 2.6)
        pauses = len(re.findall(r"[.!?;:,]", text)) * (perf.pause_short_ms / 1000.0)
        return silence((secs + pauses) * 1000.0)

    # -- hear -------------------------------------------------------------

    def hear(self, wav_bytes: bytes, cfg: ListenConfig) -> Hearing:
        text = self._pending
        # Seeded on the text itself: same input, same mishearing, every time.
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)

        toks = text.split()
        out: list[str] = []
        confidences: list[float] = []
        keyterms = {k.lower() for k in cfg.keyterms}

        for tok in toks:
            bare = re.sub(r"^\W+|\W+$", "", tok).lower()
            lead = tok[: len(tok) - len(tok.lstrip("\"'([")) ]
            trail = tok[len(tok.rstrip(".,;:!?\"')]")) :]
            conf = 0.97

            if bare and bare not in keyterms:
                if bare in _DROPPABLE and rng.random() < self.drop_rate:
                    continue  # unstressed, unheard
                if bare in _CONFUSIONS and rng.random() < self.confuse_rate:
                    options = _CONFUSIONS[bare]
                    bare = options[int(rng.integers(len(options)))]
                    conf = 0.55 + 0.25 * float(rng.random())
                elif rng.random() < 0.02 and len(bare) > 4:
                    # generic phonetic slip on a long word
                    i = int(rng.integers(1, len(bare) - 1))
                    bare = bare[:i] + bare[i + 1:]
                    conf = 0.48 + 0.3 * float(rng.random())

            word = lead + bare + trail if bare else tok
            out.append(word)
            confidences.append(conf)

        transcript = " ".join(out)
        transcript = _smart_format(transcript) if cfg.smart_format else transcript.lower()

        words = [
            {"word": re.sub(r"\W", "", w).lower(), "confidence": c}
            for w, c in zip(out, confidences)
        ]
        self._pending = ""
        return Hearing(transcript=transcript, words=words, raw={"offline": True})


def _smart_format(text: str) -> str:
    """The silent editor: one flat paragraph, house-style punctuation.

    The real feature does more than this, but it does at least this -- and
    notice that it deletes every line break the performance was written with.
    """
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = t.replace(" — ", ", ").replace("—", ", ").replace("…", ".")
    t = re.sub(r"[;:]", ",", t)
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p[:1].upper() + p[1:] if p else p for p in parts]
    t = " ".join(parts)
    if t and t[-1] not in ".!?":
        t += "."
    return t
