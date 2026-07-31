"""Thin Deepgram client: /v1/speak to perform, /v1/listen to mishear.

One API does both halves of the loop, which is why this whole piece fits on a
laptop with a single key.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import requests

from .audio import concat, decode_wav, silence
from .config import (
    SAMPLE_RATE,
    SPEAK_CHAR_LIMIT,
    ListenConfig,
    PerformanceConfig,
)

SPEAK_URL = "https://api.deepgram.com/v1/speak"
LISTEN_URL = "https://api.deepgram.com/v1/listen"

FALLBACK_VOICE = "aura-asteria-en"  # aura-1, available on every plan

# Split points that a human reader would breathe at. The captured group keeps
# the punctuation attached to the phrase it ends.
_BREATH_RE = re.compile(r"(?<=[.!?;:,—–])\s+|\n+")
_LONG_PAUSE_CHARS = set(".!?…")


@dataclass
class Hearing:
    """What the machine came back with."""

    transcript: str
    words: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def mean_confidence(self) -> float:
        vals = [w.get("confidence") for w in self.words if w.get("confidence") is not None]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def min_confidence(self) -> float:
        vals = [w.get("confidence") for w in self.words if w.get("confidence") is not None]
        return float(np.min(vals)) if vals else float("nan")

    def low_confidence_words(self, threshold: float = 0.80) -> list[dict]:
        return [
            w for w in self.words
            if w.get("confidence") is not None and w["confidence"] < threshold
        ]


class DeepgramError(RuntimeError):
    pass


class DeepgramClient:
    def __init__(self, api_key: str, timeout: int = 180, max_retries: int = 4,
                 verbose: bool = True):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.session = requests.Session()
        self._voice_fallback_warned = False

    # -- plumbing ---------------------------------------------------------

    def _post(self, url: str, *, params: dict, headers: dict, data: bytes) -> requests.Response:
        headers = {"Authorization": f"Token {self.api_key}", **headers}
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(
                    url, params=params, headers=headers, data=data, timeout=self.timeout
                )
            except requests.RequestException as e:  # network flake
                last = e
            else:
                if r.status_code < 400:
                    return r
                if r.status_code in (429, 500, 502, 503, 504):
                    last = DeepgramError(f"{r.status_code}: {r.text[:300]}")
                else:
                    raise DeepgramError(
                        f"Deepgram {r.status_code} on {url}: {r.text[:500]}"
                    )
            wait = 2.0 ** attempt
            if self.verbose:
                print(f"    ! retry {attempt + 1}/{self.max_retries} in {wait:.0f}s ({last})")
            time.sleep(wait)
        raise DeepgramError(f"gave up on {url} after {self.max_retries} tries: {last}")

    # -- speak ------------------------------------------------------------

    def _speak_once(self, text: str, voice: str) -> np.ndarray:
        params = {
            "model": voice,
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE),
            "container": "wav",
        }
        try:
            r = self._post(
                SPEAK_URL,
                params=params,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"text": text}).encode("utf-8"),
            )
        except DeepgramError as e:
            # An unavailable Aura-2 voice shouldn't end the run.
            if voice != FALLBACK_VOICE and ("400" in str(e) or "404" in str(e)):
                if not self._voice_fallback_warned:
                    print(f"    ! voice {voice} unavailable, falling back to {FALLBACK_VOICE}")
                    self._voice_fallback_warned = True
                return self._speak_once(text, FALLBACK_VOICE)
            raise
        samples, rate = decode_wav(r.content)
        if rate != SAMPLE_RATE:
            raise DeepgramError(f"expected {SAMPLE_RATE} Hz back, got {rate}")
        return samples

    def perform(self, text: str, perf: PerformanceConfig) -> np.ndarray:
        """Turn a score into a performance.

        With ``breath_groups`` on, each phrase is synthesised separately and
        real silence is inserted between them. The phrasing therefore exists
        only in the waveform -- the transcript has nowhere to put it, and that
        is the first thing every iteration loses.
        """
        pieces = _breath_groups(text) if perf.breath_groups else [text.strip()]
        pieces = [p for p in pieces if p.strip()]
        if not pieces:
            raise ValueError("nothing to perform: the score is empty")

        out: list[np.ndarray] = [silence(perf.lead_silence_ms)]
        for i, piece in enumerate(pieces):
            for chunk in _hard_wrap(piece, SPEAK_CHAR_LIMIT):
                out.append(self._speak_once(chunk, perf.voice))
            if i < len(pieces) - 1:
                tail = piece.rstrip()[-1:] if piece.rstrip() else ""
                pause = perf.pause_long_ms if tail in _LONG_PAUSE_CHARS else perf.pause_short_ms
                out.append(silence(pause))
        out.append(silence(perf.tail_silence_ms))
        return concat(out)

    # -- listen -----------------------------------------------------------

    def hear(self, wav_bytes: bytes, cfg: ListenConfig) -> Hearing:
        params = {
            "model": cfg.model,
            "language": cfg.language,
            "smart_format": str(bool(cfg.smart_format)).lower(),
            "punctuate": str(bool(cfg.punctuate)).lower(),
            "numerals": str(bool(cfg.numerals)).lower(),
        }
        if cfg.keyterms:
            # nova-3 calls them keyterms; harmless elsewhere
            params["keyterm"] = cfg.keyterms
        r = self._post(
            LISTEN_URL,
            params=params,
            headers={"Content-Type": "audio/wav"},
            data=wav_bytes,
        )
        payload = r.json()
        try:
            alt = payload["results"]["channels"][0]["alternatives"][0]
        except (KeyError, IndexError) as e:
            raise DeepgramError(f"unexpected /listen payload: {str(payload)[:400]}") from e
        return Hearing(
            transcript=(alt.get("transcript") or "").strip(),
            words=alt.get("words") or [],
            raw=payload,
        )


# --- text -> breath groups -----------------------------------------------


def _breath_groups(text: str) -> list[str]:
    parts = _BREATH_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _hard_wrap(text: str, limit: int) -> list[str]:
    """Last-resort splitter for a phrase longer than the API will take."""
    text = text.strip()
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > limit:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out
