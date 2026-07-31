"""Run configuration for a transcription-gap loop."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

DEFAULT_SPEAK_MODEL = "aura-2-thalia-en"
DEFAULT_LISTEN_MODEL = "nova-3"

# Deepgram /v1/speak rejects very long bodies; we chunk well under the limit
# anyway because breath-groups are how the performance gets its phrasing.
SPEAK_CHAR_LIMIT = 1800

SAMPLE_RATE = 24000  # linear16 mono, what we ask Aura for and hand back to Nova


@dataclass
class RoomConfig:
    """The channel the performance passes through on its way to being heard.

    Lucier's room was a physical resonance. Ours is whatever we do to the
    waveform between speaking and listening. All of it is optional: with the
    defaults everything is off and the only degradation is the transcriber's
    own bias, which is the point of the piece.
    """

    noise_db: float = -90.0        # additive white noise floor, dBFS
    gain_db: float = 0.0           # pre-listen gain
    lowpass_hz: float = 0.0        # 0 = no filter; try 3400 for telephone
    room_decay: float = 0.0        # 0..1 crude feedback-comb reverberance
    room_delay_ms: float = 45.0    # spacing of that comb
    clip: float = 1.0              # hard-clip ceiling, 1.0 = none

    @property
    def active(self) -> bool:
        return (
            self.noise_db > -90.0
            or self.gain_db != 0.0
            or self.lowpass_hz > 0
            or self.room_decay > 0
            or self.clip < 1.0
        )


@dataclass
class PerformanceConfig:
    """How the text is turned back into a spoken performance.

    ``breath_groups`` is the load-bearing one. Splitting on punctuation and
    inserting real silence means the phrasing lives in the audio, where the
    transcript has no column for it. Each pass through the loop, that
    performance information is what gets thrown away first.
    """

    breath_groups: bool = True
    pause_short_ms: int = 180      # after , ; : and dashes
    pause_long_ms: int = 520       # after . ? ! and line breaks
    lead_silence_ms: int = 120
    tail_silence_ms: int = 400
    voice: str = DEFAULT_SPEAK_MODEL


@dataclass
class ListenConfig:
    """How the machine hears. ``smart_format`` is the silent editor: it
    repunctuates, capitalises and 'corrects' without ever being asked."""

    model: str = DEFAULT_LISTEN_MODEL
    smart_format: bool = True
    punctuate: bool = True
    numerals: bool = False
    language: str = "en"
    keyterms: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    seed_path: str = "seeds/lucier.txt"
    out_dir: str = "outputs/run"
    iterations: int = 12
    keep_audio: bool = True
    stop_on_fixed_point: bool = True
    fixed_point_patience: int = 2   # identical transcripts in a row before stopping
    seed_audio: str | None = None   # start from a human reading instead of Aura
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    listen: ListenConfig = field(default_factory=ListenConfig)
    room: RoomConfig = field(default_factory=RoomConfig)

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        nested = {
            "performance": PerformanceConfig,
            "listen": ListenConfig,
            "room": RoomConfig,
        }
        for key, klass in nested.items():
            if isinstance(d.get(key), dict):
                allowed = {f.name for f in fields(klass)}
                d[key] = klass(**{k: v for k, v in d[key].items() if k in allowed})
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def api_key() -> str:
    """Deepgram key from the environment, with a .env fallback so the piece
    runs on a laptop without anyone setting up a shell profile."""
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if key:
        return key
    for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("DEEPGRAM_API_KEY") and "=" in line:
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val:
                        return val
    raise RuntimeError(
        "No DEEPGRAM_API_KEY found. Put it in your environment or in a .env "
        "file next to this project (see .env.example)."
    )
