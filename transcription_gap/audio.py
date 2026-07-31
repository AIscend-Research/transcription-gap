"""Mono 16-bit audio helpers. stdlib ``wave`` plus numpy, nothing else."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, RoomConfig


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes -> float32 mono in [-1, 1], plus sample rate."""
    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def encode_wav(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    """float32 mono -> WAV bytes."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def read_wav_file(path: str | Path) -> tuple[np.ndarray, int]:
    return decode_wav(Path(path).read_bytes())


def write_wav_file(path: str | Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    Path(path).write_bytes(encode_wav(samples, rate))


def silence(ms: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(max(0, int(rate * ms / 1000.0)), dtype=np.float32)


def concat(chunks: list[np.ndarray]) -> np.ndarray:
    real = [c for c in chunks if c is not None and len(c)]
    if not real:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(real).astype(np.float32)


def duration_s(samples: np.ndarray, rate: int = SAMPLE_RATE) -> float:
    return len(samples) / float(rate)


# --- the room ------------------------------------------------------------


def _one_pole_lowpass(x: np.ndarray, cutoff_hz: float, rate: int) -> np.ndarray:
    """Single-pole IIR lowpass. Deliberately cheap and a little ugly; a
    telephone band is not a nice filter either."""
    if cutoff_hz <= 0 or cutoff_hz >= rate / 2:
        return x
    dt = 1.0 / rate
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    a = dt / (rc + dt)
    out = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):  # sample-serial by nature; runs fine at 24k mono
        acc += a * (x[i] - acc)
        out[i] = acc
    return out


def _comb_reverb(x: np.ndarray, decay: float, delay_ms: float, rate: int) -> np.ndarray:
    """Feedback comb filter: the crudest possible standing-wave. Vectorised
    per-delay-block so a 30s take stays instant."""
    if decay <= 0:
        return x
    d = max(1, int(rate * delay_ms / 1000.0))
    y = x.astype(np.float32).copy()
    g = float(np.clip(decay, 0.0, 0.95))
    for start in range(d, len(y), d):
        end = min(start + d, len(y))
        y[start:end] += g * y[start - d:start - d + (end - start)]
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 1.0:
        y /= peak
    return y


def apply_room(samples: np.ndarray, room: RoomConfig, rate: int = SAMPLE_RATE,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Push the performance through the channel before the machine hears it."""
    if not room.active:
        return samples
    y = samples.astype(np.float32).copy()

    if room.room_decay > 0:
        y = _comb_reverb(y, room.room_decay, room.room_delay_ms, rate)
    if room.lowpass_hz > 0:
        y = _one_pole_lowpass(y, room.lowpass_hz, rate)
    if room.gain_db != 0.0:
        y *= float(10.0 ** (room.gain_db / 20.0))
    if room.noise_db > -90.0:
        rng = rng or np.random.default_rng()
        amp = float(10.0 ** (room.noise_db / 20.0))
        y = y + rng.normal(0.0, amp, size=y.shape).astype(np.float32)
    if room.clip < 1.0:
        y = np.clip(y, -room.clip, room.clip)

    return np.clip(y, -1.0, 1.0)
