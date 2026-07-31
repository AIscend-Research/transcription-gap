"""transcription-gap: make the errors the agent.

A spoken-word score is performed, transcribed, and the transcript becomes the
score for the next performance. Nobody edits anything. The piece is whatever
the speech recogniser's mishearings accumulate into.
"""

from .config import ListenConfig, PerformanceConfig, RoomConfig, RunConfig
from .loop import TranscriptionGapLoop
from .report import build_html, write_report

__all__ = [
    "TranscriptionGapLoop",
    "RunConfig",
    "PerformanceConfig",
    "ListenConfig",
    "RoomConfig",
    "write_report",
    "build_html",
]
__version__ = "0.1.0"
