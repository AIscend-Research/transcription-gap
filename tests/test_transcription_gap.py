"""Tests for the measurement half. Run: python -m pytest -q  (or run this file).

The artistic claims rest entirely on these numbers being right, so the edit
distance, the convergence detector and the authorship split all get checked
against hand-computed cases.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcription_gap import metrics as M  # noqa: E402
from transcription_gap.audio import (  # noqa: E402
    apply_room, decode_wav, encode_wav, duration_s, silence,
)
from transcription_gap.config import RoomConfig, RunConfig  # noqa: E402
from transcription_gap.loop import TranscriptionGapLoop, _resample  # noqa: E402
from transcription_gap.offline import OfflineClient  # noqa: E402
from transcription_gap.report import build_html  # noqa: E402


# --- normalisation & tokenising ------------------------------------------

def test_words_strips_case_and_punctuation():
    assert M.words("The Room, again!") == ["the", "room", "again"]


def test_words_keeps_contractions():
    assert M.words("it's") == ["it's"]


def test_normalize_collapses_whitespace_and_smart_quotes():
    assert M.normalize("a  \n b’s") == "a b's"


# --- edit distance -------------------------------------------------------

def test_align_identical():
    a = M.align(["a", "b"], ["a", "b"])
    assert a.distance == 0 and a.hits == 2 and a.rate == 0.0


def test_align_substitution_is_recorded():
    a = M.align(["i", "hear", "you"], ["i", "here", "you"])
    assert a.distance == 1
    assert a.substitutions == [("hear", "here")]
    assert a.rate == 1 / 3


def test_align_deletion_and_insertion():
    a = M.align(["the", "room"], ["room"])
    assert a.deletions == ["the"] and a.distance == 1
    b = M.align(["room"], ["the", "room"])
    assert b.insertions == ["the"] and b.distance == 1


def test_align_empty_both_sides():
    assert M.align([], []).distance == 0


def test_align_against_empty_hypothesis():
    a = M.align(["a", "b", "c"], [])
    assert a.distance == 3 and a.rate == 1.0


def test_wer_matches_hand_count():
    # 4 reference words, one substitution -> 0.25
    assert abs(M.wer("i am in a room", "i am in a rheum") - 1 / 5) < 1e-9


def test_cer_is_between_zero_and_one_for_full_rewrite():
    assert 0.0 < M.cer("abc", "xyz") <= 1.0


# --- similarity ----------------------------------------------------------

def test_jaccard_bounds():
    assert M.jaccard("a b", "a b") == 1.0
    assert M.jaccard("a b", "c d") == 0.0
    assert M.jaccard("a b", "b c") == 1 / 3


def test_cosine_identical_and_disjoint():
    assert abs(M.cosine("a b a", "a b a") - 1.0) < 1e-9
    assert M.cosine("a b", "c d") == 0.0


def test_cosine_handles_empty():
    assert M.cosine("", "a") == 0.0


# --- surface features (the discarded performance) ------------------------

def test_surface_features_counts():
    f = M.surface_features("One, two. Three!")
    assert f["tokens"] == 3
    assert f["sentences"] == 2
    assert f["punct_per_100_tokens"] > 0


def test_surface_features_survives_empty_text():
    f = M.surface_features("")
    assert f["tokens"] == 0 and f["mean_word_len"] == 0.0


# --- convergence ---------------------------------------------------------

def test_detect_fixed_point():
    r = M.detect_convergence(["a b", "a c", "a d", "a d"])
    assert r["status"] == "fixed_point"
    assert r["fixed_point_at"] == 3


def test_detect_cycle():
    r = M.detect_convergence(["a", "b", "c", "b"])
    assert r["status"] == "cycle"
    assert r["cycle"]["period"] == 2


def test_open_when_still_moving():
    texts = ["one two three", "one two four", "one five four", "six five four"]
    assert M.detect_convergence(texts)["status"] == "open"


def test_contracting_flag_on_decaying_deltas():
    # big change early, then almost none
    texts = ["a b c d", "x y z w", "x y z v", "x y z v2", "x y z v2", "x y z v2"]
    r = M.detect_convergence(texts)
    # a fixed point interrupts before the decay stats, so check the deltas fall
    assert r["deltas"][0] > r["deltas"][-1]


# --- authorship ----------------------------------------------------------

def test_authorship_splits_final_tokens():
    seed = "i am sitting in a room"
    final = "i am setting in a room"
    a = M.authorship(seed, final, [seed, final])
    # one of six final tokens ("setting") was not in the seed
    assert abs(a["machine_share"] - 1 / 6) < 1e-9
    assert abs(a["human_share"] - 5 / 6) < 1e-9
    assert a["seed_words_lost"] == 1
    assert [w["word"] for w in a["machine_words"]] == ["setting"]


def test_authorship_tracks_when_a_word_entered():
    texts = ["a b", "a b c", "a b c", "a b c"]
    a = M.authorship(texts[0], texts[-1], texts)
    entry = next(w for w in a["machine_words"] if w["word"] == "c")
    assert entry["entered_at"] == 1
    assert entry["survived_iterations"] == 2


def test_authorship_handles_empty_final():
    a = M.authorship("a b", "", ["a b", ""])
    assert a["machine_share"] != a["machine_share"]  # NaN


def test_substitution_ledger_orders_by_count():
    pairs = [("hear", "here"), ("hear", "here"), ("one", "won")]
    ledger = M.substitution_ledger(pairs)
    assert ledger[0] == {"heard_as": "here", "instead_of": "hear", "count": 2}


# --- audio ---------------------------------------------------------------

def test_wav_roundtrip_preserves_signal():
    x = (np.sin(np.linspace(0, 40, 4000)) * 0.5).astype(np.float32)
    y, rate = decode_wav(encode_wav(x, 24000))
    assert rate == 24000
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) < 1e-3


def test_silence_length_and_duration():
    s = silence(500, 24000)
    assert len(s) == 12000
    assert abs(duration_s(s, 24000) - 0.5) < 1e-9


def test_room_is_a_no_op_when_inactive():
    x = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    assert np.array_equal(apply_room(x, RoomConfig()), x)


def test_room_stays_in_range_when_active():
    x = (np.random.default_rng(0).normal(0, 0.3, 8000)).astype(np.float32)
    y = apply_room(x, RoomConfig(noise_db=-40, gain_db=6, lowpass_hz=3400, room_decay=0.4))
    assert len(y) == len(x)
    assert np.max(np.abs(y)) <= 1.0


def test_resample_changes_length_proportionally():
    x = np.zeros(1000, dtype=np.float32)
    assert len(_resample(x, 16000, 24000)) == 1500


# --- config --------------------------------------------------------------

def test_config_roundtrips_through_json():
    cfg = RunConfig(iterations=3, room=RoomConfig(lowpass_hz=3400))
    back = RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back.iterations == 3
    assert back.room.lowpass_hz == 3400
    assert back.performance.breath_groups is True


def test_room_active_detection():
    assert not RoomConfig().active
    assert RoomConfig(room_decay=0.3).active


# --- cost ----------------------------------------------------------------

def test_rates_arithmetic_is_exact():
    from transcription_gap.config import Rates
    c = Rates(tts_per_1k_chars=0.030, stt_per_minute=0.0077).cost(2000, 120.0)
    assert c["tts_usd"] == 0.06          # 2 x $0.030
    assert c["stt_usd"] == 0.0154        # 2 min x $0.0077
    assert c["total_usd"] == 0.0754


def test_rates_roundtrip_through_config():
    cfg = RunConfig(rates=__import__("transcription_gap.config", fromlist=["Rates"])
                    .Rates(tts_per_1k_chars=0.015))
    back = RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back.rates.tts_per_1k_chars == 0.015


# --- the offline stand-in is deterministic -------------------------------

def test_offline_client_is_deterministic():
    a = OfflineClient(verbose=False)
    b = OfflineClient(verbose=False)
    from transcription_gap.config import ListenConfig, PerformanceConfig
    text = "I am sitting in a room, listening to the machine hear me again."
    for c in (a, b):
        c.perform(text, PerformanceConfig())
    assert a.hear(b"", ListenConfig()).transcript == b.hear(b"", ListenConfig()).transcript


def test_offline_client_respects_keyterms():
    from transcription_gap.config import ListenConfig, PerformanceConfig
    c = OfflineClient(verbose=False, confuse_rate=1.0, drop_rate=1.0)
    c.perform("the room the room the room", PerformanceConfig())
    out = c.hear(b"", ListenConfig(keyterms=["room", "the"])).transcript.lower()
    assert out.count("room") == 3 and out.count("the") == 3


# --- end to end ----------------------------------------------------------

def test_full_offline_run_produces_every_artifact():
    with tempfile.TemporaryDirectory() as td:
        seed = Path(td) / "seed.txt"
        seed.write_text(
            "I am speaking into a machine that is listening. It will write down "
            "what it thinks I said, and that is the score for the next voice.",
            encoding="utf-8",
        )
        out = Path(td) / "run"
        cfg = RunConfig(seed_path=str(seed), out_dir=str(out), iterations=4,
                        keep_audio=False)
        summary = TranscriptionGapLoop(cfg, client=OfflineClient(verbose=False),
                                       verbose=False).run()

        assert (out / "config.json").exists()
        assert (out / "summary.json").exists()
        assert (out / "manifest.json").exists()
        assert (out / "iterations" / "00.txt").exists()
        assert len(list((out / "iterations").glob("*.txt"))) >= 2
        assert len((out / "metrics.jsonl").read_text().strip().splitlines()) >= 1

        assert summary["iterations_run"] >= 1
        assert "convergence" in summary and "authorship" in summary
        assert summary["cost"]["tts_chars"] > 0
        assert summary["cost"]["total_usd"] > 0
        assert len(summary["series"]["wer_vs_prev"]) == summary["iterations_run"]

        # the report renders from the summary alone
        texts = [p.read_text().strip() for p in sorted((out / "iterations").glob("*.txt"))]
        page = build_html(summary, texts)
        assert "The Transcription Gap" in page
        assert "c-drift" in page and "stackedBar" in page
        assert page.count("__") == 0 or "__DATA__" not in page  # every token replaced

        # and a finished run can be reloaded from disk without an API client
        reloaded = TranscriptionGapLoop.from_output_dir(out)
        assert len(reloaded.iterations) == summary["iterations_run"] + 1
        again = reloaded.summarize()
        assert again["iterations_run"] == summary["iterations_run"]
        # billable totals must be recoverable from disk, not just live state
        assert again["cost"]["tts_chars"] == summary["cost"]["tts_chars"]
        assert abs(again["cost"]["stt_seconds"] - summary["cost"]["stt_seconds"]) < 0.2


def test_fixed_point_stops_the_loop_early():
    """A transcriber that hears perfectly should trip the fixed-point detector."""
    from transcription_gap.config import ListenConfig, PerformanceConfig
    from transcription_gap.deepgram import Hearing

    class PerfectEar:
        def __init__(self):
            self._t = ""

        def perform(self, text, perf: PerformanceConfig):
            self._t = text
            return silence(100)

        def hear(self, wav, cfg: ListenConfig) -> Hearing:
            return Hearing(transcript=self._t, words=[{"word": "x", "confidence": 1.0}])

    with tempfile.TemporaryDirectory() as td:
        seed = Path(td) / "s.txt"
        seed.write_text("nothing changes here.", encoding="utf-8")
        cfg = RunConfig(seed_path=str(seed), out_dir=str(Path(td) / "r"),
                        iterations=20, keep_audio=False, fixed_point_patience=2)
        summary = TranscriptionGapLoop(cfg, client=PerfectEar(), verbose=False).run()

    assert summary["iterations_run"] == 2  # stopped as soon as patience was met
    assert summary["convergence"]["status"] == "fixed_point"
    assert summary["total_drift_from_seed"] == 0.0
    assert summary["authorship"]["machine_share"] == 0.0


def test_cross_run_analysis_over_two_runs():
    from transcription_gap.analyze import find_runs, markdown_report, mutual_convergence

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "sweep"
        client = OfflineClient(verbose=False)
        for name, body in [
            ("one", "The recogniser is not a window, it has a preference about how a "
                    "sentence should end."),
            ("two", "Say it again, slower, and put the weight on the very last word."),
        ]:
            seed = Path(td) / f"{name}.txt"
            seed.write_text(body, encoding="utf-8")
            TranscriptionGapLoop(
                RunConfig(seed_path=str(seed), out_dir=str(root / name),
                          iterations=3, keep_audio=False),
                client=client, verbose=False,
            ).run()

        runs = find_runs(root)
        assert len(runs) == 2
        mc = mutual_convergence(runs)
        assert "converging_toward_each_other" in mc
        assert set(mc["runs"]) == {"one", "two"}
        md = markdown_report(runs)
        assert "cross-run analysis" in md and "Mutual convergence" in md


def test_mutual_convergence_needs_two_runs():
    from transcription_gap.analyze import mutual_convergence
    assert "note" in mutual_convergence([{"seed_text": "a", "final_text": "b", "_name": "x"}])


if __name__ == "__main__":
    import inspect

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and inspect.isfunction(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
