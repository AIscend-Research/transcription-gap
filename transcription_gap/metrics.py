"""Measurement: how far the text moved, and whether it stopped moving.

Two families of number here.

*Drift* metrics compare consecutive iterations (how fast is it still changing)
and each iteration against the original seed (how far has it travelled).

*Attractor* metrics ask the question the piece is actually about: when the
motion stops, whose text are we looking at. ``authorship`` splits the final
vocabulary into words the human wrote and words the transcriber introduced and
then kept.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9']+")
_PUNCT = set(",.;:!?—–-\"'()[]…")


# --- normalisation --------------------------------------------------------


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    """Lexical content only: case, punctuation and spacing stripped out.

    Everything this function throws away is exactly what ``surface_features``
    measures separately -- the performance lives in the discarded part.
    """
    return _WORD_RE.findall(normalize(text))


def chars(text: str) -> list[str]:
    return list(normalize(text))


# --- edit distance --------------------------------------------------------


@dataclass
class Alignment:
    distance: int
    substitutions: list[tuple[str, str]]
    deletions: list[str]
    insertions: list[str]
    hits: int

    @property
    def error_rate_denominator(self) -> int:
        return self.hits + len(self.substitutions) + len(self.deletions)

    @property
    def rate(self) -> float:
        d = self.error_rate_denominator
        return self.distance / d if d else 0.0


def align(ref: list[str], hyp: list[str]) -> Alignment:
    """Levenshtein with backtrace, so we keep the individual mishearings.

    The substitution list is the interesting output: it is a log of what the
    machine decided each word really was.
    """
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return Alignment(0, [], [], [], 0)

    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        prev, cur = d[i - 1], d[i]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)

    subs: list[tuple[str, str]] = []
    dels: list[str] = []
    ins: list[str] = []
    hits = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                if cost:
                    subs.append((ref[i - 1], hyp[j - 1]))
                else:
                    hits += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels.append(ref[i - 1])
            i -= 1
            continue
        ins.append(hyp[j - 1])
        j -= 1

    subs.reverse(); dels.reverse(); ins.reverse()
    return Alignment(int(d[n][m]), subs, dels, ins, hits)


def wer(ref: str, hyp: str) -> float:
    return align(words(ref), words(hyp)).rate


def cer(ref: str, hyp: str) -> float:
    return align(chars(ref), chars(hyp)).rate


# --- set / distribution similarity ---------------------------------------


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def cosine(a: str, b: str) -> float:
    ca, cb = Counter(words(a)), Counter(words(b))
    if not ca or not cb:
        return 0.0
    keys = set(ca) | set(cb)
    va = np.array([ca[k] for k in keys], dtype=float)
    vb = np.array([cb[k] for k in keys], dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denom) if denom else 0.0


# --- the discarded half ---------------------------------------------------


def surface_features(text: str) -> dict:
    """Everything ``words()`` deletes -- i.e. the performance markers.

    Punctuation, capitalisation and line structure are the transcript's only
    channels for inflection, and they are precisely what smart-formatting
    rewrites to its own house style.
    """
    toks = words(text)
    n = max(1, len(toks))
    letters = [c for c in text if c.isalpha()]
    return {
        "tokens": len(toks),
        "types": len(set(toks)),
        "type_token_ratio": len(set(toks)) / n,
        "chars": len(text),
        "mean_word_len": float(np.mean([len(t) for t in toks])) if toks else 0.0,
        "punct_per_100_tokens": 100.0 * sum(1 for c in text if c in _PUNCT) / n,
        "sentences": max(1, len(re.findall(r"[.!?]+", text))),
        "mean_sentence_tokens": len(toks) / max(1, len(re.findall(r"[.!?]+", text))),
        "capitalized_ratio": (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0,
        "lines": len([ln for ln in text.splitlines() if ln.strip()]),
    }


# --- per-iteration record -------------------------------------------------


@dataclass
class StepMetrics:
    iteration: int
    wer_vs_prev: float
    cer_vs_prev: float
    wer_vs_seed: float
    cer_vs_seed: float
    jaccard_vs_prev: float
    jaccard_vs_seed: float
    cosine_vs_seed: float
    substitutions: int
    deletions: int
    insertions: int
    new_words: int          # words never seen in any earlier iteration
    surviving_seed_words: int
    seed_vocab_retained: float
    mean_confidence: float
    min_confidence: float
    audio_seconds: float
    surface: dict

    def to_dict(self) -> dict:
        return asdict(self)


def step_metrics(iteration: int, seed: str, prev: str, cur: str, *,
                 seen_vocab: set[str], mean_conf: float, min_conf: float,
                 audio_seconds: float) -> StepMetrics:
    a_prev = align(words(prev), words(cur))
    a_seed = align(words(seed), words(cur))
    seed_vocab = set(words(seed))
    cur_vocab = set(words(cur))
    return StepMetrics(
        iteration=iteration,
        wer_vs_prev=a_prev.rate,
        cer_vs_prev=cer(prev, cur),
        wer_vs_seed=a_seed.rate,
        cer_vs_seed=cer(seed, cur),
        jaccard_vs_prev=jaccard(prev, cur),
        jaccard_vs_seed=jaccard(seed, cur),
        cosine_vs_seed=cosine(seed, cur),
        substitutions=len(a_prev.substitutions),
        deletions=len(a_prev.deletions),
        insertions=len(a_prev.insertions),
        new_words=len(cur_vocab - seen_vocab),
        surviving_seed_words=len(cur_vocab & seed_vocab),
        seed_vocab_retained=(len(cur_vocab & seed_vocab) / len(seed_vocab)) if seed_vocab else 0.0,
        mean_confidence=mean_conf,
        min_confidence=min_conf,
        audio_seconds=audio_seconds,
        surface=surface_features(cur),
    )


# --- convergence ----------------------------------------------------------


def detect_convergence(texts: list[str], tol: float = 1e-9) -> dict:
    """Did the loop land somewhere, and where.

    Three outcomes, borrowed from the image-generation literature:
      * ``fixed_point``  -- consecutive iterations identical; the attractor is
        a single text the transcriber will reproduce forever.
      * ``cycle``        -- the loop is orbiting between k states.
      * ``open``         -- still drifting when we ran out of iterations.
    """
    norm = [normalize(t) for t in texts]
    result: dict = {"status": "open", "fixed_point_at": None, "cycle": None}

    for i in range(1, len(norm)):
        if norm[i] and norm[i] == norm[i - 1]:
            result["status"] = "fixed_point"
            result["fixed_point_at"] = i
            break

    if result["status"] == "open":
        seen: dict[str, int] = {}
        for i, t in enumerate(norm):
            if t in seen:
                result["status"] = "cycle"
                result["cycle"] = {"start": seen[t], "end": i, "period": i - seen[t]}
                break
            seen[t] = i

    # Rate of change over the last third: is it decaying towards zero?
    deltas = [wer(norm[i - 1], norm[i]) for i in range(1, len(norm))]
    result["deltas"] = deltas
    if len(deltas) >= 4:
        k = max(2, len(deltas) // 3)
        head, tail = float(np.mean(deltas[:k])), float(np.mean(deltas[-k:]))
        result["early_mean_delta"] = head
        result["late_mean_delta"] = tail
        result["decay_ratio"] = (tail / head) if head > tol else float("nan")
        result["contracting"] = bool(tail < head)
        # Geometric extrapolation of when the deltas would reach ~0.
        if head > tol and 0 < tail < head:
            per_step = (tail / head) ** (1.0 / max(1, len(deltas) - k))
            if 0 < per_step < 1:
                import math
                result["projected_steps_to_fixed_point"] = float(
                    math.log(0.001 / max(tail, tol)) / math.log(per_step)
                )
    return result


def authorship(seed: str, final: str, all_texts: list[str]) -> dict:
    """Whose voice did it settle into.

    Counted over the final text's tokens: how many are words the human wrote,
    and how many are words that entered somewhere in the loop -- introduced by
    a mishearing -- and were then re-performed and re-heard until they stuck.
    The second number is the credit nobody gives the transcriber.
    """
    seed_vocab = set(words(seed))
    final_toks = words(final)
    if not final_toks:
        return {"machine_share": float("nan"), "human_share": float("nan")}

    from_machine = [t for t in final_toks if t not in seed_vocab]
    machine_vocab = sorted(set(from_machine))

    # When did each surviving machine word first appear, and did it persist?
    first_seen: dict[str, int] = {}
    for i, t in enumerate(all_texts):
        for w in set(words(t)):
            first_seen.setdefault(w, i)

    survivors = []
    for w in machine_vocab:
        idx = first_seen.get(w, len(all_texts) - 1)
        survived = len(all_texts) - 1 - idx
        survivors.append({"word": w, "entered_at": idx, "survived_iterations": survived})
    survivors.sort(key=lambda s: (-s["survived_iterations"], s["word"]))

    lost = sorted(seed_vocab - set(final_toks))
    return {
        "machine_share": len(from_machine) / len(final_toks),
        "human_share": 1.0 - len(from_machine) / len(final_toks),
        "machine_vocab_size": len(machine_vocab),
        "seed_words_lost": len(lost),
        "seed_words_lost_list": lost[:200],
        "machine_words": survivors[:200],
    }


def substitution_ledger(pairs: list[tuple[str, str]]) -> list[dict]:
    """Most frequent word -> word rewrites across the whole run."""
    c = Counter(pairs)
    return [
        {"heard_as": hyp, "instead_of": ref, "count": n}
        for (ref, hyp), n in c.most_common(200)
    ]
