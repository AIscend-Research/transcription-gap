# transcription-gap

> after Alvin Lucier, *I Am Sitting in a Room* (1969)

Speech recognition never transcribes a performance faithfully. It mishears, it
drops inflection, it silently corrects grammar. This project makes those errors
the author.

A spoken-word score is performed, transcribed, and **the transcript becomes the
score for the next performance**. Nobody edits anything. Round and round until
the text stops moving — then we ask what it stopped at, and whose text it is.

Lucier re-recorded a tape in a room until the room's resonant frequencies ate
his speech. Here the room is a speech-recognition model, and what it resonates
at is its own language prior.

## The claim

Two claims, one aesthetic and one empirical.

**Aesthetic:** transcription bias is an author nobody credits. Every transcript
is a collaboration billed to one party. When a recogniser is uncertain it does
not say so — it picks the likelier word and hands back a clean line. Iterate
that and the mishearings accumulate into a piece with a co-writer who is never
in the credits.

**Empirical:** published work shows generative *image* loops collapse into a
house style — feed a model's output back as its input and it converges to an
attractor. This asks whether the speech domain does the same thing, and
supplies the measurement: drift curves per iteration, fixed-point and cycle
detection, and a cross-run test for whether *different* scores through the
*same* transcriber end up closer to each other than they started.

Runs on a laptop, one API, under an hour, no participants.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your Deepgram key in
```

Deepgram does both halves of the loop — `/v1/speak` (Aura) performs, `/v1/listen`
(Nova) mishears — so one key is the whole setup.

## Run it

```bash
# what will this cost? (spends nothing)
.venv/bin/python -m transcription_gap estimate -n 25

# one loop
.venv/bin/python -m transcription_gap run --seed seeds/lucier.txt -n 25

# no key? exercise the whole pipeline against a simulated transcriber
.venv/bin/python -m transcription_gap run --offline -n 10

# the actual experiment: eight scores, one transcriber, do they meet
.venv/bin/python -m transcription_gap sweep --seeds "seeds/*.txt" -n 25
```

Each run writes to `outputs/<name>/`:

| file | what it is |
|---|---|
| `iterations/00.txt` … | the score, then every take |
| `audio/01.wav` … | each performance as it was heard |
| `metrics.jsonl` | one row of measurements per iteration |
| `summary.json` | drift series, convergence verdict, authorship split, cost |
| `report.html` | self-contained report — open it in a browser |
| `manifest.json` | index of the above |

`sweep` additionally writes `analysis.md` with the cross-run attractor test.

## What gets measured

**Drift.** Word and character error rate of each take against the one before it
(how fast it is still moving) and against the original score (how far it has
travelled). Plus vocabulary overlap, bag-of-words cosine, and the transcriber's
own mean word confidence — confidence *rising* while overlap falls is the loop
becoming fluent in its own text.

**Convergence.** `fixed_point` (consecutive takes identical — the attractor is a
single text the machine will reproduce forever), `cycle` (orbiting between k
states), or `open` (still drifting when we ran out of iterations). Also whether
step-to-step deltas are contracting, and a geometric extrapolation of how many
more iterations a fixed point would take.

**Authorship.** The number the piece is actually about. Of the final text's
tokens, what share are words the human wrote, and what share entered through a
mishearing and then survived being re-performed and re-heard until they stuck?
The report shows each surviving machine word, when it entered, and how long it
lasted — plus the score's own words that were never heard again.

**The discarded half.** Token count, type-token ratio and punctuation density
per iteration. Punctuation and line structure are a transcript's only channels
for phrasing; watching them collapse *is* the performance being deleted.

**Mutual convergence** (across runs). Mean pairwise similarity between the runs'
seeds versus between their finals. If the finals are closer, the loop is pulling
everything toward a common destination rather than degrading each text
independently — and the words every final shares that no seed contained are that
attractor's own vocabulary.

## The seeds

Eight scores across deliberately different registers, so "it converges to an
attractor" can be a claim about the transcriber rather than about one text:

| seed | register | what it stresses |
|---|---|---|
| `lucier.txt` | procedural self-description | the piece stating its own method |
| `credits.txt` | argumentative prose | abstract nouns, no concrete anchors |
| `litany.txt` | performance instructions | heavy phrasing, repetition, dashes |
| `dialogue.txt` | two-speaker exchange | quotation marks, interruption, short turns |
| `form.txt` | a form read aloud | numerals, labels, list structure |
| `verse.txt` | lineated verse | line breaks, rhyme, deliberate homophones |
| `technical.txt` | clinical exposition | jargon and low-frequency vocabulary |
| `plain.txt` | unstructured speech | hedges, self-interruption, "sort of" |

They are original texts in Lucier's lineage, not his text. Add your own — a
seed is just a `.txt` file, and `sweep` picks up everything matching the glob.
More seeds is the cheapest way to strengthen the result.

## Cost

Both billable quantities scale with text length: characters into Aura, minutes
out of it and into Nova. `estimate` prices a plan up front, and every finished
run records what it actually spent in `summary.json`.

Eight seeds × 25 iterations ≈ **$6** total — about $0.75 per run, roughly
2¢ per iteration. At the rates published on 2026-07-31 (Aura-2 $0.030/1k chars,
Nova-3 monolingual $0.0077/min, pay-as-you-go). Override with `--rate-tts` /
`--rate-stt` if they've moved, or if you're on Growth tier where they're lower.

Runs that reach a fixed point stop early and cost less than the estimate.

## Where the performance lives

`--breath-groups` (on by default) is the load-bearing detail. The score is split
at its punctuation, each phrase is synthesised separately, and real silence is
inserted between them — so the phrasing exists in the waveform, where the
transcript has no column for it. That information is what every iteration
throws away first, and the punctuation-density curve is watching it go.

The other side of the same point is `smart_format`. It is the silent editor:
repunctuating, recapitalising, "correcting", never asked and never logged. Run
the same score with `--no-smart-format` and diff the two attractors — that
difference is one feature's authorship, isolated.

## Knobs

```
-n, --iterations      how many times round (default 12)
--voice               Deepgram Aura voice (default aura-2-thalia-en)
--listen-model        Deepgram STT model (default nova-3)
--no-smart-format     disable the silent editor
--no-breath-groups    synthesise flat, no performed phrasing
--keyterm WORD        bias the transcriber toward a word (repeatable)
--seed-audio FILE.wav start from a human reading instead of Aura
--run-to-end          keep going after a fixed point
--offline             simulated transcriber, no API, no key
--lowpass HZ          e.g. 3400 for a telephone channel
--room-decay 0..0.95  crude standing waves — Lucier's actual room
--noise-db / --gain-db
```

The room defaults to *off*: with no channel degradation the only thing eroding
the text is the transcriber's own bias, which is the point. Turn the room on to
put Lucier's physical mechanism back in and see which author wins.

### As a library

```python
from transcription_gap import TranscriptionGapLoop, RunConfig, write_report

cfg = RunConfig(seed_path="seeds/lucier.txt", out_dir="outputs/x", iterations=12)
summary = TranscriptionGapLoop(cfg).run()
print(summary["authorship"]["machine_share"])
write_report("outputs/x")
```

## Tests

```bash
.venv/bin/python tests/test_transcription_gap.py   # or: python -m pytest -q
```

38 tests, no network. The artistic claim rests entirely on the numbers being
right, so the edit distance, convergence detector and authorship split are all
checked against hand-computed cases.

## Notes

- `--offline` is a scripted caricature of a recogniser (homophone confusion,
  dropped function words, house-style reformatting), not a speech model. It is
  for testing the pipeline. Only a real Deepgram run measures a real
  transcriber; the report header records which one produced it.
- The seeds are original texts written in Lucier's lineage, not his text.
