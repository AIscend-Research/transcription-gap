# transcription-gap

> after Alvin Lucier, *I Am Sitting in a Room* (1969)

Speech recognition never transcribes a performance faithfully. It mishears, it
drops inflection, it silently corrects grammar. This project makes those errors
the author.

A spoken-word score is performed, transcribed, and **the transcript becomes the
score for the next performance**. Nobody edits anything. Round and round until
the text stops moving. Then we ask what it stopped at, and whose text it is.

Lucier re-recorded a tape in a room until the room's resonant frequencies ate
his speech. Here the room is a speech-recognition model, and what it resonates
at is its own language prior.

## Result

Measured on 2026-08-11: 8 scores, one transcriber (Deepgram Aura 2 speaking,
Nova 3 listening), 101 iterations, $3.02.

**Seven of eight scores reach a fixed point, most within ten iterations, and the
fixed point is close to the score they started from.** Final drift is 0.000 to
0.142 WER. Between 93% and 100% of every final text is still the words the human
wrote. The transcriber is close to an identity function on its own output.

**What it does write, it writes permanently, with no signal that it wrote
anything.** Mean confidence sits at 0.98 from the first iteration to the last
and never dips at a substitution. The loop is not a slow erosion. It is a small
number of irreversible decisions taken silently and early.

The clearest single instance is the last line of `lucier.txt`:

```
score:  The mishearings are the author now.
final:  The mishearings are he offer now.
```

`author` is one of eight words in that score that the transcriber never heard
again.

### What did not hold

Three predictions in earlier drafts of this project are not supported by the
data, and the code no longer claims them.

**There is no shared attractor.** This was the headline hypothesis, that
different scores through the same transcriber would converge toward a common
house style the way generative image loops do. Mean pairwise vocabulary overlap
moved from 0.1420 to 0.1439, a change of +0.0019, against a per-pair standard
deviation of 0.0094. Only 13 of 28 seed pairs ended closer together than they
started, which is worse than chance. Mean pairwise cosine moved the wrong way,
0.529 down to 0.517. No word is shared by every final text and absent from every
seed. Each score converges to its own fixed point, not to a common one. Reported
as a negative result.

**Confidence does not rise as the text drifts.** It is flat at roughly 0.98
throughout. The transcriber was maximally certain at iteration 1 and had nowhere
to go. The interesting version of this is not a crossover, it is that certainty
is constant while correctness is not.

**The performance channel does not collapse.** Punctuation per 100 tokens rises
(10.6 to 12.7 on `lucier`), type-token ratio rises slightly, and token counts
barely move (154 to 153, 128 to 128, 105 to 103). `smart_format` reformats
aggressively at every step, but it does not strip structure away over time.

## The claim

**Aesthetic:** transcription bias is an author nobody credits. Every transcript
is a collaboration billed to one party. When a recogniser is uncertain it does
not say so. It picks the likelier word and hands back a clean line. The
measurement above puts a number on that co-writer: small (0 to 7%), silent
(0.98 confidence at every substitution), and permanent (surviving substitutions
persist to the fixed point).

**Empirical:** published work reports that generative image loops collapse into
a house style when a model's output is fed back as its input. This asks whether
the speech domain does the same thing and supplies the measurement. The answer
here is no, at this scale and with this transcriber.

Runs on a laptop, one API, under three hours, no participants.

## Numbers for the paper

### Per-run convergence

| run | iters | status | fixed at | drift vs score | machine share |
|---|---|---|---|---|---|
| credits | 12 | fixed_point | 7 | 0.039 | 3.3% |
| dialogue | 3 | fixed_point | 2 | 0.000 | 0.0% |
| form | 25 | open | (none) | 0.065 | 4.1% |
| litany | 9 | fixed_point | 8 | 0.023 | 1.6% |
| lucier | 25 | fixed_point | 22 | 0.142 | 7.0% |
| plain | 5 | fixed_point | 2 | 0.000 | 0.0% |
| technical | 9 | fixed_point | 3 | 0.019 | 1.9% |
| verse | 13 | fixed_point | 10 | 0.104 | 3.9% |

Only `form` was still moving at iteration 25.

### Cross-run convergence

| quantity | seeds | finals | change |
|---|---|---|---|
| mean pairwise vocabulary overlap | 0.1420 | 0.1439 | +0.0019 |
| mean pairwise cosine | 0.529 | 0.517 | -0.012 |
| pairs that ended closer | | 13 of 28 | |
| per-pair change, standard deviation | | 0.0094 | |
| words in every final, in no seed | | 0 | |

The change is one fifth of its own dispersion. Do not report it as convergence.

### Substitutions that persisted

Homophone and near-homophone swaps dominate, followed by contraction and
numeral reformatting.

| heard as | instead of | count |
|---|---|---|
| i'm | am | 3 |
| i'd | i | 3 |
| here | hear | 3 |
| planar | plainer | 3 |
| not | nought | 2 |
| 11 | eleven | 2 |
| to | into | 2 |
| towards | toward | 2 |
| weights | reweights | 2 |
| towel | tell | 1 |
| mind | line | 1 |
| met | meant | 1 |
| vocation | location | 1 |
| clearer | plearer | 1 |

On `lucier`, the words that entered through mishearing and survived to the fixed
point were `i'm`, `offer`, `towards`, `nobody's`, `out`, `but`, `machines`,
`smooth`. The score words never heard again were `author`, `into`, `its`,
`nobody`, `on`, `smooths`, `toward`, `weight`.

### Figures

`figures.py` writes vector PDFs to `figures/`. Per-iteration figures are drawn
from `lucier`, the longest run with the largest drift.

| file | shows |
|---|---|
| `drift.pdf` | WER against the previous take and against the score, with the fixed point marked |
| `confidence.pdf` | mean confidence flat at 0.98 while vocabulary overlap falls to 0.83 |
| `discarded.pdf` | punctuation per 100 tokens and type-token ratio per iteration |
| `authorship.pdf` | machine share of the final text, per seed |
| `convergence.pdf` | all eight seeds, WER against their own score |

```bash
.venv/bin/python figures.py                  # from outputs/sweep
.venv/bin/python figures.py --run outputs/run
```

The script refuses to plot runs produced by the offline simulator unless given
`--allow-offline`, so simulated numbers cannot reach a figure by accident.

### Excerpt worth quoting in full

`lucier.txt`, score against fixed point at iteration 22. The drift is legible
without any metric: possessives lost, clauses fused, and a negation-adjacent
substitution (`smooths what it cannot place` to `smooth, but it cannot place`)
that changes the sentence's meaning while reading as fluent English.

```
score:  I am speaking into a machine that is listening.
final:  I'm speaking it to a machine that is listening.

score:  ...to hear the way the machine smooths what it cannot place.
final:  ...to hear the way machines smooth, but it cannot place.

score:  You will hear my sentences give way to its sentences.
final:  You will hear my sentences give way to sentences.

score:  The mishearings are the author now.
final:  The mishearings are he offer now.
```

### Still needed before submission

- A citation for the image-loop convergence result. The empirical framing
  positions this as the speech-domain counterpart, and that prior work must be
  named rather than asserted.
- Author block and biographies. The track is single-blind, so names are
  included.
- The NeurIPS Creative AI paper template.

### Limitations to state

- One transcriber, one voice, one language. The negative convergence result is
  about Nova 3 with Aura 2, not about speech recognition generally.
- Eight scores, 28 pairs. Underpowered for a small effect, which is exactly why
  +0.0019 should not be read as a positive.
- Synthetic speech throughout. A human reader would introduce disfluency and
  prosody that Aura does not produce. `--seed-audio` accepts a human reading.
- TTS and STT are not deterministic. An earlier partial sweep put `credits` at a
  fixed point by iteration 4; this one reached it at 7. Convergence iteration is
  not stable across runs, though convergence itself was reached in both.
- `form` did not converge in 25 iterations, so its final state is a truncation,
  not a fixed point.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your Deepgram key in
```

Deepgram does both halves of the loop. `/v1/speak` (Aura) performs and
`/v1/listen` (Nova) mishears, so one key is the whole setup. Figures need
`matplotlib`, which is not in `requirements.txt`:

```bash
.venv/bin/pip install matplotlib
```

Keep the key out of git. `.env` is in `.gitignore`, and if it was ever committed
the key must be rotated, since removing it from history does not unpublish it.

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

The full sweep took about 100 minutes and $3.02. Iterations run at roughly 0.5
to 0.85 times the duration of the audio they transcribe.

Each run writes to `outputs/<name>/`:

| file | what it is |
|---|---|
| `iterations/00.txt` ... | the score, then every take |
| `audio/01.wav` ... | each performance as it was heard |
| `metrics.jsonl` | one row of measurements per iteration |
| `summary.json` | drift series, convergence verdict, authorship split, cost |
| `report.html` | self-contained report, open it in a browser |
| `manifest.json` | index of the above |

`sweep` additionally writes `analysis.md` with the cross-run attractor test.

### Checking a sweep is real

The offline simulator returns in milliseconds, a real API iteration takes
roughly half the duration of its audio. A ratio near zero means the run never
reached Deepgram.

```bash
.venv/bin/python -c "
import json,glob
for p in sorted(glob.glob('outputs/sweep/*/summary.json')):
    d=json.load(open(p))
    print(f\"{p.split('/')[2]:10} {d['iterations_run']:>3} iters  ratio {d['elapsed_seconds']/d['cost']['stt_seconds']:.2f}  {d['convergence']['status']}\")
"
```

## What gets measured

**Drift.** Word and character error rate of each take against the one before it
(how fast it is still moving) and against the original score (how far it has
travelled). Plus vocabulary overlap, bag-of-words cosine, and the transcriber's
own mean word confidence.

**Convergence.** `fixed_point` (consecutive takes identical, so the attractor is
a single text the machine will reproduce forever), `cycle` (orbiting between k
states), or `open` (still drifting when we ran out of iterations). Also whether
step-to-step deltas are contracting, and a geometric extrapolation of how many
more iterations a fixed point would take.

**Authorship.** The number the piece is actually about. Of the final text's
tokens, what share are words the human wrote, and what share entered through a
mishearing and then survived being re-performed and re-heard until they stuck?
The report shows each surviving machine word, when it entered, and how long it
lasted, plus the score's own words that were never heard again.

**Structure.** Token count, type-token ratio and punctuation density per
iteration. Punctuation and line structure are a transcript's only channels for
phrasing.

**Mutual convergence** (across runs). Mean pairwise similarity between the runs'
seeds versus between their finals, reported with its dispersion so a change
smaller than the noise is not mistaken for an effect.

## The seeds

Eight scores across deliberately different registers, so a claim about
convergence can be a claim about the transcriber rather than about one text:

| seed | register | what it stresses |
|---|---|---|
| `lucier.txt` | procedural self-description | the piece stating its own method |
| `credits.txt` | argumentative prose | abstract nouns, no concrete anchors |
| `litany.txt` | performance instructions | heavy phrasing, repetition, imperatives |
| `dialogue.txt` | two-speaker exchange | quotation marks, interruption, short turns |
| `form.txt` | a form read aloud | numerals, labels, list structure |
| `verse.txt` | lineated verse | line breaks, rhyme, deliberate homophones |
| `technical.txt` | clinical exposition | jargon and low-frequency vocabulary |
| `plain.txt` | unstructured speech | hedges, self-interruption, "sort of" |

They are original texts in Lucier's lineage rather than his text. Add your own:
a seed is just a `.txt` file, and `sweep` picks up everything matching the glob.
More seeds is the cheapest way to strengthen the result, and at 28 pairs the
cross-run test is currently underpowered.

## Where the performance lives

`--breath-groups` (on by default) is the load-bearing detail. The score is split
at its punctuation, each phrase is synthesised separately, and real silence is
inserted between them, so the phrasing exists in the waveform, where the
transcript has no column for it.

The other side of the same point is `smart_format`. It is the silent editor:
repunctuating, recapitalising, "correcting", never asked and never logged. Run
the same score with `--no-smart-format` and diff the two attractors. That
difference is one feature's authorship, isolated, and it is the most obvious
unrun experiment in the repo.

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
--room-decay 0..0.95  crude standing waves, Lucier's actual room
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
  dropped function words, house-style reformatting) rather than a speech model.
  It is for testing the pipeline. It is also badly calibrated: on the same
  seeds it produced roughly triple the real drift (0.21 against 0.07) and
  triple the real machine share (12% against 4%), and it emits non-words such
  as "whch" and "pcks" that a real recogniser cannot produce, since a recogniser
  only outputs words in its vocabulary. Only a Deepgram run measures a real
  transcriber, and the report header records which one produced it.
- The seeds are original texts written in Lucier's lineage rather than his text.
