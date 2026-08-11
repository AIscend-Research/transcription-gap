# Transcription gap: cross-run analysis

8 run(s).

## Per-run convergence

| run | iters | status | fixed at | early Δ | late Δ | contracting | drift | machine share | smart_format |
|---|---|---|---|---|---|---|---|---|---|
| credits | 12 | fixed_point | 7 | 0.008 | 0.002 | yes | 0.039 | 0.033 | yes |
| dialogue | 3 | fixed_point | 2 | – | – | – | 0.000 | 0.000 | yes |
| form | 25 | open | – | 0.012 | 0.001 | yes | 0.065 | 0.041 | yes |
| litany | 9 | fixed_point | 8 | 0.005 | 0.000 | yes | 0.023 | 0.016 | yes |
| lucier | 25 | fixed_point | 22 | 0.010 | 0.006 | yes | 0.142 | 0.070 | yes |
| plain | 5 | fixed_point | 2 | 0.000 | 0.000 | no | 0.000 | 0.000 | yes |
| technical | 9 | fixed_point | 3 | 0.006 | 0.004 | yes | 0.019 | 0.019 | yes |
| verse | 13 | fixed_point | 10 | 0.014 | 0.002 | yes | 0.104 | 0.039 | yes |

## Mutual convergence (is there a shared attractor?)

- mean pairwise vocabulary overlap, seeds: **0.142**
- mean pairwise vocabulary overlap, finals: **0.143**
- change: **0.001** (converging)
- mean pairwise cosine, seeds → finals: 0.529 → 0.517
- words shared by every final text but present in no seed (0): none

## Most persistent rewrites

| heard as | instead of | count |
|---|---|---|
| i'm | am | 3 |
| i'd | i | 3 |
| i | i'd | 3 |
| here | hear | 3 |
| planar | plainer | 3 |
| not | nought | 2 |
| 11 | eleven | 2 |
| naught | not | 2 |
| not | naught | 2 |
| and | i'm | 2 |
| to | into | 2 |
| i'm | and | 2 |
| towards | toward | 2 |
| weights | reweights | 2 |
| reweights | weights | 2 |
| plainer | planar | 2 |
| recognizer | recogniser | 1 |
| towel | tell | 1 |
| here's | is | 1 |
| mind | line | 1 |
| met | meant | 1 |
| 10 | ten | 1 |
| vocation | location | 1 |
| 12 | twelve | 1 |
| am | item | 1 |
| item | am | 1 |
| location | vocation | 1 |
| to | too | 1 |
| plearer | clearer | 1 |
| clearer | plearer | 1 |

## Final texts

### credits

> Every transcript is a collaboration that only one party gets credited for. The recognizer is not a window. It has a prior, which is to say, it has taste, which is to say, it has a preference about how a sentence should end. When it is uncertain, it does not tell you it is uncertain. It picks the likelier word and hands you a clean mind. And the cleanliness is the towel. Ask who wrote the resulting document. And the answer is the speaker. Always the speaker. Never the model that decided which of two homophones the speaker must have met. So, here's the accounting. Feed the transcript back in as the script. Perform it. Transcribe it again. Do this until the drift stops. Whatever text is standing at the end belongs to whichever party could not stop editing. That is not a metaphor. It is a measurement, and it has a number.

### dialogue

> Say that again? I said the tape is running. No. Before that, you said something about the room. I said the room is doing most of the work. That's not what I heard. I heard you say the room is doing more of the work, which is a different claim. Then write down what you heard. I did. That's the problem. Read it back to me. It says, the room is doing more of the work. It says it flatly, as if neither of us hesitated. And where did the hesitation go? Into the part that isn't written down. Then say it again. And this time, hesitate louder. You can't hesitate louder. No. But it will look the same either way. So it hardly matters what you do.

### form

> Item one, duration of the recording. Four minutes, eleven seconds. Item two, number of speakers present. One. Item three, number of speakers credited. One. Item four: Number of parties who made an editorial decision during the recording. Two. Item five: Mean across all words. Point nine one. Item six: Confidence. Minimum across all words. Not. Point four four. Item seven: Words the system marked uncertain. 11. Item eight. Words the system flagged to the reader as uncertain. Zero. Item nine. Corrections applied without notice. Never logged. Item 10. Location of the original. Overwritten item 11: Signature of the party responsible for the final text. Unavailable item 12: Please read this form aloud, and file the transcript in place of the form.

### litany

> Say it again slower. Say it again. But this time, cut the weight on the last word. Say it the way you would say it if someone had already misheard you once. Louder. Not louder. Clearer. There is a difference, and the difference is the whole piece. Again. Now, the same words. No pauses. Flat. The way a form is read out. Notice that the transcript of the first reading and the transcript of the last reading are the same transcript. Every choice I made, every hesitation I meant, arrives as one paragraph with the commas put back in the usual places. Again. Again. Until the thing I was doing with my voice is gone, but only the words are left. And then. Until the words go to.

### lucier

> I'm speaking it to a machine that is listening. I'm speaking it to a machine that will write down what it thinks I said. And what it writes down will be read back to it aloud in a voice that is not mine. I am doing this not so much to demonstrate a fact about speech as to hear the way machines smooth, but it cannot place. You will hear my sentences give way to sentences. You will not hear the moment it happens because it happens in the gap between the two where nobody's keeping a record. What I regard is the performance, the pauses. The way I put out a word, the breath before a clause, has no column in the transcript. So it goes first. What remains is a text that has been corrected again and again towards something the machine finds more likely. I'm not going to intervene. The mishearings are he offer now.

### plain

> So, anyway, I was going to say, and this is probably nothing. But I noticed the other day that when I read something back that got written down for me, it sounds like a person who agrees with me about everything, which is not what I'm like. I hedge. I stop in the middle. I say sort of a lot, and then I take it back. None of that survives. It comes back tidy. It comes back like I knew where the sentence was going when I started it, and I didn't. I never do. I don't think anyone did this on purpose. That's sort of the thing. There's no one to be annoyed at. It just tidies because tidying is what it's for, and the tidy version is the one that gets kept.

### technical

> The acoustic model assigns a posterior over phoneme sequences, and the language model reweights that posterior towards sequences it has seen before. Decoding is the ardnax over the product. Nothing in this pipeline has a representation for emphasis. For a health vowel, for a pause used as punctuation. And so, those quantities are not degraded during decoding. They are absent from the search space entirely. Observe that the resulting error is not symmetric. A rare word is replaced by a common one far more often than the reverse. Because the prior is a distribution over a corpus, and the corpus is not you, iterate the substitution, and the text migrates monotonically toward the high probability region, which is to say toward the mean of everything the model was trained on. This is a convergence result rather than a failure mode. The system is behaving as specified. The specification simply did not include your voice.

### verse

> What I said and what was written are not the same and never were. What I meant and what was taken are the two halves of one word. Here, it once. And here, it doubled. And here, and here again. Whose system machine has settled? And it will not settle twice the same. Every pass, it makes it planar. Every pass, it takes a line. What is left is smooth and certain, and it does not sound like mine. Say the word until it isn't. Save a room until it's air. What remains is what it wanted, and it wanted me not bare.
