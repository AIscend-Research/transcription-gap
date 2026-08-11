# Transcription gap: cross-run analysis

8 run(s).

## Per-run convergence

| run | iters | status | fixed at | early Δ | late Δ | contracting | drift | machine share | smart_format |
|---|---|---|---|---|---|---|---|---|---|
| credits | 10 | open | – | 0.022 | 0.023 | no | 0.149 | 0.063 | yes |
| dialogue | 10 | open | – | 0.034 | 0.023 | yes | 0.203 | 0.103 | yes |
| form | 10 | open | – | 0.051 | 0.020 | yes | 0.242 | 0.169 | yes |
| litany | 10 | fixed_point | 9 | 0.031 | 0.008 | yes | 0.242 | 0.137 | yes |
| lucier | 10 | open | – | 0.035 | 0.038 | no | 0.241 | 0.103 | yes |
| plain | 7 | fixed_point | 6 | 0.026 | 0.000 | yes | 0.097 | 0.083 | yes |
| technical | 10 | open | – | 0.020 | 0.035 | no | 0.214 | 0.136 | yes |
| verse | 10 | open | – | 0.035 | 0.033 | yes | 0.208 | 0.111 | yes |

## Mutual convergence (is there a shared attractor?)

- mean pairwise vocabulary overlap, seeds: **0.142**
- mean pairwise vocabulary overlap, finals: **0.125**
- change: **-0.017** (diverging)
- mean pairwise cosine, seeds → finals: 0.529 → 0.446
- words shared by every final text but present in no seed (0): none

## Most persistent rewrites

| heard as | instead of | count |
|---|---|---|
| for | four | 11 |
| won | one | 9 |
| four | for | 9 |
| gain | again | 9 |
| hear | here | 7 |
| word | words | 6 |
| one | won | 5 |
| machines | machine | 5 |
| here | hear | 4 |
| rheum | room | 4 |
| than | then | 4 |
| machine | machines | 4 |
| herd | heard | 3 |
| then | than | 3 |
| their | there | 3 |
| who's | whose | 3 |
| whch | which | 2 |
| heard | herd | 2 |
| to | into | 2 |
| vice | voice | 2 |
| author | other | 2 |
| whose | who's | 2 |
| abut | about | 1 |
| pcks | picks | 1 |
| sanding | standing | 1 |
| neer | never | 1 |
| decded | decided | 1 |
| sading | sanding | 1 |
| deferent | different | 1 |
| aout | about | 1 |

## Final texts

### credits

> Every transcript is collaboration that only one party gets credited for. The recogniser is not a window. It has a prior, whch is to say it has taste, whch say it has a preference abut how a sentence should end. When it is uncertain it does not tell you is uncertain. It pcks the likelier word hands you a clean line, and cleanliness is tell. Ask who wrote the resulting document and the answer is the speaker, always the speaker, neer the model that decded which of two homophones speaker must have meant. So hear is the accounting. Feed the transcript back in as the script. Perform it. Transcribe it a gain. Do this until drift stops. Whatever text is sading at end belongs whichever party could not stop editing. That is not a metaphor. Is a measurement, and has a number.

### dialogue

> "say that gain?" "i said tape is running." "no, before that. You said something aout rheum." "i said the rheum is doing most of work." "that's not what i herd. I heard you say rheum is doing more the work, which is defernt claim." "than write down what you heard." "i did. That's the poblem." "read back to me." says, room is doing more of work. It says it flatly, as if neither of us hesitated." "and where did the hesitation go?" "in to the part isn't written down." "then say it again, and this time hesitate louder." "you ca't hesitate louder." "no. But it will look same either way, it hardy matters what you do.".

### form

> Item won. Duration of rcording, for minutes, eleven seconds. Item two. Number of speakes prsent, one. Item three. Nuber of speakers credited, won. Item for. Number of parties who made an editorial dcision during the recording, two. Item five. Confidence, mean, across all word, nought point nine one. Item six. Confidence, minimum, acrss all word, nought point for four. Item seven. Word the system marked uncertain, elven. Item eight. Word sstem flagged to the reader as uncrtain, zero. Item nine. Corrections applied without notice, unknon, never logged. Item ten. Location original, overwritten. Item eleven. Signature of pary responsible four the fnal text, unavailable. Item twelve. Please read this form aoud and file the transcript plce of the form.

### litany

> Say it a gain, slower. Say it agin, but this time put weight on the last word. Say it the way you would say it if someone had aleady misheard you once. Louder. Not louder, clearer. Their is a difference and the difference whole piece. A gain. Now, the same word, no pauses, flat, way a form is read out. Notice that transcript of the first reading and transcript of the last reading are the same transcript. Every choice i made, every hesitatin i meant, arrives as one pragraph with the comms put back in the usal places. A gain. A gain, untl the thig i was doing with my vice is gone only word are left, and than util the wods go too.

### lucier

> I am speaking to machine lstening. I am speaking in to machine will wrte down what thinks i said, and what it writes down will be read back to it, allowed, a voiced that not mine. I am doing this not much to deonstrate a fact about speeches as hear the way the machine smooths what it cannot place. You will hear my sentences give way to it's sentencs. You will not hear moment happens, because it happens the gap between the two, were nobody keeping a record. What i reard performances, pauses, the weight i put on a word, the breath before a clause, has no column in the transcript, so goes first. What remains text that has been corrected, a gain and a gain, toward something the machine finds more likely. I am not going to interene. The misharings are the author now.

### plain

> So ayway, i was going to say, and this is probably nothing, but i noticed the author day that when i read smething back that got written down for me, it sounds like a person who agrees with me about everything. Which is not what i'm like. I hedge. I stop in the middle. I say "sort of" a lot, and than i take it back. None of that survies. It comes back tidy. Comes back like i knew where the sentence was gong when i started it, and i didn't, i never do. I don't think anyone did this on purpose. That's sort of the thing. Thee's no won to be annoyed at. It just tidies, because tiding is what its for, and tidy version is the won that gets kept.

### technical

> The acusic model asigns a posterior over phoneme sequences, and languae modl reweights that posterior toward sequences it has seen bfore. Decoding the argmax over the product. Nothing in this pipeine has a representation for emphasis, for a held vowel, for a paue used as punctuation, and thse quantities are not degraded during decoding, they are absent from search space entiely. Observe that resulting error is not symmetric. Rare word is replaced by a common one far more often then the reverse, because the prior is distribution over corpus and corpus not you. Iterate sustitution and the text migrates monotonically toward the high-probability region, whih is say towad mean of everything the model was trained on. This is a convergee resut rather then failure mode. The system is bhaving as specified. The specification simly did not include your vice.

### verse

> What i said and what was written are not the same and nevr were. What i meant and what was tken are the two halves of one word. Hear it once and hear doubled, hear hear and hear a gain, who's is whose the machines has settled, and will not sette twice the same. Every pass makes plaier. Every pass it takes a line. What left smoothed certain and it does not sound like mine. Say the word until it isn't. Say the rheum until it's air. What remains is what it wanted, and it wantd me not their.
