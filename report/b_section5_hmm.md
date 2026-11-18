# Section 5 — HMM and Viterbi route reconstruction (DRAFT, Person B: Tejasvi)

Sources (every number below is in these files): `results/b_hmm_comparison.csv`, `b_hmm_pertrip.csv`,
`b_hmm_best.json`, `b_hmm_tuning.csv`, `b_hmm_tuning_grid1.csv`, `b_hmm_tuning_grid2.csv`,
`b_hmm_worse_obs.csv`, `results/logs/b_hmm_tune.log`, `b_hmm_compare.log`; plots `b_hmm_pertrip.png`,
`b_hmm_gain.png`, `b_hmm_tuning.png`. Code: `src/hmm/viterbi.py`, `tune_decoder.py`, `compare_hmm.py`.

Classifier used throughout: the final exported model (Bayesian-optimised LightGBM, `artifacts/model.joblib`).
Metrics use the same definitions as the model table (`src/common/metrics.py`).

## 1. What are the hidden states?
For GPS observation *i*, the hidden state is the road segment (arc) the vehicle is on. The state space at
step *i* is the candidate set of that observation (K ≤ 10 nearest arcs within R = 50 m), so every state is
an `arc_id` and the candidate-to-road-segment correspondence is preserved. Observations are decoded in
time order; the decoder returns one arc per observation.

## 2. How are emission probabilities calculated?
The classifier already combines distance, heading, speed and the neighbouring observations (19 features),
so we reuse its output instead of building a separate distance model:

    emission(arc | observation) = alpha · log( p )

where *p* is the classifier probability that the arc is the true one (clipped to ≥ 1e-6) and `alpha` is a
tuned weight on the classifier evidence.

## 3. How are transition probabilities calculated?
Transitions use the directed road network between observation *i* and *i+1*. We work with costs and
score a transition as −cost (a relative score; Viterbi only compares paths):

| From arc a to arc b | Cost |
|---|---|
| b = a (still on the same road) | 0 |
| b reachable from a after d further directed arcs, d = 0 … 5 | `beta · (1 + d)` |
| b not reachable within that limit | `invalid` (finite, so decoding never fails) |
| b is the reverse arc of a (U-turn), added on top | `uturn` |

**Viterbi recurrence.** With δ₁(s) = emission₁(s) and, for i ≥ 2,

    δᵢ(b) = emissionᵢ(b) + max over a of [ δᵢ₋₁(a) − cost(a → b) ],    ψᵢ(b) = the maximising a,

the route is read by backtracking from the best final state, following ψ. With at most 10 states per
step the cost is O(n · K²); decoding all 19,160 student-test observations took about 1.2 s in total
(`b_hmm_compare.log`).

## 4. How are missing candidates and large GPS gaps handled?
- **No candidate within 50 m:** that observation is dropped from the chain and gets no prediction (counted
  as wrong in Top-1). In student_test every observation had at least one candidate; the candidate *miss
  rate* (true arc not among the candidates) is 0.26 % on student_test and 1.66 % on train
  (`results/a_dataset_summary.json`).
- **GPS gap:** if the time between two decoded observations is more than 30 s, or observation indices are
  missing in between, the connectivity rule is switched off for that step (all transitions cost 0) and
  the chain effectively restarts. A vehicle can legitimately travel far during a gap, so adjacency would be
  unreliable. In the raw data 28 of 80 train trips and 2 of 10 student-test trips have a gap > 30 s
  (`results/a_eda_summary.json`).

## 5. What happens when consecutive observations suggest disconnected road segments?
The jump is allowed but charged `invalid`, so it is taken only when the evidence for it outweighs the
penalty. With the tuned values (`invalid` = 48, `alpha` = 0.1) the penalty is far larger than any
difference in classifier evidence, so in practice the decoder only jumps between unconnected arcs
across a gap. (Reasoning from the cost structure; not separately measured.)

## 6. Did HMM decoding improve the final route?
### Decoder parameters (optimised on training data only)
- Objective: observation-level Top-1 of the decoded route, pooled over the 78 usable training trips.
- Probabilities: the **out-of-fold** predictions of the tuned LightGBM (each trip scored by a model that never
  saw it), so the decoder is not tuned on over-confident in-sample probabilities. These are filtered
  sequences: the two anomalous trips and observations > 50 m from their reference arc were removed, and a
  removed observation restarts the chain like a gap.
- The 10 student-test trips were **not** used for tuning.
- Three grid searches, 216 settings in total (`b_hmm_tuning*.csv`, `b_hmm_tuning.png`):

| Grid | Settings | Best Top-1 | Note |
|---|---|---|---|
| 1 | 54 | 0.9327 | best point on the grid edge for all four parameters, so widened |
| 2 | 81 | 0.9529 | still on the edge for three parameters, so widened once more |
| 3 | 81 | **0.9557** | interior to the full searched range; the top five settings are within 0.0002 |

Final: `alpha = 0.1, beta = 3, invalid = 48, uturn = 16` (two settings tied on Top-1, we kept the less
extreme one). The classifier-only Top-1 on the same training observations is 0.8874. Some settings are
worse than no HMM (lowest trial 0.8776), so the parameters matter.

### Classifier-only vs classifier + HMM

| | Train OOF (tuned on, 78 trips) | **student_test (held out, 10 trips)** |
|---|---|---|
| Top-1, classifier only | 0.8874 | 0.8873 |
| Top-1, classifier + HMM | 0.9557 (+0.0683) | **0.9555 (+0.0683)** |
| Route continuity, classifier only → HMM | n/a | 0.9825 → **0.9998** |
| Route precision (arcs on the reference route) | n/a | 0.8851 → 0.9848 |
| Route recall | n/a | 0.8943 → 0.8912 |
| Route F1 | n/a | 0.8871 → 0.9345 |

The gain on held-out trips equals the gain on the trips the decoder was tuned on (+0.0683 both), so the
tuning did not overfit. On student_test the HMM fixed **1,610** observations and made **302** worse.

| Trip | Top-1 classifier | Top-1 + HMM | Change | Fixed | Made worse |
|---|---|---|---|---|---|
| 00000000 | 0.9497 | 0.9884 | +0.0388 | 107 | 10 |
| 00000007 | 0.9498 | 1.0000 | +0.0502 | 68 | 0 |
| 00000009 | 0.8818 | 0.9247 | +0.0429 | 65 | 20 |
| 00000040 | 0.9482 | 0.9920 | +0.0437 | 140 | 9 |
| 00000052 | 0.9771 | 0.9924 | +0.0153 | 6 | 2 |
| 00000056 | 0.9090 | 0.9677 | +0.0586 | 272 | 40 |
| 00000065 | 0.8516 | 0.8451 | **−0.0065** | 27 | 32 |
| 00000076 | 0.9145 | 0.9758 | +0.0612 | 106 | 15 |
| 00000077 | 0.7471 | 0.8987 | +0.1516 | 757 | 164 |
| 00000091 | 0.8959 | 0.9554 | +0.0595 | 62 | 10 |

### Where HMM decoding did not help
- **Trip 00000065** is the only trip that got slightly worse (27 fixed, 32 broken).
- **Route recall did not improve** (0.8943 → 0.8912). The HMM can only choose among the candidates, so a
  reference arc that was never a candidate (0.26 % of student-test observations) cannot be recovered; the
  gain comes from correcting wrong choices, which raises precision and continuity.
- **Direction (reverse-twin) confusion is both what the HMM fixes most and what it breaks most.**
  About 92 % of the road network is two-way (mean per trip, `a_eda_summary.json`); a two-way road is stored as
  two arcs at the same position. Of the 302 observations made worse, in **181 (59.9 %)** the HMM chose the opposite-direction twin of the
  reference arc, 78 (25.8 %) a neighbouring arc (≤ 2 hops) and 43 (14.2 %) an arc elsewhere. Of the 1,610
  fixed observations, 741 (46.0 %) were cases where the classifier had picked the opposite twin. In trip
  00000065, 24 of its 32 worsened observations are twin errors.
- A likely explanation (not tested): staying on a chosen arc costs nothing, so once the decoder locks onto
  the wrong direction of a two-way road for a stretch, it has little reason to switch, and the labels are
  least certain where roads run in parallel.

## Figures to include
`b_hmm_pertrip.png` (per-trip Top-1), `b_hmm_gain.png` (gain on tuned-on vs held-out data),
`b_hmm_tuning.png` (all 216 settings), and the Folium map with the HMM route and a made-worse layer
(`b_hmm_worse_obs.csv`).
