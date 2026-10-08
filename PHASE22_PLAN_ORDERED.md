# Phase 2.2 — order-preserving repair, preregistered before timing

The initial V4 partial-sum variant is rejected, not silently superseded:
`docs/phase22/cross_policy_gemv_diagnostic` fails the preregistered 1e-2
cross-policy relative-logit gate at position 173 (0.02041785), despite KL
1.8243e-5 and passing the original HF gates. No V4 performance run occurred.
Its implementation and artifacts remain. PHASE22_PLAN.md is unchanged.

This is a replacement implementation of the SAME value-loop experiment,
not a stacked optimization: use the original 64-thread/head block, load four
successive probabilities and values ahead, then perform the four additions
in their original order. Keep QK, softmax, grid, logical traffic, and every
surrounding kernel unchanged. No partial-sum regrouping remains in this path.

**Refined mechanism:** dependent value loads/instructions, rather than just
floating-point addition dependency, materially limit the serial value loop.
Four independent loads expose instruction-level parallelism without changing
the numerical sum. The strong competitor remains strided QK access/latency;
another competitor is that the compiler already schedules equivalent loads.

**Prediction:** >=25% lower long attention time; 8–22% long GPU-forward and
5–20% complete-generation latency reduction. Short regression <=5%. Same
three-process end-to-end criteria, clocks, contexts, raw-sample retention and
unchanged controls as PHASE22_PLAN.md. This deliberately keeps the original
performance bar instead of fitting a lower bar after a failed implementation.

**Additional correctness constraint:** baseline versus ordered-load variant
must be BIT-IDENTICAL on all logits and captured activations across all 1024
positions and reset/prefill checks, as well as passing original HF gates for
both precisions. Explicitly inspect compiler resources/instructions if the
change fails to expose independent loads. No gate is weakened.

**Kill:** any correctness failure, <8% long GPU-forward gain, central long
generation gain <5%, or consistent >5% short regression means no promotion.
If this loses, finish with a negative result about this value-loop mechanism;
do not add key-coalescing or split-context kernels in this iteration.

The scientific contrast is now: unsafe reassociation (failed correctness),
versus order-preserving memory-latency hiding (untimed at this registration).
