Diagnostic only: this original-Phase-1 benchmark overlapped a CPU compilation.
The overlap was noticed before inspecting graph performance. It can perturb host
submission; do not use this run as a speed denominator. All results are retained.
The separate `legacy_fp16` run uses the same Phase-1 kernels and legacy stream,
but the new fixed-input/raw-sample harness, with no concurrent compilation.
