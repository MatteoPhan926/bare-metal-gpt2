# Interrupted, not a completed benchmark

The Codex server restart killed the serial suite during this process. There
is no completion receipt. Existing stdout/stderr/telemetry are retained here;
the incomplete run was renamed from `fp16_run3`, and the missing third process
will be rerun. Runs 1 and 2 for both precisions and llama were already complete
and are kept, not repeated or selected by speed. No samples from this partial
process enter the aggregate.
