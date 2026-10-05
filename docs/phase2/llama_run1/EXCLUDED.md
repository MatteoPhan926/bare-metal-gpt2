Diagnostic: this initial custom harness left llama.cpp debug logging enabled,
including a log on every graph replay. Detected by reading stderr after the run.
Do not use as the external speed denominator; `llama_matched1..3` disable debug
logging like upstream llama-bench. Original samples remain available here.
