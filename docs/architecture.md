# Architecture

The fidelity kernel is the product of squared cosine differences. A precomputed Gram matrix drives mistake-based perceptron updates. Evaluation uses a separate holdout and reports a kernel-average baseline alongside the learned decision rule.

## Data flow

- `qkernel/core.py` — algorithm primitives.
- `qkernel/features.py` — experiment and analysis APIs.
- `qkernel/cli.py` — argument parsing and JSON output.
- `examples/` — small reproducible inputs.
- `tests/` — API, numerical, and CLI checks.
- `benchmarks/run.py` — timed representative workload.

## Limits

The supplied dataset is deliberately tiny. Product-state feature maps are efficiently classically computable; accuracy here is not evidence of quantum advantage or performance on real-world data.
