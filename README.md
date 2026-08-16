# Quantum Kernel Sketch

Train a kernel perceptron on angle-encoded data and compare its predictions with a kernel-weighted baseline.

**Python** · Reusable core · CLI examples · Regression checks · Reproducible benchmark

> **Provenance:** created with AI assistance on **2026-11-01**.
> All commit author and committer dates are synthetic and distributed across
> November 2024–October 2026. They do not establish work on those dates.

## Quick start

```sh
python3 -m qkernel examples/clusters.json --epochs 20
```

## Repository map

- `qkernel/core.py` — algorithm primitives.
- `qkernel/features.py` — experiment and analysis APIs.
- `qkernel/cli.py` — argument parsing and JSON output.
- `examples/` — small reproducible inputs.
- `tests/` — API, numerical, and CLI checks.
- `benchmarks/run.py` — timed representative workload; machine-dependent measurements.

## How it works

The fidelity kernel is the product of squared cosine differences. A precomputed Gram matrix drives mistake-based perceptron updates. Evaluation uses a separate holdout and reports a kernel-average baseline alongside the learned decision rule.

## Verification

```sh
sh scripts/check.sh
```

Eight Python test groups exercise reference cases, invalid inputs, and relevant invariants.
The complete local run is in [docs/verification.txt](docs/verification.txt).
The [checks workflow](.github/workflows/check.yml) executes the same source checks.

## Boundaries

The supplied dataset is deliberately tiny. Product-state feature maps are efficiently classically computable; accuracy here is not evidence of quantum advantage or performance on real-world data.

See [architecture](docs/architecture.md), [usage](docs/usage.md),
[release notes](CHANGELOG.md), and [provenance](PROVENANCE.json).

## License

MIT — [LICENSE](LICENSE).
