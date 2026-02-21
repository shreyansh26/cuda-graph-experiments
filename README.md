# PyTorch CUDA Graphs POC

This POC compares eager training vs CUDA Graph replay for a small MLP and exports PyTorch profiler traces that open in Perfetto.

## Files

- `cuda_graph_poc.py`: training + benchmarking + optional profiler trace export
- `results/*.json`: benchmark/profile metrics from runs in this workspace
- `profiles/*.json`: Chrome trace files (`eager.json`, `cuda_graph.json`) for Perfetto

## Environment

- Conda env: `shreyansh-env-py12`
- GPU: `1` (via `CUDA_VISIBLE_DEVICES=1`)

## Run Benchmarks (no profiler)

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n shreyansh-env-py12 \
  python cuda_graph_poc.py --metrics-json results/eager_default.json

CUDA_VISIBLE_DEVICES=1 conda run -n shreyansh-env-py12 \
  python cuda_graph_poc.py --use-cuda-graph --metrics-json results/graph_default.json
```

## Run Profiler Traces for Perfetto

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n shreyansh-env-py12 \
  python cuda_graph_poc.py --profile --train-steps 300 --warmup-steps 20 \
  --trace-name eager --metrics-json results/eager_profile.json

CUDA_VISIBLE_DEVICES=1 conda run -n shreyansh-env-py12 \
  python cuda_graph_poc.py --use-cuda-graph --profile --train-steps 300 --warmup-steps 20 \
  --trace-name cuda_graph --metrics-json results/graph_profile.json
```

Generated traces:

- `profiles/eager.json`
- `profiles/cuda_graph.json`

## Open in Perfetto

1. Open https://ui.perfetto.dev
2. Click `Open trace file`
3. Load either `profiles/eager.json` or `profiles/cuda_graph.json`

## Current Validation Results

From `results/eager_default.json` and `results/graph_default.json`:

- Eager total time: `1.2975 s`
- CUDA Graph total time: `0.5449 s`
- Speedup: `2.38x`
- Final loss delta: `9.54e-07`

This satisfies:

1. CUDA Graph run is faster for same parameters.
2. Final losses match (to near machine precision) with fixed seed.
