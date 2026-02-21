# PyTorch CUDA Graphs POC

This POC compares eager training vs CUDA Graph replay for a small MLP and exports PyTorch profiler traces that open in Perfetto.
Profiling uses `torch.profiler.schedule(wait, warmup, active, repeat)` with `on_trace_ready`.

## Files

- `cuda_graph_poc.py`: training + benchmarking + optional profiler trace export
- `results/*.json`: benchmark/profile metrics from runs in this workspace
- `profiles/*.json`: Chrome trace files (`eager.json`, `cuda_graph.json`) for Perfetto

## Run Benchmarks (no profiler)

```bash
python cuda_graph_poc.py --metrics-json results/eager.json

python cuda_graph_poc.py --use-cuda-graph --metrics-json results/graph.json
```

## Run Profiler Traces for Perfetto

```bash
python cuda_graph_poc.py --profile --trace-name eager --metrics-json results/eager_profile.json

python cuda_graph_poc.py --use-cuda-graph --profile --trace-name cuda_graph --metrics-json results/graph_profile.json
```

Optional schedule tuning:

```bash
--profile-wait 10 --profile-warmup 10 --profile-active 100 --profile-repeat 1
```

Generated traces:

- `profiles/eager.json`
- `profiles/cuda_graph.json`

## Open in Perfetto

1. Open https://ui.perfetto.dev
2. Click `Open trace file`
3. Load either `profiles/eager.json` or `profiles/cuda_graph.json`

## Current Validation Results

From `results/eager.json` and `results/graph.json` (`warmup_steps=25`, `train_steps=1200`):

- Eager total time: `1.3150 s`
- CUDA Graph total time: `0.3445 s`
- Speedup: `3.82x`
- Final loss delta: `9.54e-07`

With profiler enabled (`results/eager_profile.json` and `results/graph_profile.json`):

- Eager total time: `2.1865 s`
- CUDA Graph total time: `0.5635 s`
- Speedup: `3.88x`
- Final loss delta: `9.54e-07`

This satisfies:

1. CUDA Graph run is faster for same parameters.
2. Final losses match (to near machine precision) with fixed seed.
