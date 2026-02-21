#!/usr/bin/env python3
"""POC: compare eager training vs CUDA Graph replay in PyTorch.

The script trains a tiny MLP on synthetic, fixed-shape data and reports:
- total wall-clock training time
- average step time
- final loss

Optional PyTorch profiler traces are exported as Chrome trace JSON files,
which can be loaded directly into https://ui.perfetto.dev.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity


class TinyMLP(nn.Module):
    """Small MLP with many lightweight kernels to expose launch overhead."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, depth: int) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA Graphs training POC with profiling")
    parser.add_argument("--use-cuda-graph", action="store_true", help="Capture and replay the training step")
    parser.add_argument("--seed", type=int, default=1234, help="Global seed for deterministic setup")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-2)

    parser.add_argument("--warmup-steps", type=int, default=25, help="Untimed eager warmup steps")
    parser.add_argument("--train-steps", type=int, default=1200, help="Timed benchmark steps")

    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--profile-dir", type=Path, default=Path("profiles"), help="Trace output directory")
    parser.add_argument(
        "--trace-name",
        type=str,
        default=None,
        help="Optional trace filename stem (default: eager or cuda_graph)",
    )

    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional path to write metrics JSON")

    args = parser.parse_args()

    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.train_steps <= 0:
        parser.error("--train-steps must be > 0")
    if args.depth <= 0:
        parser.error("--depth must be > 0")

    return args


def set_determinism(seed: int) -> None:
    """Set deterministic seeds and backend flags.

    We avoid forcing `torch.use_deterministic_algorithms(True)` here because that
    requires a specific cuBLAS workspace env var and can fail out-of-the-box.
    Fixed seeds still keep eager vs graph comparisons reproducible for this POC.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_dataset(
    *,
    steps: int,
    batch_size: int,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic synthetic batches directly on GPU.

    Shapes:
    - inputs:  [steps, batch_size, input_dim]
    - targets: [steps, batch_size]
    """

    data_gen = torch.Generator(device=device)
    data_gen.manual_seed(seed + 99)

    inputs = torch.randn(
        steps,
        batch_size,
        input_dim,
        device=device,
        generator=data_gen,
        dtype=torch.float32,
    )

    # Teacher projection defines deterministic class targets.
    teacher_w = torch.randn(input_dim, num_classes, device=device, generator=data_gen)
    logits = torch.einsum("sbi,ic->sbc", inputs, teacher_w)
    targets = logits.argmax(dim=-1)

    return inputs, targets


def build_profiler(enabled: bool) -> Optional[torch.profiler.profile]:
    if not enabled:
        return None

    return torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def run_once(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this POC.")

    device = torch.device("cuda")
    set_determinism(args.seed)

    model = TinyMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        depth=args.depth,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Plain SGD keeps eager and graph update math equivalent for parity checks.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # We use one additional untimed "prime" step.
    # In graph mode, capture executes once and becomes that prime step.
    total_steps = args.warmup_steps + 1 + args.train_steps
    inputs, targets = make_dataset(
        steps=total_steps,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        device=device,
        seed=args.seed,
    )

    def eager_step(batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        return loss

    mode = "cuda_graph" if args.use_cuda_graph else "eager"

    static_loss: Optional[torch.Tensor] = None
    graph: Optional[torch.cuda.CUDAGraph] = None
    static_x: Optional[torch.Tensor] = None
    static_y: Optional[torch.Tensor] = None

    if args.use_cuda_graph:
        static_x = torch.empty_like(inputs[0])
        static_y = torch.empty_like(targets[0])

        # Warmup on a side stream before capture so parameter grads/state are materialized.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for step in range(args.warmup_steps):
                static_x.copy_(inputs[step], non_blocking=True)
                static_y.copy_(targets[step], non_blocking=True)
                eager_step(static_x, static_y)

        torch.cuda.current_stream().wait_stream(warmup_stream)

        # Prime batch for capture (this performs one untimed training update).
        prime_step = args.warmup_steps
        static_x.copy_(inputs[prime_step], non_blocking=True)
        static_y.copy_(targets[prime_step], non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = model(static_x)
            static_loss = criterion(static_output, static_y)
            static_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    else:
        for step in range(args.warmup_steps):
            eager_step(inputs[step], targets[step])

        # Match the untimed prime update used by graph capture path.
        eager_step(inputs[args.warmup_steps], targets[args.warmup_steps])

    profiler = build_profiler(args.profile)
    if profiler is not None:
        profiler.__enter__()

    start_step = args.warmup_steps + 1
    end_step = start_step + args.train_steps

    torch.cuda.synchronize()
    start_t = time.perf_counter()

    for step in range(start_step, end_step):
        if args.use_cuda_graph:
            assert graph is not None and static_x is not None and static_y is not None
            static_x.copy_(inputs[step], non_blocking=True)
            static_y.copy_(targets[step], non_blocking=True)
            graph.replay()
            loss = static_loss
        else:
            loss = eager_step(inputs[step], targets[step])

        if profiler is not None:
            profiler.step()

    torch.cuda.synchronize()
    total_time_s = time.perf_counter() - start_t

    trace_path = None
    if profiler is not None:
        profiler.__exit__(None, None, None)
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        trace_stem = args.trace_name or mode
        trace_path = args.profile_dir / f"{trace_stem}.json"
        profiler.export_chrome_trace(str(trace_path))

    # One final synchronized read for scalar loss reporting.
    final_loss = float(loss.detach().item())

    result: Dict[str, Any] = {
        "mode": mode,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "train_steps": args.train_steps,
        "total_time_s": total_time_s,
        "avg_step_ms": (total_time_s / args.train_steps) * 1000.0,
        "final_loss": final_loss,
        "trace_file": str(trace_path) if trace_path is not None else None,
    }

    return result


def main() -> None:
    args = parse_args()
    result = run_once(args)

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
