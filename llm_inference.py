#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from typing import List, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DecodeStats:
    prefill_s: float
    decode_s: float
    total_s: float
    generated_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward-pass-only chat inference for Llama 1B Instruct."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt for inference.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="Optional system prompt. Pass empty string to disable.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to decode.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warmup runs (excluded from timing).",
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=10,
        help="Number of timed runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(device: torch.device, dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def token_is_eos(token_id: torch.Tensor, eos_token_id: Union[int, Sequence[int], None]) -> bool:
    if eos_token_id is None:
        return False
    val = int(token_id.item())
    if isinstance(eos_token_id, int):
        return val == eos_token_id
    return val in set(int(x) for x in eos_token_id)


@torch.inference_mode()
def forward_decode_greedy(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Union[int, Sequence[int], None],
    device: torch.device,
) -> tuple[torch.Tensor, DecodeStats]:
    sync_if_cuda(device)
    t0 = time.perf_counter()
    out = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
    sync_if_cuda(device)
    t1 = time.perf_counter()

    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated: List[torch.Tensor] = [next_token]
    past_key_values = out.past_key_values

    for _ in range(max_new_tokens - 1):
        if token_is_eos(next_token, eos_token_id):
            break
        out = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)

    sync_if_cuda(device)
    t2 = time.perf_counter()

    generated_ids = torch.cat(generated, dim=1)
    stats = DecodeStats(
        prefill_s=t1 - t0,
        decode_s=t2 - t1,
        total_s=t2 - t0,
        generated_tokens=int(generated_ids.shape[1]),
    )
    return generated_ids, stats


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.timed_iters <= 0:
        raise ValueError("--timed-iters must be > 0")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be >= 0")

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    model.to(device)
    model.eval()

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    eos_token_id = model.generation_config.eos_token_id
    prompt_tokens = int(prompt_ids.shape[1])

    print(f"Model: {args.model_id}")
    print(f"Device: {device.type}, dtype: {dtype}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Warmup iters: {args.warmup_iters}, timed iters: {args.timed_iters}")

    for _ in range(args.warmup_iters):
        _ = forward_decode_greedy(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
        )

    all_stats: List[DecodeStats] = []
    last_generated_ids = None
    for _ in range(args.timed_iters):
        generated_ids, stats = forward_decode_greedy(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
        )
        last_generated_ids = generated_ids
        all_stats.append(stats)

    avg_prefill = sum(s.prefill_s for s in all_stats) / len(all_stats)
    avg_decode = sum(s.decode_s for s in all_stats) / len(all_stats)
    avg_total = sum(s.total_s for s in all_stats) / len(all_stats)
    avg_gen_tokens = sum(s.generated_tokens for s in all_stats) / len(all_stats)

    total_toks_per_s = avg_gen_tokens / avg_total if avg_total > 0 else float("inf")
    decode_toks_per_s = avg_gen_tokens / avg_decode if avg_decode > 0 else float("inf")

    print("")
    print("Timing (averaged over timed iterations):")
    print(f"  Prefill latency: {avg_prefill * 1e3:.3f} ms")
    print(f"  Decode latency:  {avg_decode * 1e3:.3f} ms")
    print(f"  Total latency:   {avg_total * 1e3:.3f} ms")
    print(f"  New tokens:      {avg_gen_tokens:.2f}")
    print(f"  Throughput total: {total_toks_per_s:.2f} tok/s")
    print(f"  Throughput decode: {decode_toks_per_s:.2f} tok/s")

    if last_generated_ids is not None:
        response = tokenizer.decode(last_generated_ids[0], skip_special_tokens=True)
        print("")
        print("Model response:")
        print(response)


if __name__ == "__main__":
    main()
