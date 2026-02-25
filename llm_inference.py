#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_bank import VARIABLE_SIZE_PROMPTS_64  # pyright: ignore[reportMissingImports]


@dataclass
class DecodeStats:
    prefill_s: float
    decode_s: float
    total_s: float
    generated_tokens: int


class ModelRunner:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        decode_mode: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = model.generation_config.eos_token_id
        self.decode_fn: Callable[
            [torch.Tensor, torch.Tensor, int, Union[int, Sequence[int], None]],
            tuple[torch.Tensor, DecodeStats],
        ] = self._select_decode_fn(decode_mode)

    def _select_decode_fn(
        self, decode_mode: str
    ) -> Callable[
        [torch.Tensor, torch.Tensor, int, Union[int, Sequence[int], None]],
        tuple[torch.Tensor, DecodeStats],
    ]:
        decode_fns = {
            "kv_cache": self.forward_decode_greedy,
            "no_kv_cache": self.forward_decode_greedy_without_kvcache,
        }
        if decode_mode not in decode_fns:
            raise ValueError(f"Unsupported decode mode: {decode_mode}")
        return decode_fns[decode_mode]

    def _sync_if_cuda(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _normalize_eos_ids(eos_token_id: Union[int, Sequence[int], None]) -> List[int]:
        if eos_token_id is None:
            return []
        if isinstance(eos_token_id, int):
            return [int(eos_token_id)]
        return [int(x) for x in eos_token_id]

    @staticmethod
    def _eos_mask(token_ids: torch.Tensor, eos_ids: Sequence[int]) -> torch.Tensor:
        if len(eos_ids) == 0:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        mask = token_ids.eq(eos_ids[0])
        for eos_id in eos_ids[1:]:
            mask |= token_ids.eq(eos_id)
        return mask

    def build_prompt_batch(
        self, prompts: Sequence[str], system_prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
        formatted_prompts: List[str] = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        tokenized = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True)
        prompt_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        return prompt_ids, attention_mask, prompt_lengths

    @torch.inference_mode()
    def forward_decode_greedy(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Union[int, Sequence[int], None] = None,
    ) -> tuple[torch.Tensor, DecodeStats]:
        batch_size = int(prompt_ids.shape[0])
        eos_ids = self._normalize_eos_ids(
            self.eos_token_id if eos_token_id is None else eos_token_id
        )
        eos_fill_id = eos_ids[0] if len(eos_ids) > 0 else None

        self._sync_if_cuda()
        t0 = time.perf_counter()
        out = self.model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        self._sync_if_cuda()
        t1 = time.perf_counter()

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated: List[torch.Tensor] = [next_token]
        generated_lengths = torch.ones(batch_size, dtype=torch.int64, device=prompt_ids.device)
        finished = self._eos_mask(next_token, eos_ids).squeeze(1)
        past_key_values = out.past_key_values
        running_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

        for _ in range(max_new_tokens - 1):
            if bool(torch.all(finished)):
                break
            model_input_ids = next_token
            if eos_fill_id is not None:
                model_input_ids = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(model_input_ids, eos_fill_id),
                    model_input_ids,
                )
            out = self.model(
                input_ids=model_input_ids,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if eos_fill_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_fill_id),
                    next_token,
                )
            generated.append(next_token)
            unfinished = ~finished
            generated_lengths += unfinished.to(generated_lengths.dtype)
            finished |= self._eos_mask(next_token, eos_ids).squeeze(1)
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=running_attention_mask.dtype,
                        device=running_attention_mask.device,
                    ),
                ],
                dim=1,
            )

        self._sync_if_cuda()
        t2 = time.perf_counter()

        generated_ids = torch.cat(generated, dim=1)
        stats = DecodeStats(
            prefill_s=t1 - t0,
            decode_s=t2 - t1,
            total_s=t2 - t0,
            generated_tokens=int(generated_lengths.sum().item()),
        )
        return generated_ids, stats

    @torch.inference_mode()
    def forward_decode_greedy_without_kvcache(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Union[int, Sequence[int], None] = None,
    ) -> tuple[torch.Tensor, DecodeStats]:
        batch_size = int(prompt_ids.shape[0])
        eos_ids = self._normalize_eos_ids(
            self.eos_token_id if eos_token_id is None else eos_token_id
        )
        eos_fill_id = eos_ids[0] if len(eos_ids) > 0 else None

        self._sync_if_cuda()
        t0 = time.perf_counter()
        out = self.model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        self._sync_if_cuda()
        t1 = time.perf_counter()

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated: List[torch.Tensor] = [next_token]
        generated_lengths = torch.ones(batch_size, dtype=torch.int64, device=prompt_ids.device)
        finished = self._eos_mask(next_token, eos_ids).squeeze(1)
        running_input_ids = torch.cat([prompt_ids, next_token], dim=1)
        running_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

        for _ in range(max_new_tokens - 1):
            if bool(torch.all(finished)):
                break
            out = self.model(
                input_ids=running_input_ids,
                attention_mask=running_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if eos_fill_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_fill_id),
                    next_token,
                )
            generated.append(next_token)
            unfinished = ~finished
            generated_lengths += unfinished.to(generated_lengths.dtype)
            finished |= self._eos_mask(next_token, eos_ids).squeeze(1)
            running_input_ids = torch.cat([running_input_ids, next_token], dim=1)
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=running_attention_mask.dtype,
                        device=running_attention_mask.device,
                    ),
                ],
                dim=1,
            )

        self._sync_if_cuda()
        t2 = time.perf_counter()

        generated_ids = torch.cat(generated, dim=1)
        stats = DecodeStats(
            prefill_s=t1 - t0,
            decode_s=t2 - t1,
            total_s=t2 - t0,
            generated_tokens=int(generated_lengths.sum().item()),
        )
        return generated_ids, stats

    def run_decode_benchmark(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        warmup_iters: int,
        timed_iters: int,
        eos_token_id: Union[int, Sequence[int], None] = None,
    ) -> tuple[torch.Tensor, List[DecodeStats]]:
        for _ in range(warmup_iters):
            _ = self.decode_fn(
                prompt_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )

        all_stats: List[DecodeStats] = []
        last_generated_ids = None
        for _ in range(timed_iters):
            generated_ids, stats = self.decode_fn(
                prompt_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
            last_generated_ids = generated_ids
            all_stats.append(stats)

        if last_generated_ids is None:
            raise RuntimeError("No decode output produced; timed_iters must be > 0")
        return last_generated_ids, all_stats

    @staticmethod
    def print_timing_summary(all_stats: List[DecodeStats], batch_size: int) -> None:
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
        print(f"  New tokens total: {avg_gen_tokens:.2f}")
        print(f"  New tokens/sample: {avg_gen_tokens / batch_size:.2f}")
        print(f"  Throughput total: {total_toks_per_s:.2f} tok/s")
        print(f"  Throughput decode: {decode_toks_per_s:.2f} tok/s")

    def decode_responses(self, generated_ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


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
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to pick from static prompt bank.",
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
    parser.add_argument(
        "--decode-mode",
        type=str,
        default="kv_cache",
        choices=["kv_cache", "no_kv_cache"],
        help="Decode implementation to benchmark.",
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


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.timed_iters <= 0:
        raise ValueError("--timed-iters must be > 0")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.batch_size > len(VARIABLE_SIZE_PROMPTS_64):
        raise ValueError(
            f"--batch-size must be <= {len(VARIABLE_SIZE_PROMPTS_64)} "
            "(size of static prompt bank)"
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        decode_mode=args.decode_mode,
    )

    selected_prompts = VARIABLE_SIZE_PROMPTS_64[: args.batch_size]
    prompt_ids, attention_mask, prompt_lengths = runner.build_prompt_batch(
        prompts=selected_prompts,
        system_prompt=args.system_prompt,
    )
    min_prompt_tokens = min(prompt_lengths)
    max_prompt_tokens = max(prompt_lengths)
    avg_prompt_tokens = sum(prompt_lengths) / len(prompt_lengths)

    print(f"Model: {args.model_id}")
    print(f"Device: {device.type}, dtype: {dtype}")
    print(f"Decode mode: {args.decode_mode}")
    print(f"Batch size: {args.batch_size}")
    print(
        "Prompt tokens (min / avg / max): "
        f"{min_prompt_tokens} / {avg_prompt_tokens:.2f} / {max_prompt_tokens}"
    )
    print(f"Warmup iters: {args.warmup_iters}, timed iters: {args.timed_iters}")

    last_generated_ids, all_stats = runner.run_decode_benchmark(
        prompt_ids=prompt_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )
    runner.print_timing_summary(all_stats, args.batch_size)

    responses = runner.decode_responses(last_generated_ids)
    print("")
    print("Model responses:")
    for idx, response in enumerate(responses):
        print(f"[{idx}] {response}")


if __name__ == "__main__":
    main()
