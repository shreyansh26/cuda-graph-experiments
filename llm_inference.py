import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

from prompt_bank import VARIABLE_SIZE_PROMPTS_64


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
        self.decode_mode = decode_mode
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_vars: dict[int, dict[str, Any]] = {}
        self.graph_bs: List[int] = []
        self.graph_pool: Any = None
        self.graph_max_batch_size = 0
        self.graph_max_cache_len = 0
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
            "kv_cache_cudagraphs": self.forward_decode_greedy_cudagraphs,
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

    @staticmethod
    def _build_graph_batch_sizes(max_batch_size: int) -> List[int]:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        # Capture a small exact set first, then coarse-grained buckets in steps of 16.
        rounded_max_bs = 8 if max_batch_size <= 8 else ((max_batch_size + 15) // 16) * 16
        return [1, 2, 4, 8] + list(range(16, rounded_max_bs + 1, 16))

    def _select_graph_batch_size(self, batch_size: int) -> int:
        # self.graph_bs is sorted ascending; this returns the smallest bucket >= batch_size.
        return next(bs for bs in self.graph_bs if bs >= batch_size)

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
    def capture_cudagraphs(self, max_batch_size: int, max_cache_len: int) -> None:
        if self.device.type != "cuda":
            return
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if max_cache_len <= 0:
            raise ValueError("max_cache_len must be > 0")

        # We capture one graph per batch-size bucket and choose at runtime by ceil(batch_size).
        graph_bs = self._build_graph_batch_sizes(max_batch_size)
        self.graph_bs = graph_bs
        self.graphs = {}
        self.graph_vars = {}
        self.graph_pool = None

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0 if self.eos_token_id is None else int(self.eos_token_id)

        # Capture larger buckets first so the shared graph pool is sized from the largest case.
        for bs in reversed(graph_bs):
            graph = torch.cuda.CUDAGraph()
            past_key_values = StaticCache(
                config=self.model.config,
                max_cache_len=max_cache_len,
            )
            input_ids = torch.full(
                (bs, 1),
                int(pad_token_id),
                dtype=torch.int64,
                device=self.device,
            )
            position_ids = torch.zeros((bs, 1), dtype=torch.int64, device=self.device)
            cache_position = torch.zeros((1,), dtype=torch.int64, device=self.device)

            # One eager warmup to materialize kernels/allocations before capture.
            past_key_values.reset()
            _ = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values.reset()

            # Capture with static tensor addresses; replay will reuse these exact buffers.
            with torch.cuda.graph(graph, self.graph_pool):
                out = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            if self.graph_pool is None:
                # Reuse one graph memory pool across all captures to reduce VRAM duplication.
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            # Keep captured input/output tensors so caller can mutate inputs before replay.
            self.graph_vars[bs] = {
                "past_key_values": past_key_values,
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "logits": out.logits,
            }
            torch.cuda.synchronize(self.device)

        self.graph_max_batch_size = graph_bs[-1]
        self.graph_max_cache_len = max_cache_len

    @torch.inference_mode()
    def forward_decode_greedy_cudagraphs(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Union[int, Sequence[int], None] = None,
    ) -> tuple[torch.Tensor, DecodeStats]:
        if self.device.type != "cuda":
            return self.forward_decode_greedy(
                prompt_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )

        batch_size = int(prompt_ids.shape[0])
        prompt_len = int(prompt_ids.shape[1])
        required_cache_len = prompt_len + max_new_tokens
        if (
            len(self.graphs) == 0
            or self.graph_max_batch_size < batch_size
            or self.graph_max_cache_len < required_cache_len
        ):
            # (Re)capture whenever current request exceeds captured batch/cache limits.
            self.capture_cudagraphs(
                max_batch_size=batch_size,
                max_cache_len=required_cache_len,
            )

        # Pick the smallest captured bucket that can accommodate this request.
        graph_bs = self._select_graph_batch_size(batch_size)
        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars[graph_bs]
        past_key_values: StaticCache = graph_vars["past_key_values"]
        graph_input_ids: torch.Tensor = graph_vars["input_ids"]
        graph_position_ids: torch.Tensor = graph_vars["position_ids"]
        graph_cache_position: torch.Tensor = graph_vars["cache_position"]
        graph_logits: torch.Tensor = graph_vars["logits"]

        eos_ids = self._normalize_eos_ids(
            self.eos_token_id if eos_token_id is None else eos_token_id
        )
        eos_fill_id = eos_ids[0] if len(eos_ids) > 0 else None
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_fill_id if eos_fill_id is not None else 0

        prefill_input_ids = torch.full(
            (graph_bs, prompt_len),
            int(pad_token_id),
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        # Build a graph_bs-shaped prefill so KV cache layout matches the selected graph bucket.
        prefill_attention_mask = torch.zeros(
            (graph_bs, prompt_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        prefill_input_ids[:batch_size].copy_(prompt_ids)
        prefill_attention_mask[:batch_size].copy_(attention_mask)
        if graph_bs > batch_size:
            # Dummy rows must not be fully masked (can cause invalid attention softmax paths).
            prefill_attention_mask[batch_size:, -1] = 1

        # Position ids are computed independently per row (dim=-1), never across rows.
        # For a dummy row with mask [0, 0, ..., 1]:
        #   cumsum -> [0, 0, ..., 1]
        #   minus 1 -> [-1, -1, ..., 0]
        #   masked_fill(mask == 0, 0) -> [0, 0, ..., 0]
        # So dummy rows do not inherit any values from valid rows; they get neutral position ids.
        prefill_position_ids = prefill_attention_mask.to(torch.int64).cumsum(dim=-1) - 1
        prefill_position_ids.masked_fill_(prefill_attention_mask == 0, 0)

        self._sync_if_cuda()
        t0 = time.perf_counter()
        past_key_values.reset()
        out = self.model(
            input_ids=prefill_input_ids,
            attention_mask=prefill_attention_mask,
            position_ids=prefill_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        self._sync_if_cuda()
        t1 = time.perf_counter()

        next_token = torch.argmax(out.logits[:batch_size, -1, :], dim=-1, keepdim=True)
        generated: List[torch.Tensor] = [next_token]
        generated_lengths = torch.ones(batch_size, dtype=torch.int64, device=prompt_ids.device)
        finished = self._eos_mask(next_token, eos_ids).squeeze(1)
        running_position_ids = attention_mask.to(torch.int64).sum(dim=1, keepdim=True)
        running_cache_position = torch.tensor(
            [prompt_len],
            dtype=torch.int64,
            device=prompt_ids.device,
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
            # Overwrite captured input buffers in-place; replay reads these current values.
            graph_input_ids.fill_(int(pad_token_id))
            graph_input_ids[:batch_size].copy_(model_input_ids)
            graph_position_ids.zero_()
            graph_position_ids[:batch_size].copy_(running_position_ids)
            graph_cache_position.copy_(running_cache_position)
            # Executes the graph captured in capture_cudagraphs with current buffer contents.
            graph.replay()
            # Dummy rows are ignored; only real request rows are consumed.
            logits = graph_logits[:batch_size, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
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
            # Advance per-sequence position and shared decode step for next replay.
            running_position_ids += 1
            running_cache_position += 1

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
        if self.decode_mode == "kv_cache_cudagraphs":
            required_cache_len = int(prompt_ids.shape[1] + max_new_tokens)
            if (
                len(self.graphs) == 0
                or self.graph_max_batch_size < int(prompt_ids.shape[0])
                or self.graph_max_cache_len < required_cache_len
            ):
                # Ensure warmup/timed loops measure replay path, not first-time capture.
                self.capture_cudagraphs(
                    max_batch_size=int(prompt_ids.shape[0]),
                    max_cache_len=required_cache_len,
                )

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
        choices=["kv_cache", "kv_cache_cudagraphs", "no_kv_cache"],
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
