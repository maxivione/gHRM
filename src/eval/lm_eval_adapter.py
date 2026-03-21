"""lm-eval-harness adapter for CausalHRM.

Usage:
    lm_eval --model ghrm \
        --model_args checkpoint=/path/to/ckpt,max_seq_len=512 \
        --tasks hellaswag,arc_easy,winogrande \
        --batch_size 8
"""
from __future__ import annotations

import torch
import tiktoken
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance

from src.models.causal_hrm import CausalHRM, CausalHRMConfig


@register_model("ghrm")
class GHRMEvalWrapper(LM):
    def __init__(self, checkpoint: str = "", max_seq_len: int = 512,
                 device: str = "cuda", batch_size: int = 1, **kwargs):
        super().__init__()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = int(batch_size)

        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_id = self.enc.eot_token

        # Load or create model
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=self._device, weights_only=False)
            cfg_dict = ckpt.get('config', {})
            cfg_dict['max_seq_len'] = int(max_seq_len)
            cfg = CausalHRMConfig(**cfg_dict)
            self.model = CausalHRM(cfg).to(self._device)
            self.model.load_state_dict(ckpt['model'])
        else:
            cfg = CausalHRMConfig(max_seq_len=int(max_seq_len))
            self.model = CausalHRM(cfg).to(self._device)

        self.model.eval()
        self.config = cfg

    @property
    def eot_token_id(self):
        return self.eot_id

    @property
    def max_length(self):
        return self.config.max_seq_len

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.enc.encode(string, allowed_special="all")

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self.enc.decode(tokens)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run model forward, return logits (B, T, V)."""
        with torch.no_grad():
            out = self.model(input_ids.to(self._device))
        return out['logits']

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok_encode(context) if context else [self.eot_id]
            cont_ids = self.tok_encode(continuation)
            all_ids = ctx_ids + cont_ids

            # Truncate from left if needed
            if len(all_ids) > self.config.max_seq_len:
                all_ids = all_ids[-self.config.max_seq_len:]
                cont_ids = cont_ids[-(len(all_ids) - max(len(ctx_ids), 0)):]

            input_ids = torch.tensor([all_ids], device=self._device)
            logits = self._model_call(input_ids)[0]  # (T, V)

            # Shift: logits[i] predicts token[i+1]
            # We want logprobs of cont_ids, which start at position len(ctx_ids)
            cont_start = len(all_ids) - len(cont_ids)
            log_probs = torch.log_softmax(logits, dim=-1)

            total_ll = 0.0
            is_greedy = True
            for i, tok_id in enumerate(cont_ids):
                pos = cont_start + i - 1  # logits at pos predict token at pos+1
                if pos < 0:
                    continue
                ll = log_probs[pos, tok_id].item()
                total_ll += ll
                if logits[pos].argmax().item() != tok_id:
                    is_greedy = False

            results.append((total_ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
        results = []
        for req in requests:
            (text,) = req.args
            token_ids = self.tok_encode(text)
            if not token_ids:
                results.append((0.0,))
                continue

            # Prepend EOT
            all_ids = [self.eot_id] + token_ids

            total_ll = 0.0
            # Process in windows
            stride = self.config.max_seq_len
            for start in range(0, len(all_ids) - 1, stride):
                end = min(start + stride, len(all_ids))
                chunk = all_ids[start:end]
                input_ids = torch.tensor([chunk], device=self._device)
                logits = self._model_call(input_ids)[0]
                log_probs = torch.log_softmax(logits, dim=-1)

                # Sum logprobs for each predicted token
                target_start = max(1, start) - start  # skip context-only positions
                for i in range(target_start, len(chunk) - 1):
                    total_ll += log_probs[i, chunk[i + 1]].item()

            results.append((total_ll,))
        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", ["\n"])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_ids = self.tok_encode(context)
            if len(ctx_ids) > self.config.max_seq_len:
                ctx_ids = ctx_ids[-self.config.max_seq_len:]

            input_ids = torch.tensor([ctx_ids], device=self._device)
            output_ids = self.model.generate(
                input_ids, max_new_tokens=max_gen,
                temperature=0.0001, top_k=1,  # greedy
            )
            gen_ids = output_ids[0, len(ctx_ids):].tolist()
            gen_text = self.tok_decode(gen_ids)

            # Truncate at stop sequences
            for stop in until:
                idx = gen_text.find(stop)
                if idx >= 0:
                    gen_text = gen_text[:idx]
                    break

            results.append(gen_text)
        return results
