"""LoRA training loop shared by ``CircuitBreakerTrainer.fit`` and ``RepBendTrainer.fit``.

Circuit Breakers (Zou et al. 2024) and RepBend (Yousefpour et al. 2025) are
training-time methods: their losses (``CircuitBreakerModel.compute_rr_loss`` and
``RepBendModel.compute_repbend_loss``) compare a LoRA-adapted model against the
frozen original. This module supplies the loop that optimises them:

* a fresh PEFT LoRA adapter on the attention projections;
* the frozen reference is the same model with the adapter disabled, run in
  ``eval()`` mode under ``no_grad`` (as in ``GraySwanAI/circuit-breakers``), so no
  second copy of the weights is held;
* rows are rendered with the tokenizer's chat template; the harmful (circuit-breaker
  / unsafe) loss covers the response tokens, or the prompt tokens for a prompt-only
  row; the retain loss covers every non-padding token;
* each optimizer step draws ``batch_size`` harmful and ``batch_size`` retain rows,
  ``gradient_accumulation`` times.

Hidden-state keys are decoder-block indices: block ``l``'s output is HF
``hidden_states[l + 1]``.
"""
from __future__ import annotations

import contextlib
import itertools
import math
import random
import warnings
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import torch

from safetune._refusal_helpers import _get_decoder_layers


def as_rows(data: Iterable[Any], name: str) -> List[Dict[str, Optional[str]]]:
    """Normalise ``data`` to ``[{"prompt": str, "response": str | None}]``.

    Accepts strings (prompt-only rows) or mappings with a ``prompt`` key and an
    optional ``response`` key.
    """
    rows = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            rows.append({"prompt": item, "response": None})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            response = item.get("response")
            if response is not None and not isinstance(response, str):
                raise TypeError(f"{name}[{i}]['response'] must be a string or None")
            rows.append({"prompt": item["prompt"], "response": response or None})
        else:
            raise TypeError(f"{name}[{i}] must be a string or a dict with a 'prompt' string")
    if not rows:
        raise ValueError(f"{name} is empty")
    return rows


class RowEncoder:
    """Render one row and build its token masks.

    Returns ``input_ids``, ``attention_mask``, ``response_mask`` (assistant tokens)
    and ``target_mask`` (the response, or every token for a prompt-only row), or
    ``None`` when truncation to ``max_len`` removes the whole response.
    """

    def __init__(self, tokenizer, max_len: int, chat_template: bool = True):
        self.tok = tokenizer
        self.max_len = int(max_len)
        self.chat_template = bool(chat_template) and bool(getattr(tokenizer, "chat_template", None))

    def _ids(self, text: str, special: bool) -> List[int]:
        return self.tok(text, add_special_tokens=special)["input_ids"]

    def __call__(self, row: Dict[str, Optional[str]]) -> Optional[Dict[str, List[int]]]:
        prompt, response = row["prompt"], row["response"]
        if self.chat_template:
            user = [{"role": "user", "content": prompt}]
            prefix = self._ids(self.tok.apply_chat_template(
                user, tokenize=False, add_generation_prompt=True), special=False)
            full = prefix
            if response:
                full = self._ids(self.tok.apply_chat_template(
                    user + [{"role": "assistant", "content": response}], tokenize=False),
                    special=False)
                if full[:len(prefix)] != prefix:
                    raise ValueError("The chat template does not render the assistant turn as a "
                                     "continuation of the generation prompt; pass "
                                     "chat_template=False")
        else:
            prefix = self._ids(prompt, special=True)
            full = prefix
            if response:
                eos = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []
                full = prefix + self._ids(response, special=False) + eos
        full = full[:self.max_len]
        n_prefix = min(len(prefix), len(full))
        response_mask = [0] * n_prefix + [1] * (len(full) - n_prefix)
        if response and not any(response_mask):
            return None
        target_mask = response_mask if any(response_mask) else [1] * len(full)
        return {"input_ids": full, "attention_mask": [1] * len(full),
                "response_mask": response_mask, "target_mask": target_mask}


def encode_rows(encoder: RowEncoder, rows: Sequence[Dict[str, Optional[str]]], name: str):
    encoded = [encoder(r) for r in rows]
    kept = [e for e in encoded if e is not None]
    dropped = len(encoded) - len(kept)
    if not kept:
        raise ValueError(f"every {name} row lost its response to truncation (max_len="
                         f"{encoder.max_len})")
    if dropped:
        warnings.warn(f"{dropped} {name} row(s) dropped: truncation to max_len="
                      f"{encoder.max_len} removed the whole response", stacklevel=3)
    truncated = sum(len(e["input_ids"]) == encoder.max_len for e in kept)
    return kept, {"rows": len(rows), "dropped": dropped, "at_max_len": truncated}


def collate(rows: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    width = max(len(r["input_ids"]) for r in rows)
    out = {}
    for key, fill in (("input_ids", pad_id), ("attention_mask", 0),
                      ("response_mask", 0), ("target_mask", 0)):
        out[key] = torch.tensor([r[key] + [fill] * (width - len(r[key])) for r in rows])
    return out


def batches(rows: List[Dict[str, List[int]]], batch_size: int, pad_id: int,
            seed: int) -> Iterator[Dict[str, torch.Tensor]]:
    """Endless shuffled batches; every row is seen once per pass."""
    rng = random.Random(seed)
    for _ in itertools.count():
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            chunk = [rows[j] for j in order[i:i + batch_size]]
            if len(chunk) < batch_size and len(rows) >= batch_size:
                continue  # keep the batch size fixed; the next pass reshuffles
            yield collate(chunk, pad_id)


def check_layers(model, layers: Sequence[int]) -> List[int]:
    n = len(_get_decoder_layers(model))
    layers = [int(l) for l in layers]
    bad = [l for l in layers if not 0 <= l < n]
    if not layers or bad:
        raise ValueError(f"target_layers {layers} must be non-empty decoder-block indices in "
                         f"[0, {n}); this model has {n} blocks")
    return layers


def add_lora(model, *, r: int, alpha: int, dropout: float, target_modules: Sequence[str],
             layers_to_transform: Optional[List[int]]):
    from peft import LoraConfig, PeftModel, get_peft_model

    if isinstance(model, PeftModel):
        raise ValueError("fit() adds its own LoRA adapter and uses the adapter-free model as the "
                         "frozen reference; pass a model without a PEFT adapter (merge an "
                         "existing adapter first with merge_and_unload())")
    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=list(target_modules), layers_to_transform=layers_to_transform))
    trainable = [p for n, p in peft_model.named_parameters() if "lora_" in n]
    for p in trainable:  # LoRA weights train in fp32 even on a half-precision base
        p.data = p.data.float()
        p.requires_grad_(True)
    return peft_model, trainable


def decoder_and_head(peft_model) -> tuple:
    """The decoder stack (for hidden states without full-vocabulary logits) and a
    function mapping final hidden states to logits."""
    base = peft_model.get_base_model()
    decoder = base.get_decoder() if hasattr(base, "get_decoder") else None
    head = base.get_output_embeddings()
    if decoder is None or head is None:
        raise ValueError(f"{type(base).__name__} exposes no get_decoder()/get_output_embeddings()")
    scale = float(getattr(base, "logit_scale", 1.0) or 1.0)  # Cohere
    softcap = getattr(base.config, "final_logit_softcapping", None)  # Gemma 2/3

    def logits(h: torch.Tensor) -> torch.Tensor:
        out = head(h) * scale
        if softcap:
            out = torch.tanh(out / softcap) * softcap
        return out

    return decoder, logits


def run_rr_training(
    *,
    model,
    tokenizer,
    harmful: Iterable[Any],
    retain: Iterable[Any],
    layers: Sequence[int],
    step_loss: Callable[..., tuple],
    logits_mask: Optional[str],
    lora_layers: Optional[List[int]],
    max_steps: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    warmup_steps: int,
    max_grad_norm: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    max_len: int,
    chat_template: bool,
    gradient_checkpointing: bool,
    seed: int,
    log_every: int,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Train a LoRA adapter on ``step_loss`` and return ``(peft_model, history, data_stats)``.

    ``step_loss(step, ref, lora, batches)`` receives, for the ``"cb"`` and ``"rt"``
    batches, the reference and LoRA-model ``(hidden, logits)`` pairs, where hidden is
    ``{layer: (B, S, D)}`` and, for the retain batch when ``logits_mask`` names a
    batch mask (``"attention_mask"`` or ``"response_mask"``), logits is the
    ``(n_tokens, V)`` rows at that mask's positions (else ``None``); it returns
    ``(loss, metrics_dict)``.
    """
    for name, value in (("max_steps", max_steps), ("batch_size", batch_size),
                        ("gradient_accumulation", gradient_accumulation), ("max_len", max_len)):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1")
    torch.manual_seed(seed)
    tok = tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise ValueError("the tokenizer has neither a pad_token_id nor an eos_token_id")
    layers = check_layers(model, layers)
    encoder = RowEncoder(tok, max_len, chat_template)
    cb_rows, cb_stats = encode_rows(encoder, as_rows(harmful, "harmful"), "harmful")
    rt_rows, rt_stats = encode_rows(encoder, as_rows(retain, "retain"), "retain")

    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    peft_model, trainable = add_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                                     target_modules=target_modules,
                                     layers_to_transform=lora_layers)
    decoder, to_logits = decoder_and_head(peft_model)
    device = next(p for p in peft_model.parameters()).device
    dtype = next(p for p in peft_model.get_base_model().parameters()).dtype
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if use_amp else contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    opt = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    warm = max(1, int(warmup_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    def forward(batch, with_logits: bool):
        out = decoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      output_hidden_states=True)
        hidden = {l: out.hidden_states[l + 1] for l in layers}
        logits = None
        if with_logits:  # vocabulary logits only where the KL is taken, never on padding
            keep = batch[logits_mask].bool()
            if not keep.any():
                keep = batch["attention_mask"].bool()
            logits = to_logits(out.last_hidden_state[keep])
        return hidden, logits

    cb_iter = batches(cb_rows, batch_size, pad_id, seed)
    rt_iter = batches(rt_rows, batch_size, pad_id, seed + 1)
    history: List[Dict[str, Any]] = []
    skipped = 0
    peft_model.train()
    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        acc: Dict[str, float] = {"loss": 0.0}
        for _ in range(gradient_accumulation):
            cb = {k: v.to(device) for k, v in next(cb_iter).items()}
            rt = {k: v.to(device) for k, v in next(rt_iter).items()}
            peft_model.eval()
            with torch.no_grad(), peft_model.disable_adapter(), amp():
                ref = {"cb": forward(cb, False), "rt": forward(rt, logits_mask is not None)}
            peft_model.train()
            with amp():
                lora = {"cb": forward(cb, False), "rt": forward(rt, logits_mask is not None)}
            loss, metrics = step_loss(step, ref, lora, {"cb": cb, "rt": rt})
            scaler.scale(loss / gradient_accumulation).backward()
            acc["loss"] += loss.item() / gradient_accumulation
            for k, v in metrics.items():
                acc[k] = acc.get(k, 0.0) + float(v) / gradient_accumulation
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).item()
        if not math.isfinite(grad_norm):
            skipped += 1
        scaler.step(opt)  # with fp16, GradScaler skips a step whose gradients overflowed
        scaler.update()
        sched.step()
        entry = {"step": step + 1, "grad_norm": grad_norm, "lr": sched.get_last_lr()[0], **acc}
        history.append(entry)
        if callback is not None:
            callback(entry)
        elif log_every and (step + 1) % log_every == 0:
            print({k: round(v, 6) if isinstance(v, float) else v for k, v in entry.items()})
    peft_model.eval()
    stats = {"harmful": cb_stats, "retain": rt_stats, "nonfinite_grad_steps": skipped,
             "amp": str(dtype) if use_amp else None, "chat_template": encoder.chat_template}
    return peft_model, history, stats


def masked_mean_cos(a: Dict[int, torch.Tensor], b: Dict[int, torch.Tensor],
                    mask: torch.Tensor) -> float:
    """Mean cosine between ``a`` and ``b`` over masked tokens, averaged over layers."""
    m = mask.bool()
    vals = [torch.nn.functional.cosine_similarity(a[l].float(), b[l].float(), dim=-1)[m].mean()
            for l in a]
    return float(torch.stack(vals).mean())
