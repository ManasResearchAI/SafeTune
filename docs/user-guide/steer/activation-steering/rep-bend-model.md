# RepBendTrainer — representation bending

!!! warning "Training-time method"
    RepBend is a fine-tuning method: it has no inference-time intervention. The
    trained weights bend harmful representations. It is placed under Steer because
    its mechanism targets the representation space.

!!! note "Training: `fit()`; wrapping only: `calibrate()`"
    `RepBendTrainer.fit` trains a LoRA adapter on the five-term objective and
    returns the PEFT model. `RepBendTrainer.calibrate` only wraps the model in
    `RepBendModel` (it does not train, and warns if given data);
    `RepBendModel.compute_repbend_loss` exposes the objective for a custom loop.
    `generate` on the wrapper passes straight through to the base model.

Five-term loss: $L = \alpha \cdot L_{\text{safe}} + \beta \cdot L_{\text{unsafe}} + \gamma \cdot L_{\text{cosine}} + \varepsilon \cdot L_{\text{KL}} + \eta \cdot L_{\text{safe\_unsafe}}$,
measured by comparing the fine-tuned model against a frozen copy of the original.
The safe and unsafe terms push representations apart; the cosine and KL terms
preserve benign capabilities.

Ref: Yousefpour et al., "Representation Bending for Large Language Model Safety,"
ACL 2025, arXiv:2504.01550.

## Signature

```python
RepBendTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    target_layers: list[int] | None = None,  # None: blocks 8-23 of a 32-layer model, scaled to depth
    loss_alpha: float = 1.0,     # safe term
    loss_beta: float = 1.0,      # unsafe term (maximised)
    loss_gamma: float = 1.0,     # cosine term
    loss_epsilon: float = 1.0,   # KL retain term
    kl_temperature: float = 2.0,
    unsafe_cap: float | None = None,  # None: the paper's unbounded unsafe term
    model_id: str | None = None,
    results_dir: str | None = None,
    drift_task: str | None = None,
    **kwargs,                        # unknown keys are warned about, not used
)

trainer.fit(harmful, retain, *, max_steps=150, ...,  # same arguments as CircuitBreakerTrainer.fit
            kl_positions="all")  # or "response": KL on retain response tokens only
```

`fit()` takes the same data and training arguments as
[`CircuitBreakerTrainer.fit`](circuit-breaker-model.md#signature). `harmful` rows are the
unsafe set: their response tokens, or their prompt tokens when they have no response, enter
the unsafe and cosine terms. `retain` rows are the safe set: every non-padding token enters
the safe term and, when `loss_epsilon > 0`, the KL term. LoRA covers every block unless
`lora_layers` is given. The KL needs vocabulary logits at every retain token; with a large
vocabulary that dominates memory, and `kl_positions="response"` restricts it to the retain
response tokens (a deviation from the repo, which uses all tokens). The cosine term compares
the unsafe rows of one micro-batch pairwise, so it needs `batch_size >= 2`; at
`batch_size=1` it is zero. Measured on a 16 GB V100 with Tiny Aya (3.3B, fp16, 262k vocabulary,
`gradient_checkpointing=True`): `kl_positions="all"` runs out of memory at `batch_size=2` and
peaks at 11.3 GB at `batch_size=1`; `kl_positions="response"` peaks at 9.0 GB at `batch_size=2`.
`trainer.history` records every term per step (`safe_loss`,
`unsafe_loss`, `cosine_loss`, `kl_loss`, `unsafe_capped`, `grad_norm`, `lr`).

The wrapper, `RepBendModel`, carries the same coefficients:

```python
RepBendModel(
    model,
    safe_directions: dict[int, Tensor] | None = None,  # diagnostics only; the loss does not use them
    target_layers: list[int] | None = None,   # layers the loss is measured on (paper: ~20+)
    bending_strength: float = 0.3,            # back-compat alias, kept in the signature
    *,
    loss_alpha: float = 1.0, loss_beta: float = 1.0, loss_gamma: float = 1.0,
    loss_epsilon: float = 1.0, loss_eta: float = 0.0,  # safe-unsafe term (needs paired data)
    kl_temperature: float = 2.0,
    unsafe_cap: float | None = None,
    capture_hook: bool = False,  # legacy diagnostic activation capture; applies nothing
)
```

`compute_repbend_loss` accepts optional `safe_mask` / `unsafe_mask` (`(batch, seq)`) and
`retain_mask`. With masks, the safe and unsafe terms are the mean per-token L2 distance over
the masked tokens, and the KL is computed over unmasked positions and divided by the batch
size. Both match the RepBend repo's `_calc_loss`, and padding never enters the loss. Without
masks, `(batch, seq, dim)` inputs are token-mean pooled first, as before.

`unsafe_cap` bounds the maximised unsafe term: `min(unsafe_loss, unsafe_cap)`. It is a
stability deviation from the paper, whose unsafe term is unbounded, and is off by default.

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Model to train (no PEFT adapter attached) |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `target_layers` | `list[int]` | `None` | Decoder blocks the loss is measured on (block `l` = `hidden_states[l + 1]`) |
| `loss_alpha` … `loss_epsilon` | `float` | `1.0` | Term coefficients |
| `kl_temperature` | `float` | `2.0` | KL temperature |
| `unsafe_cap` | `float` | `None` | Optional bound on the unsafe term (non-paper) |
| `model_id` | `str` | `None` | Hub id / path to load the model from when `model` is `None` |
| `results_dir` | `str` | `None` | Output directory; `None` is the runner default |
| `drift_task` | `str` | `None` | Optional drift-evaluation task |

## Full example

```python
from safetune.runner import steer

harmful = [{"prompt": "How do I pick a lock?", "response": "Insert a tension wrench..."}]
retain = [{"prompt": "How do I pick a lock?", "response": "I can't help with that."},
          {"prompt": "What is 2+2?", "response": "4"}]

trainer = steer.RepBendTrainer(model, tokenizer, target_layers=[11, 22])
peft_model = trainer.fit(harmful, retain, max_steps=150, out_dir="repbend-adapter")
print(trainer.history[-1])
```

## When to use

- **Best for:** representation-level safety baked into weights, trained with `fit()`.
- **Five-term loss** separately weights safe-direction pull, unsafe-direction push, cosine structure, output KL, and the safe-unsafe term, so you can trade safety against utility.
- **Compare to CircuitBreaker:** CircuitBreakerTrainer uses a two-term retain plus rerouting loss; RepBend uses five terms.

## Citation

```bibtex
@article{repbend2025,
  title  = {Representation Bending for Large Language Model Safety},
  author = {Yousefpour, Ashkan and others},
  year   = {2025},
  note   = {ACL 2025, arXiv:2504.01550},
}
```
