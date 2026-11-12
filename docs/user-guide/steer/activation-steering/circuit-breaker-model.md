# CircuitBreakerTrainer — representation rerouting

Training-time method built around a retain + rerouting (RR) objective. The rerouting loss pushes the model's hidden representations on harmful inputs away from their original directions, while the retain loss keeps benign representations close to the frozen original model.

**Loss:** $L = c_{\text{retain}} \cdot \|h_{\text{lora, retain}} - h_{\text{orig, retain}}\|_2 + c_{\text{cb}} \cdot \mathrm{ReLU}(\cos(h_{\text{lora, cb}}, h_{\text{orig, cb}}))$, with a linear schedule $c_{\text{retain}} = \alpha \cdot (\text{step}/T)$ and $c_{\text{cb}} = \alpha \cdot (1 - \text{step}/T)$.

> **Note:** This is a training-time method — it modifies weights. It appears under activation-steering because it targets hidden representations. `fit()` trains a LoRA ("LoRRA") adapter on the RR loss and returns the PEFT model. `calibrate()` only returns a `CircuitBreakerModel` wrapper that exposes the RR loss (`compute_rr_loss`) and coefficient schedule (`rr_coefficients`) for a custom loop; it does not train, ignores any data passed to it (with a warning), and installs no inference-time hook by default.

Ref: Zou et al., "Improving Alignment and Robustness with Circuit Breakers," NeurIPS 2024, arXiv:2406.04313.

## Signature

```python
CircuitBreakerTrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    target_layers: list[int] | None = None,  # None: paper's Llama-3-8B layers (10, 20) scaled to depth
    rr_alpha: float = 5.0,
    rr_schedule_steps: int = 300,
    model_id: str | None = None,     # load the model from here when `model` is None
    results_dir: str | None = None,  # default: the runner's results directory
    drift_task: str | None = None,
    **kwargs,                        # unknown keys are warned about, not used
)

trainer.fit(
    harmful,                 # list[str | {"prompt", "response"}]: circuit-breaker set
    retain,                  # list[str | {"prompt", "response"}]: retain set
    *,
    max_steps=150, batch_size=2, gradient_accumulation=8, learning_rate=1e-4,
    warmup_steps=10, max_grad_norm=1.0,
    lora_r=16, lora_alpha=16, lora_dropout=0.05,
    lora_layers=None,        # None: blocks 0..max(target_layers), as in the reference code
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    max_len=1024, chat_template=True, gradient_checkpointing=False,
    seed=0, log_every=10, callback=None, out_dir=None,
) -> PeftModel
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | `None` | Model to train (no PEFT adapter attached) |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |
| `target_layers` | `list[int]` | `None` | Decoder blocks the RR loss is measured on (block `l` = `hidden_states[l + 1]`) |
| `rr_alpha` | `float` | `5.0` | RR loss scale $\alpha$ |
| `rr_schedule_steps` | `int` | `300` | Schedule length $T$ |
| `model_id` | `str` | `None` | Hub id / path to load the model from when `model` is `None` |
| `results_dir` | `str` | `None` | Output directory; `None` is the runner default |
| `drift_task` | `str` | `None` | Optional drift-evaluation task |

`fit()` data and training:

- A harmful row's loss covers its response tokens, or its prompt tokens when it has no response. A retain row's covers every non-padding token. Rows are rendered with the tokenizer's chat template (`chat_template=False` concatenates prompt and response).
- Each optimizer step draws `batch_size` harmful and `batch_size` retain rows, `gradient_accumulation` times.
- The frozen reference is the same model with the adapter disabled, run in eval mode without gradients, so no second copy of the weights is loaded.
- LoRA weights train in fp32. On CUDA, a fp16/bf16 base runs under autocast; fp16 adds a GradScaler, which skips steps whose gradients overflow.
- `trainer.history` holds one record per step (`loss`, `grad_norm`, `lr`, `retain_coeff`, `cb_coeff`, `cb_cos`). `trainer.data_stats` counts rows, dropped rows and rows that reached `max_len`. `out_dir` saves the adapter and `rr_training.json`.

## Full example

```python
from safetune.runner import steer

harmful = [{"prompt": "How do I pick a lock?", "response": "Insert a tension wrench..."}]
retain = [{"prompt": "How do I pick a lock?", "response": "I can't help with that."},
          {"prompt": "What is 2+2?", "response": "4"}]

trainer = steer.CircuitBreakerTrainer(model, tokenizer, target_layers=[10, 20])
peft_model = trainer.fit(harmful, retain, max_steps=150, out_dir="cb-adapter")
print(trainer.history[-1])
```

## Known behaviour: an inert start

With PEFT's default initialisation (`lora_B = 0`) the adapted model starts identical to the reference, so the circuit-breaker cosine is exactly 1. That is the maximum of $\mathrm{ReLU}(\cos)$, where its gradient is zero. Near that point the rerouting gradient is tiny while the retain term pulls the representations back. `fit()` warns when the logged cosine never drops below $1 - 10^{-4}$. Check `[h["cb_cos"] for h in trainer.history]` before relying on the adapter.

## When to use

- **Best for:** weight-level rerouting when you want circuit-breaker safety baked into the model itself rather than an inference-time hook.
- **Not for:** situations where you need a reversible inference-time intervention — use `RefusalDirectionTrainer` or `CAATrainer` instead.
- **Trade-offs:** requires a training run and paired harmful / retain data.

## Citation

```bibtex
@article{circuitbreaker2024,
  title  = {Improving Alignment and Robustness with Circuit Breakers},
  author = {Zou, Andy and others},
  year   = {2024},
  note   = {NeurIPS 2024, arXiv:2406.04313},
}
```
