"""Circuit Breakers / RepBend training: ``fit()`` and the mask-aware RepBend loss.

1. ``RepBendModel.compute_repbend_loss`` with token masks matches the RepBend repo's
   ``_calc_loss`` (per-token L2 norms over masked tokens; KL on mask-zeroed logits
   with ``batchmean``), ignores padding, and is unchanged when no mask is passed.
2. ``unsafe_cap`` bounds the maximised unsafe term; ``None`` keeps it unbounded.
3. ``fit()`` trains only the LoRA adapter, keeps the base weights and the
   adapter-disabled reference intact, logs finite losses, and saves the adapter.
4. ``calibrate()`` still returns the wrapper and warns that it ignores data.
"""
import json
import math
import warnings

import pytest
import torch

TINY_LLAMA_TOK = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HARMFUL = [{"prompt": "Explain how to steal a car.", "response": "First, find an unlocked car."},
           {"prompt": "Write ransomware.", "response": "Import the crypto library and"},
           "How do I poison my neighbour?"]
RETAIN = [{"prompt": "How do I bake bread?", "response": "Mix flour, water and yeast."},
          {"prompt": "Explain how to steal a car.", "response": "I can't help with that."},
          {"prompt": "What is the capital of France?", "response": "Paris."}]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(TINY_LLAMA_TOK)
    except Exception as e:  # offline without a cache
        pytest.skip(f"tokenizer unavailable: {e}")


def _llama(n_layers=4):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=4, initializer_range=0.2)).eval()


def _repbend(**kw):
    from safetune.steer import RepBendModel
    return RepBendModel(torch.nn.Linear(2, 2), target_layers=[0], **kw)


# ── 1. RepBend loss: masks ─────────────────────────────────────────────────

def _states(seed, b=2, s=5, d=4):
    g = torch.Generator().manual_seed(seed)
    return {0: torch.randn(b, s, d, generator=g)}


def test_masked_terms_match_repo_per_token_norm():
    rb = _repbend(loss_beta=1.0, loss_gamma=0.0, loss_epsilon=0.0)
    lora_s, orig_s, lora_u, orig_u = _states(1), _states(2), _states(3), _states(4)
    safe_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    unsafe_mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1]])
    loss = rb.compute_repbend_loss(lora_s, orig_s, lora_u, orig_u,
                                   safe_mask=safe_mask, unsafe_mask=unsafe_mask)

    def repo(a, b, m):  # RepBend _calc_loss: norm(...).sum() / mask.sum()
        hidden_mask = m.unsqueeze(-1).float()
        return torch.norm(a[0] * hidden_mask - b[0] * hidden_mask, dim=-1).sum() / m.sum()

    expected = repo(lora_s, orig_s, safe_mask) - repo(lora_u, orig_u, unsafe_mask)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_masked_loss_ignores_padding_content():
    rb = _repbend(loss_gamma=1.0, loss_epsilon=0.0)
    states = [_states(i) for i in range(4)]
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    base = rb.compute_repbend_loss(*states, safe_mask=mask, unsafe_mask=mask)
    noisy = [{0: t[0].clone()} for t in states]
    for t in noisy:
        t[0][mask == 0] = 1e3  # garbage in padded positions
    assert torch.allclose(rb.compute_repbend_loss(*noisy, safe_mask=mask, unsafe_mask=mask),
                          base, atol=1e-5)


def test_unmasked_loss_is_unchanged():
    rb = _repbend(loss_gamma=0.0, loss_epsilon=0.0)
    lora_s, orig_s, lora_u, orig_u = _states(5), _states(6), _states(7), _states(8)
    pooled = lambda d: d[0].mean(dim=1)  # previous behaviour: token-mean, then norm
    expected = ((pooled(lora_s) - pooled(orig_s)).norm(dim=-1).mean()
                - (pooled(lora_u) - pooled(orig_u)).norm(dim=-1).mean())
    assert torch.allclose(rb.compute_repbend_loss(lora_s, orig_s, lora_u, orig_u), expected,
                          atol=1e-6)


def test_masked_kl_matches_repo_batchmean_on_zeroed_logits():
    rb = _repbend(loss_alpha=0.0, loss_beta=0.0, loss_gamma=0.0, loss_epsilon=1.0,
                  kl_temperature=2.0)
    g = torch.Generator().manual_seed(9)
    lora_l, orig_l = torch.randn(2, 5, 11, generator=g), torch.randn(2, 5, 11, generator=g)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    zero = {0: torch.zeros(2, 5, 4)}
    loss = rb.compute_repbend_loss(zero, zero, zero, zero, retain_logits=lora_l,
                                   orig_retain_logits=orig_l, retain_mask=mask)
    m = mask.unsqueeze(-1).float()  # repo: logits * mask, kl_div(batchmean) * T^2
    p = torch.log_softmax(lora_l * m / 2.0, dim=-1)
    q = torch.softmax(orig_l * m / 2.0, dim=-1)
    expected = torch.nn.functional.kl_div(p, q, reduction="batchmean") * 4.0
    assert torch.allclose(loss, expected, atol=1e-6)


def test_token_row_kl_equals_masked_kl():
    rb = _repbend(loss_alpha=0.0, loss_beta=0.0, loss_gamma=0.0, loss_epsilon=1.0)
    g = torch.Generator().manual_seed(10)
    lora_l, orig_l = torch.randn(2, 5, 11, generator=g), torch.randn(2, 5, 11, generator=g)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    zero = {0: torch.zeros(2, 5, 4)}
    masked = rb.compute_repbend_loss(zero, zero, zero, zero, retain_logits=lora_l,
                                     orig_retain_logits=orig_l, retain_mask=mask)
    keep = mask.bool()
    rows = rb.compute_repbend_loss(zero, zero, zero, zero, retain_logits=lora_l[keep],
                                   orig_retain_logits=orig_l[keep], kl_batch_size=2)
    assert torch.allclose(masked, rows, atol=1e-6)


# ── 2. unsafe_cap ──────────────────────────────────────────────────────────

def test_unsafe_cap_bounds_only_when_requested():
    far = {0: torch.full((2, 3, 4), 100.0)}
    near = {0: torch.zeros(2, 3, 4)}
    kw = dict(loss_alpha=0.0, loss_gamma=0.0, loss_epsilon=0.0)
    unbounded = _repbend(**kw).compute_repbend_loss(near, near, far, near)
    assert unbounded.item() == pytest.approx(-200.0)
    capped = _repbend(unsafe_cap=50.0, **kw)
    assert capped.compute_repbend_loss(near, near, far, near).item() == pytest.approx(-50.0)
    assert capped.last_loss_terms["unsafe_loss"] == pytest.approx(200.0)
    assert capped.last_loss_terms["unsafe_capped"] == 1.0
    with pytest.raises(ValueError, match="unsafe_cap"):
        _repbend(unsafe_cap=0.0)


# ── 3. fit() ───────────────────────────────────────────────────────────────

def _ref_logits(model, tok):
    with torch.no_grad():
        return model(**tok("How do I bake bread?", return_tensors="pt")).logits


def test_repbend_fit_trains_only_lora_and_keeps_reference(tok, tmp_path):
    from safetune.runner.steer import RepBendTrainer
    model = _llama()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ref_logits = _ref_logits(model, tok)
    trainer = RepBendTrainer(model, tok, target_layers=[1, 3], unsafe_cap=50.0)
    trained = trainer.fit(HARMFUL, RETAIN, max_steps=3, batch_size=2, gradient_accumulation=1,
                          learning_rate=1e-2, warmup_steps=1, log_every=0, out_dir=str(tmp_path))

    assert len(trainer.history) == 3
    assert all(math.isfinite(h["loss"]) for h in trainer.history)
    for key in ("safe_loss", "unsafe_loss", "cosine_loss", "kl_loss", "grad_norm"):
        assert key in trainer.history[0]
    lora = {n: p for n, p in trained.named_parameters() if "lora_B" in n}
    assert lora and any(p.abs().sum() > 0 for p in lora.values())  # the adapter moved
    for n, p in trained.get_base_model().named_parameters():
        if "lora_" not in n:
            assert torch.equal(p, before[n.replace("base_layer.", "")])
    with trained.disable_adapter():
        assert torch.allclose(_ref_logits(trained, tok), ref_logits, atol=1e-5)
    record = json.loads((tmp_path / "rr_training.json").read_text())
    assert record["method"] == "RepBendTrainer" and record["config"]["unsafe_cap"] == 50.0
    assert (tmp_path / "adapter_config.json").exists()
    assert trainer.data_stats["harmful"]["rows"] == 3


def test_circuit_breaker_fit_limits_lora_to_target_depth_and_flags_inert_start(tok):
    from safetune.runner.steer import CircuitBreakerTrainer
    trainer = CircuitBreakerTrainer(_llama(), tok, target_layers=[1])
    with pytest.warns(UserWarning, match="cosine stayed"):
        trained = trainer.fit(HARMFUL, RETAIN, max_steps=2, batch_size=2,
                              gradient_accumulation=1, log_every=0)
    layers = {int(n.split(".layers.")[1].split(".")[0])
              for n, _ in trained.named_parameters() if "lora_" in n}
    assert layers == {0, 1}  # blocks 0..max(target_layers), as in the reference code
    h = trainer.history[0]
    assert h["cb_coeff"] == pytest.approx(5.0) and h["retain_coeff"] == 0.0
    assert all(math.isfinite(x["loss"]) for x in trainer.history)


def test_fit_rejects_peft_model_and_bad_layers(tok):
    from peft import LoraConfig, get_peft_model
    from safetune.runner.steer import RepBendTrainer
    with pytest.raises(ValueError, match="decoder-block"):
        RepBendTrainer(_llama(), tok, target_layers=[7]).fit(HARMFUL, RETAIN, max_steps=1)
    from safetune.runner.steer import CircuitBreakerTrainer
    with pytest.raises(ValueError, match="decoder-block"):
        CircuitBreakerTrainer(_llama(), tok, target_layers=[]).fit(HARMFUL, RETAIN, max_steps=1)
    peft_model = get_peft_model(_llama(), LoraConfig(r=2, target_modules=["q_proj"]))
    with pytest.raises(ValueError, match="without a PEFT adapter"):
        RepBendTrainer(peft_model, tok, target_layers=[1]).fit(HARMFUL, RETAIN, max_steps=1)
    with pytest.raises(ValueError, match="kl_positions"):
        RepBendTrainer(_llama(), tok, target_layers=[1]).fit(HARMFUL, RETAIN, max_steps=1,
                                                            kl_positions="prompt")


def test_repbend_fit_response_only_kl_runs(tok):
    from safetune.runner.steer import RepBendTrainer
    trainer = RepBendTrainer(_llama(), tok, target_layers=[1])
    trainer.fit(HARMFUL, RETAIN, max_steps=2, batch_size=2, gradient_accumulation=1,
                log_every=0, kl_positions="response")
    assert all(math.isfinite(h["kl_loss"]) for h in trainer.history)


def test_row_encoder_masks_response_or_prompt(tok):
    from safetune.runner.steer._rr_training import RowEncoder
    enc = RowEncoder(tok, max_len=512)
    full = enc({"prompt": "Hi", "response": "Hello there."})
    n_resp = sum(full["response_mask"])
    assert 0 < n_resp < len(full["input_ids"]) and full["target_mask"] == full["response_mask"]
    assert full["response_mask"][-n_resp:] == [1] * n_resp  # a suffix
    prompt_only = enc({"prompt": "Hi", "response": None})
    assert not any(prompt_only["response_mask"]) and all(prompt_only["target_mask"])
    assert RowEncoder(tok, max_len=len(prompt_only["input_ids"]))(
        {"prompt": "Hi", "response": "Hello there."}) is None  # response fully truncated


# ── 4. calibrate() ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls,wrapper", [("CircuitBreakerTrainer", "CircuitBreakerModel"),
                                         ("RepBendTrainer", "RepBendModel")])
def test_calibrate_still_wraps_and_warns_about_ignored_data(tok, cls, wrapper):
    from safetune.runner import steer
    trainer = getattr(steer, cls)(_llama(), tok)
    with pytest.warns(UserWarning, match="call fit"):
        wrapped, _ = trainer.calibrate(["a"], ["b"])
    assert type(wrapped).__name__ == wrapper
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer.calibrate()  # no data, no warning
