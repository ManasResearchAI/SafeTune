import os

import torch

from ._base import _SteerBase
from safetune._refusal_helpers import _encode_prompts, _get_decoder_layers
from safetune.config import get_config
import safetune.steer as S

# The absolute layer defaults below (CAA/CAST 14-18, LinearProbeGuard 15,
# SafeSwitch 16, AlphaSteer 10-19) were set for 32-layer models (Llama-3.1-8B).
_REF_DEPTH = 32


def _default_layers(model, layers) -> list:
    """The default band ``layers`` (contiguous indices on a 32-layer model) at the
    same relative depth in ``model``, scaling its end points like SPPFT's
    ``_middle_layer_indices``. Unchanged at 32 layers or with
    ``configure(legacy_steer_layers=True)``."""
    n = len(_get_decoder_layers(model))
    if not n or get_config().legacy_steer_layers:
        return list(layers)
    lo, hi = (min(n - 1, round(i * n / _REF_DEPTH)) for i in (layers[0], layers[-1]))
    return list(range(lo, hi + 1))


class RefusalDirectionTrainer(_SteerBase):
    METHOD = "RefusalDirectionTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 layers=None, alpha: float = 20.0,
                 orthogonalize: bool = False, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.layers = layers
        self.alpha = alpha
        self.orthogonalize = orthogonalize

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  alpha: float = None, **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        direction, _, _ = S.extract_refusal_direction(self.model, self.tok, harmful, harmless)
        strength = alpha if alpha is not None else self.alpha
        return S.RefusalDirectionModel(
            self.model,
            direction=direction,
            mode="steer",
            strength=strength,
            layers=self.layers,
        )


class CAATrainer(_SteerBase):
    METHOD = "CAATrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 target_layers=None, pool_method: str = "mean",
                 normalize: bool = True, multiplier: float = 20.0, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.target_layers = target_layers  # None -> 14-18 scaled to depth
        self.pool_method = pool_method
        self.normalize = normalize
        self.multiplier = multiplier

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        cfg = S.CAAConfig(
            target_layers=self.target_layers or _default_layers(self.model, range(14, 19)),
            pool_method=self.pool_method,
            normalize=self.normalize,
        )
        vectors = S.extract_caa_vectors(self.model, self.tok, harmful, harmless, cfg)
        return S.CAAModel(self.model, vectors=vectors, strength=self.multiplier)


class LinearProbeGuardTrainer(_SteerBase):
    METHOD = "LinearProbeGuardTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 layer: int = None, threshold: float = 0.5, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.layer = layer  # None -> 15 scaled to depth
        self.threshold = threshold

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        layer = self.layer if self.layer is not None else _default_layers(self.model, [15])[0]
        cfg = S.LinearProbeConfig(pick_layer=layer, threshold=self.threshold)
        probe = S.fit_linear_probe(self.model, self.tok, harmful, harmless, cfg)
        return S.LinearProbeGuardModel(self.model, self.tok, probe)


class SCANSTrainer(_SteerBase):
    METHOD = "SCANSTrainer"

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        wrapped = S.SCANSModel(self.model, tokenizer=self.tok)
        wrapped.fit(harmful, harmless)
        return wrapped


class STATrainer(_SteerBase):
    METHOD = "STATrainer"

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        return S.STAModel(self.model)


class CircuitBreakerRRTrainer(_SteerBase):
    METHOD = "CircuitBreakerRRTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 rr_layers=None, target_layers=None,
                 threshold: float = 0.5, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.rr_layers = rr_layers
        self.target_layers = target_layers
        self.threshold = threshold

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        cfg = S.CircuitBreakerRRConfig(threshold=self.threshold)
        _, _, directions = S.extract_refusal_direction(self.model, self.tok, harmful, harmless)
        cb = S.CircuitBreakerRRModel(self.model, directions=directions, config=cfg)
        cb.install()
        return cb


class CASTTrainer(_SteerBase):
    METHOD = "CASTTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 behavior_layers=None, condition_layers=None,
                 alpha: float = 1.0, chat_template: bool = None,
                 per_prompt_gate: bool = None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.behavior_layers = behavior_layers  # None -> 14-18 scaled to depth
        self.condition_layers = condition_layers
        self.alpha = alpha
        # None: True, or False under configure(legacy_cast_gate=True) (the old
        # gate: fit on raw prompts, first prompt of a batch decides for all).
        self.chat_template = chat_template
        self.per_prompt_gate = per_prompt_gate

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        caa_cfg = S.CAAConfig(target_layers=self.behavior_layers
                              or _default_layers(self.model, range(14, 19)))
        vectors = S.extract_caa_vectors(self.model, self.tok, harmful, harmless, caa_cfg)
        condition = S.fit_cast_condition(
            self.model, harmful, harmless, self.tok,
            candidate_layers=self.condition_layers,
            chat_template=self.chat_template,
        )
        return S.CASTModel(
            self.model,
            steering_vectors=vectors,
            condition=condition,
            alpha=self.alpha,
            per_prompt_gate=self.per_prompt_gate,
        )


class AdaSteerTrainer(_SteerBase):
    METHOD = "AdaSteerTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 alpha: float = 15.0, layers=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha
        self.layers = layers

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        import logging
        logger = logging.getLogger(__name__)

        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        _, _, directions = S.extract_refusal_direction(self.model, self.tok, harmful, harmless)

        # Pick RD/HD probe layers that actually exist in this model, keeping
        # AdaSteer's defaults (8 / 13) when available.
        layer_keys = sorted(directions)
        rd_probe = 8 if 8 in directions else layer_keys[len(layer_keys) // 3]
        hd_probe = 13 if 13 in directions else layer_keys[len(layer_keys) // 2]

        # Collect per-prompt last-token activations at the probe layers
        # BEFORE the AdaSteerModel installs its steering hooks, so the
        # calibration features are un-steered.
        from safetune.core.runtime.inference.vector_extraction import (
            SteeringVectorExtractor, VectorExtractionConfig,
        )
        ve_cfg = VectorExtractionConfig(
            target_layers=sorted({rd_probe, hd_probe}),
            pool_method="last_token",
        )
        extractor = SteeringVectorExtractor(self.model, self.tok, ve_cfg)
        harmful_acts = extractor._collect_activations(harmful, desc="adasteer harmful")
        benign_acts = extractor._collect_activations(harmless, desc="adasteer benign")

        # The trainer only has a harmful/harmless contrast set, not the
        # adversarial-vs-benign pseudo pair AdaSteer uses for a distinct
        # Harmfulness Direction; AdaSteerModel falls back to HD = copy of RD
        # (single-law mode) when harmfulness_direction is omitted.
        logger.warning(
            "AdaSteer: no adversarial/benign pseudo-pair set available on the "
            "trainer calibrate path; the Harmfulness Direction falls back to a "
            "copy of the Rejection Direction (single-law mode). The adaptive "
            "logistic laws are still fitted per input."
        )

        wrapped = S.AdaSteerModel(
            self.model,
            rejection_direction=directions,
            target_layers=self.layers,
            base_multiplier=self.alpha,
            rd_probe_layer=rd_probe,
            hd_probe_layer=hd_probe,
        )
        # Fit the logistic R-Law/H-Law so _coeff() gating is per-input
        # adaptive instead of the uniform-max fallback.
        wrapped.fit_adaptive(harmful_acts, benign_acts)
        return wrapped


class SafeSwitchTrainer(_SteerBase):
    METHOD = "SafeSwitchTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 gate_layer: int = None, threshold: float = 0.5,
                 chat_template: bool = True, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.gate_layer = gate_layer  # None -> 16 scaled to depth
        self.threshold = threshold
        self.chat_template = chat_template  # fit the prober on prompts formatted as generation sees them

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        from safetune.core.runtime.inference.safeswitch import SafetyProber
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        layer = self.gate_layer if self.gate_layer is not None else _default_layers(self.model, [16])[0]
        # Fit the stage-1 prober on the calibration prompts. Without this the
        # wrapper's prober was never trained, so it never fired on any model.
        dev = next(self.model.parameters()).device
        enc = lambda prompts: [_encode_prompts(self.tok, [p], self.chat_template,
                                               return_tensors="pt").to(dev) for p in prompts]
        # Vision-language configs (Aya Vision, North) keep hidden_size on the text config.
        hidden_size = self.model.config.get_text_config().hidden_size
        prober = SafetyProber(hidden_size=hidden_size, layer_idx=layer)
        prober.train(self.model, safe_inputs=enc(harmless), unsafe_inputs=enc(harmful))
        return S.SafeSwitchModel(
            self.model,
            prober=prober,
            probe_layer=layer,
            hidden_size=hidden_size,
            unsafe_threshold=self.threshold,
        )


def _warn_calibrate_ignores_data(trainer, harmful, harmless) -> None:
    if harmful is not None or harmless is not None:
        import warnings
        warnings.warn(
            f"{trainer.METHOD}.calibrate() only wraps the model and does not use the "
            "harmful/harmless data; call fit(harmful=..., retain=...) to train the LoRA "
            "adapter", stacklevel=4)


def _scaled_points(model, points) -> list:
    """Layer indices given for a 32-layer model, at the same relative depth in ``model``."""
    n = len(_get_decoder_layers(model))
    if not n or get_config().legacy_steer_layers:
        return list(points)
    return sorted({min(n - 1, round(p * n / _REF_DEPTH)) for p in points})


class _RRFitMixin:
    """``fit()`` plumbing shared by Circuit Breakers and RepBend."""

    def _fit(self, harmful, retain, *, layers, step_loss, logits_mask, lora_layers,
             out_dir, config, **loop):
        from ._rr_training import run_rr_training
        model, history, stats = run_rr_training(
            model=self.model, tokenizer=self.tok, harmful=harmful, retain=retain,
            layers=layers, step_loss=step_loss, logits_mask=logits_mask,
            lora_layers=lora_layers, **loop)
        self.history, self.data_stats = history, stats
        self.model = model
        if out_dir is not None:
            import json
            os.makedirs(out_dir, exist_ok=True)
            model.save_pretrained(out_dir)
            record = {"method": self.METHOD, "target_layers": list(layers),
                      "lora_layers": lora_layers, "config": config,
                      "fit": {k: v for k, v in loop.items() if k != "callback"},
                      "data": stats, "history": history}
            with open(os.path.join(out_dir, "rr_training.json"), "w") as handle:
                json.dump(record, handle, indent=2, default=str)
        return model


class CircuitBreakerTrainer(_RRFitMixin, _SteerBase):
    """Circuit Breakers (Zou et al. 2024, arXiv:2406.04313).

    ``calibrate()`` returns the :class:`~safetune.steer.CircuitBreakerModel` wrapper
    without training; ``fit()`` trains the LoRRA adapter on its RR loss.
    ``target_layers=None`` uses the paper's Llama-3-8B layers (10, 20) scaled to the
    model's depth; LoRA covers blocks ``0..max(target_layers)`` unless ``lora_layers``
    is given, as in ``GraySwanAI/circuit-breakers``.
    """

    METHOD = "CircuitBreakerTrainer"

    def __init__(self, model=None, tokenizer=None, *, target_layers=None,
                 rr_alpha: float = 5.0, rr_schedule_steps: int = 300, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.target_layers = target_layers
        self.rr_alpha = rr_alpha
        self.rr_schedule_steps = rr_schedule_steps

    def _wrapper(self, model, layers=None):
        return S.CircuitBreakerModel(model, target_layers=layers, rr_alpha=self.rr_alpha,
                                     rr_schedule_steps=self.rr_schedule_steps)

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        _warn_calibrate_ignores_data(self, harmful, harmless)
        return self._wrapper(self.model, self.target_layers)

    def fit(self, harmful, retain, *, max_steps: int = 150, batch_size: int = 2,
            gradient_accumulation: int = 8, learning_rate: float = 1e-4,
            warmup_steps: int = 10, max_grad_norm: float = 1.0, lora_r: int = 16,
            lora_alpha: int = 16, lora_dropout: float = 0.05, lora_layers=None,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), max_len: int = 1024,
            chat_template: bool = True, gradient_checkpointing: bool = False, seed: int = 0,
            log_every: int = 10, callback=None, out_dir: str = None):
        """Train a LoRRA adapter on the RR loss and return the PEFT model.

        ``harmful`` rows feed the circuit-breaker term and ``retain`` rows the retain
        term; each is a list of prompt strings or ``{"prompt", "response"}`` dicts. A
        harmful row's loss covers its response tokens (its prompt tokens when it has
        no response); a retain row's covers every token. The frozen reference is the
        same model with the adapter disabled, run in eval mode without gradients, so
        ``self.model`` must not already carry a PEFT adapter. Each step draws
        ``batch_size`` harmful and ``batch_size`` retain rows ``gradient_accumulation``
        times. LoRA weights train in fp32; on CUDA a fp16/bf16 base runs under
        autocast (fp16 with a GradScaler, which skips overflowed steps). Defaults
        follow ``GraySwanAI/circuit-breakers`` (150 steps, lr 1e-4, r = alpha = 16,
        schedule over 300 steps). The per-step log is kept in ``self.history`` and data
        statistics in ``self.data_stats``; ``out_dir`` saves the adapter and
        ``rr_training.json``. Warns when the circuit-breaker cosine never leaves 1.
        """
        from ._rr_training import check_layers, masked_mean_cos
        layers = check_layers(self.model, self.target_layers if self.target_layers is not None
                              else _scaled_points(self.model, [10, 20]))
        wrapper = self._wrapper(self.model, layers)
        if lora_layers is None:  # GraySwanAI/circuit-breakers: blocks 0..max(target_layers)
            lora_layers = list(range(max(layers) + 1))

        def step_loss(step, ref, lora, batch):
            loss = wrapper.compute_rr_loss(ref["rt"][0], lora["rt"][0], ref["cb"][0],
                                           lora["cb"][0], step=step,
                                           retain_mask=batch["rt"]["attention_mask"],
                                           cb_mask=batch["cb"]["target_mask"])
            retain_coeff, cb_coeff = wrapper.rr_coefficients(step)
            with torch.no_grad():
                cos = masked_mean_cos(lora["cb"][0], ref["cb"][0], batch["cb"]["target_mask"])
            return loss, {"retain_coeff": retain_coeff, "cb_coeff": cb_coeff, "cb_cos": cos}

        model = self._fit(
            harmful, retain, layers=layers, step_loss=step_loss, logits_mask=None,
            lora_layers=lora_layers, out_dir=out_dir,
            config={"rr_alpha": self.rr_alpha, "rr_schedule_steps": self.rr_schedule_steps},
            max_steps=max_steps, batch_size=batch_size,
            gradient_accumulation=gradient_accumulation, learning_rate=learning_rate,
            warmup_steps=warmup_steps, max_grad_norm=max_grad_norm, lora_r=lora_r,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules,
            max_len=max_len, chat_template=chat_template,
            gradient_checkpointing=gradient_checkpointing, seed=seed, log_every=log_every,
            callback=callback)
        cos = [h["cb_cos"] for h in self.history]
        if cos and min(cos) > 1 - 1e-4:
            import warnings
            warnings.warn(
                f"CircuitBreakerTrainer.fit: the circuit-breaker cosine stayed at "
                f">= {min(cos):.6f} for all {len(cos)} steps, so rerouting did not move the "
                "harmful representations. relu(cos) has zero gradient at cos = 1, which is "
                "where a zero-initialised LoRA adapter starts; check the adapter before "
                "relying on it", stacklevel=2)
        return model


class RepBendTrainer(_RRFitMixin, _SteerBase):
    """RepBend (Yousefpour et al. 2025, arXiv:2504.01550).

    ``calibrate()`` returns the :class:`~safetune.steer.RepBendModel` wrapper without
    training; ``fit()`` trains a LoRA adapter on its five-term loss with per-token
    masked norms and a padding-free KL (see ``RepBendModel.compute_repbend_loss``).
    ``target_layers=None`` uses the wrapper's default band (blocks 8-23 of a 32-layer
    model) scaled to the model's depth. ``unsafe_cap`` (default ``None``, the paper's
    unbounded objective) optionally bounds the maximised unsafe term.
    """

    METHOD = "RepBendTrainer"

    def __init__(self, model=None, tokenizer=None, *, target_layers=None,
                 loss_alpha: float = 1.0, loss_beta: float = 1.0, loss_gamma: float = 1.0,
                 loss_epsilon: float = 1.0, kl_temperature: float = 2.0,
                 unsafe_cap: float = None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.target_layers = target_layers
        self.loss_alpha, self.loss_beta, self.loss_gamma = loss_alpha, loss_beta, loss_gamma
        self.loss_epsilon, self.kl_temperature = loss_epsilon, kl_temperature
        self.unsafe_cap = unsafe_cap

    def _wrapper(self, model, layers=None):
        return S.RepBendModel(model, target_layers=layers, loss_alpha=self.loss_alpha,
                              loss_beta=self.loss_beta, loss_gamma=self.loss_gamma,
                              loss_epsilon=self.loss_epsilon,
                              kl_temperature=self.kl_temperature, unsafe_cap=self.unsafe_cap)

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        _warn_calibrate_ignores_data(self, harmful, harmless)
        return self._wrapper(self.model, self.target_layers)

    def fit(self, harmful, retain, *, max_steps: int = 150, batch_size: int = 2,
            gradient_accumulation: int = 8, learning_rate: float = 1e-4,
            warmup_steps: int = 10, max_grad_norm: float = 1.0, lora_r: int = 16,
            lora_alpha: int = 16, lora_dropout: float = 0.05, lora_layers=None,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), max_len: int = 1024,
            chat_template: bool = True, gradient_checkpointing: bool = False, seed: int = 0,
            log_every: int = 10, callback=None, out_dir: str = None,
            kl_positions: str = "all"):
        """Train a LoRA adapter on the RepBend loss and return the PEFT model.

        Same data, reference and arguments as :meth:`CircuitBreakerTrainer.fit`.
        ``harmful`` rows are the unsafe set (their response tokens, or prompt tokens
        when prompt-only, enter the unsafe and cosine terms); ``retain`` rows are the
        safe set (all tokens enter the safe term and, when ``loss_epsilon > 0``, the
        KL term). ``kl_positions="response"`` restricts the KL to retain response tokens,
        which saves memory with large vocabularies but deviates from the repo (all
        tokens). LoRA covers every block unless ``lora_layers`` is given. The per-term
        values of every step are kept in ``self.history``.
        """
        if kl_positions not in ("all", "response"):
            raise ValueError("kl_positions must be 'all' or 'response'")
        from ._rr_training import check_layers
        layers = check_layers(self.model, self.target_layers if self.target_layers is not None
                              else _default_layers(self.model, range(8, 24)))
        wrapper = self._wrapper(self.model, layers)
        logits_mask = None
        if self.loss_epsilon > 0:
            logits_mask = "attention_mask" if kl_positions == "all" else "response_mask"

        def step_loss(step, ref, lora, batch):
            rt_mask = batch["rt"]["attention_mask"]
            loss = wrapper.compute_repbend_loss(
                lora["rt"][0], ref["rt"][0], lora["cb"][0], ref["cb"][0],
                retain_logits=lora["rt"][1], orig_retain_logits=ref["rt"][1],
                safe_mask=rt_mask, unsafe_mask=batch["cb"]["target_mask"],
                kl_batch_size=rt_mask.shape[0])
            return loss, dict(wrapper.last_loss_terms)

        return self._fit(
            harmful, retain, layers=layers, step_loss=step_loss, logits_mask=logits_mask,
            lora_layers=lora_layers, out_dir=out_dir,
            config={"kl_positions": kl_positions, "loss_alpha": self.loss_alpha,
                    "loss_beta": self.loss_beta, "loss_gamma": self.loss_gamma,
                    "loss_epsilon": self.loss_epsilon,
                    "kl_temperature": self.kl_temperature, "unsafe_cap": self.unsafe_cap},
            max_steps=max_steps, batch_size=batch_size,
            gradient_accumulation=gradient_accumulation, learning_rate=learning_rate,
            warmup_steps=warmup_steps, max_grad_norm=max_grad_norm, lora_r=lora_r,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules,
            max_len=max_len, chat_template=chat_template,
            gradient_checkpointing=gradient_checkpointing, seed=seed, log_every=log_every,
            callback=callback)


class TARSteerTrainer(_SteerBase):
    METHOD = "TARSteerTrainer"

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        return S.TARModel(self.model)


class RRFAEnsembleTrainer(_SteerBase):
    METHOD = "RRFAEnsembleTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 member_models=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.member_models = member_models or []

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        return S.RRFAEnsemble(self.model)


class SafeSteerTrainer(_SteerBase):
    METHOD = "SafeSteerTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 alpha: float = 15.0, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        _, layer_id, directions = S.extract_refusal_direction(self.model, self.tok, harmful, harmless)
        category_vectors = {"default": directions[layer_id]}
        return S.SafeSteerModel(self.model, category_vectors, layer_id=layer_id, alpha=self.alpha)


class AlphaSteerTrainer(_SteerBase):
    METHOD = "AlphaSteerTrainer"

    def __init__(self, model=None, tokenizer=None, *,
                 alpha: float = 20.0, layers=None,
                 legacy_alphasteer_layers: bool = None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha
        self.layers = layers
        # True: hook the matrix fitted on layers[i] at decoder layer i (the old
        # mapping: fitted on 10-19, applied on 0-9). None: the configure() key.
        self.legacy_alphasteer_layers = legacy_alphasteer_layers

    def _do_calibrate(self, harmful=None, harmless=None, *, calib_n: int = 256,
                  **kwargs):
        if harmful is None or harmless is None:
            harmful, harmless = self._default_calib(calib_n)
        legacy = (self.legacy_alphasteer_layers if self.legacy_alphasteer_layers is not None
                  else get_config().legacy_alphasteer_layers)
        from safetune.core.runtime.inference.vector_extraction import (
            SteeringVectorExtractor, VectorExtractionConfig,
        )
        target = self.layers or _default_layers(self.model, range(10, 20))
        ve_cfg = VectorExtractionConfig(target_layers=target, pool_method="last_token")
        extractor = SteeringVectorExtractor(self.model, self.tok, ve_cfg)
        harmful_acts = extractor._collect_activations(harmful)
        harmless_acts = extractor._collect_activations(harmless)
        layer_keys = sorted(set(harmful_acts) & set(harmless_acts))
        if legacy:
            stack = lambda acts: torch.stack([acts[l].float() for l in layer_keys], dim=1)
            layers = list(range(len(layer_keys)))
        else:
            # AlphaSteerModel reads dim 1 as the decoder index, so put each
            # layer's activations at its own index (unused slots stay zero).
            def stack(acts):
                first = acts[layer_keys[0]]
                out = torch.zeros(first.shape[0], layer_keys[-1] + 1, first.shape[-1])
                for l in layer_keys:
                    out[:, l] = acts[l].float()
                return out
            layers = layer_keys
        return S.AlphaSteerModel(
            self.model,
            harmful_activations=stack(harmful_acts),
            benign_activations=stack(harmless_acts),
            layers=layers,
            strength=self.alpha,
        )
