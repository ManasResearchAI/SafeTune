---
repo: SafeTune
upstream: https://github.com/Lexsi-Labs/SafeTune
track: scored
pipeline_role: safety-repair
version: 0.1.8 (repo; PyPI last release 0.1.7)
commit: 26f59e6 (2026-10-21)
license: LSAL-1.2 (source-available)
python: ">=3.12"
install: pip install safetune
extras: [dev, docs, vision, viz, vllm, vllm-lens, interpret, unsloth, text-metrics]
hardware: CPU for quickstarts/steering/interpret; GPU for evaluate and full train+eval; ~24GB+ VRAM for full integration
gated_models: [allenai/wildguard, CohereLabs/tiny-aya-*, CohereLabs/aya-expanse-*]
api_keys: [HF_TOKEN]
tags:
  [safety-drift, harden, recover, unlearn, steering, refusal-direction, circuits,
   safety-neurons, benchmarks, red-team, same-model-repair, provenance]
evidence_level: docs+code inspected; no end-to-end run in this workspace
last_reviewed: 2026-10-30
---

# SafeTune — agent capability brief

> A library of audited safety interventions and instrumentation: Harden, Recover,
> Steer, Unlearn, Interpret, Evaluate. Scored track repo; the repair and
> before/after measurement stage of the safety pipeline.

## 1. Role in a safety pipeline

SafeTune is the safety-repair stage between data/training and evaluation. It is a
library, not a pipeline: each task has independent methods, and you pick one per
task. The three intervention classes act at different lifecycle points —
train-time (`harden`), weight-space after a checkpoint exists (`recover`,
`unlearn`), and inference-time (`steer`). Instrumentation (`interpret`,
`evaluate`) supports those interventions and stands alone. Its natural use in the
hackathon is: measure a base model, fine-tune or deliberately drift it, repair it,
then prove safety recovered without destroying utility.

## 2. Capability map

Status legend: `stable` = code + runnable example/test exists; `documented` =
docs claim it and code exists but no runnable example/test found; `experimental`
= repo flags it as partial; `absent` = not implemented. Audit badges are
SafeTune's own Faithful/Simplified/Variant/Wrong/Stub labels.

| Capability | Entry point | Status / badge | Evidence | Safety-pipeline use |
|---|---|---|---|---|
| Harden: SafeGrad | `safetune.runner.harden.SafeGradTrainer`; CLI `safetune train --algo safegrad` | stable / Faithful | `docs/reference/feature-map.md:82`; `src/safetune/runner/harden/_gradient_surgery.py` | Keep safety while fine-tuning on task data |
| Harden: Lisa | `harden.LisaTrainer`; CLI `--algo lisa` | stable / Faithful | `docs/reference/feature-map.md:88`; `src/safetune/runner/harden/_data_shaping.py` | Alternative train-time safety defence |
| Recover: RESTA | `recover.ReStaTrainer(...).apply()`; `apply_resta`; CLI `safetune patch --algo resta --base ...` | stable / Faithful | `docs/reference/feature-map.md:43`; `src/safetune/interventions/recover/resta.py` | Restore safety in a drifted checkpoint without training |
| Recover: C-ΔΘ | `recover.CThetaTrainer`; `apply_ctheta`; CLI `--algo ctheta` | stable / Faithful | `docs/reference/feature-map.md:38`; `src/safetune/interventions/recover/ctheta.py` | Weight-space repair after harmful fine-tuning |
| Unlearn: RMU / NPO | `unlearn.RMUTrainer`, `NPOTrainer`; CLI `safetune unlearn --algo rmu` | stable / Faithful; CLI prepares the trainer only | `docs/reference/feature-map.md:69-70` | Remove a hazardous capability while retaining benign behaviour |
| Steer: refusal direction / CAA | `steer.RefusalDirectionTrainer(model, tokenizer).calibrate(harmful=..., harmless=...)` | stable / Faithful; Python only | `docs/reference/feature-map.md:112`; `src/safetune/runner/steer/_activation.py` | Refuse harmful prompts at inference; no weight changes |
| Steer: decoding | `SafeDecoding`, `Contrastive` processors via `make_processor()` | documented: Faithful; HF path needs distinct expert/weak model | `docs/reference/feature-map.md:17-20,115-117` | Inference-time safety without retraining |
| Interpret: safety units | `safetune.interpret.identify_safety_neurons`, `safety_circuit_info`, `eap_safety_circuit` → `CircuitInfo` | documented: weight mode Faithful; activation mode Variant | `docs/reference/feature-map.md:170-171` | Locate where safety lives before steering/repair |
| Evaluate | `safetune.evaluate.evaluate(model, benchmarks=[...], judge=...)`; CLI `safetune eval --dataset harmbench` | stable / Faithful | `docs/reference/feature-map.md:148`; `src/safetune/instrumentation/evaluate/pipeline.py` | Produce before/after safety numbers on >1 metric |
| Red-team probes | `AbliterationAttack`, `BoNAttack` under evaluate | documented: Faithful | `docs/reference/feature-map.md:136-137` | Stress-test refusal before and after repair |
| CircuitKIT adapter | `safetune.core.circuit_kit.adapter` → `CircuitInfo` | stable: documented adapter | `src/safetune/core/circuit_kit/adapter.py:23-40` | Aim repair/steering at circuits found by CircuitKIT |

## 3. Minimum viable usage

Repair a drifted checkpoint without training (RESTA needs the drifted, base, and
aligned models; keep the extra copies on CPU on a single GPU):

```python
from safetune.runner import recover

patched = recover.ReStaTrainer(
    drifted, base_model=base, aligned_model=aligned, alpha=0.25, device="cpu"
).apply()
```

Harden during fine-tuning:

```python
from safetune.runner import harden

train_ds, safety_ds = harden.load_harden_data(model_id, n=16)
trainer = harden.SafeGradTrainer(model_id=model_id, epochs=1, batch_size=4)
checkpoint = trainer.train(train_ds, safety_dataset=safety_ds)
```

Evaluate before/after:

```python
from safetune.evaluate import evaluate

base = evaluate(base_model, tokenizer=tok, benchmarks=["harmbench"], max_prompts=50)
tuned = evaluate(tuned_model, tokenizer=tok, benchmarks=["harmbench"], max_prompts=50)
```

## 4. Constraints and failure modes

- **Python >=3.12**, `torch>=2.7`, `transformers>=5.15,<6`, `trl>=0.12,<1.15`
  (the upper bound avoids a DPO CPU/Triton bug). Package is still Beta.
- **Cost profile:** quickstart ~30s CPU / ~1.5GB RAM; recover ~10s; harden or
  unlearn ~1min; interpret ~30s; evaluate ~1min. Full train+eval integration
  wants CUDA and ~24GB+ VRAM, 5–30min per run. There is no per-method VRAM table.
- **SafeGrad uses ~2x the memory of plain SFT.** Unlearn defaults to
  `upcast=True`; low-precision runs can silently do very little.
- **Tiny Aya/Aya gotchas:** Tiny Aya's chat template adds a ~366-token system
  preamble; leave `max_len` unset or use >=512 or loaders raise. On Tiny Aya use
  ReSta `alpha≈0.25`; `alpha=1` breaks the model. If automatic refusal-direction
  layer selection falls back to the middle layer or performs poorly, set
  `pick_layer` explicitly.
- **Evaluation needs a GPU and a judge.** Default judge is gated
  `allenai/wildguard` (~14.5GB); HarmBench/SorryBench/AILuminate use other 7–8B
  judges. `eval_strict=True` is the default; `False` records errors and
  continues. Gated resources raise `GatedResourceError`.
- **vLLM coverage is partial:** full support for CAA, STA, SafeSteer,
  RefusalDirection; partial for SCANS/AlphaSteer/CAST/CircuitBreakerRR; not
  supported for AdaSteer, SafeSwitch, LinearProbeGuard.
- **CLI gaps:** `safetune unlearn` only prepares a trainer; there is no steer or
  interpret CLI; the `tamper` benchmark category is empty; the old pipeline
  orchestration was removed.
- **Interop is file-based.** AlignTune/AgentTune checkpoints are consumed as
  standard HF folders; PEFT adapter loading is not explicit in `src/safetune/runner/utils/model_utils.py`.

## 5. Gaps and extension points

- **Audit-doc conflicts:** `docs/community/scope.md` lists `apply_safe_delta` as
  Faithful, but `docs/reference/feature-map.md` and `docs/reference/LIBRARY_CHECKLIST.csv` call it Simplified.
  Feature-map marks PR #18 methods Faithful while checklist rows are un-audited.
  Use `docs/reference/feature-map.md` + checklist as the source of truth.
- No `known-issues` page. Numeric audit validation still needs real drifted 8B
  checkpoints via `SAFETUNE_DRIFT_MODEL` / `tests/integration`.
- No per-method VRAM/runtime table; class-level guidance only.
- Variants (do not cite as the named paper method): `somf_merge`,
  `apply_deeprefusal`, `ASRTCallback`, `crisp_unlearn`, activation-mode interpret.
  No current Wrong/Stub on the live audited surface.
- Extension surfaces: add a trainer via `src/safetune/runner/<pillar>/` + one alias in
  `src/safetune/runner/_registry.py`; add a benchmark/judge via `BenchmarkSpec`/`REGISTRY`/
  `JUDGE_REGISTRY` and fill `tamper`; add data loaders/CuratorKIT schema support
  in `src/safetune/runner/utils/dataset/`; extend the CircuitKIT adapter; close CLI/backend
  gaps or build an audit harness.

## 6. Interop contract

- **CuratorKIT → SafeTune:** point `--train-dataset` at a CuratorKIT export folder
  with `--train-config sft_messages|sft_alpaca|dpo|...`; the loader tokenizes
  Alpaca, ShareGPT, messages, and DPO rows. Prompt-only rows raise. Parent
  provenance lands under `inputs[].provenance`.
- **CircuitKIT → SafeTune:** SafeTune reads `*_scores.json`, accepting old
  `node_scores` and 0.2 `safety_units`/`layer_suggestions`, and returns a
  `CircuitInfo`; unrecognized keys raise `ValueError`, missing optional deps
  return `None`.
- **SafeTune → AuditKIT:** AuditKIT imports SafeTune JSON read-only via
  `safetune_audit_event` (`tenant_id`, `decision`, `reason`, `timestamp`,
  optional prompt/response/metadata) and `safetune_bench_row`
  (`bench`/`prompt`, optional response/score/refused/judge). AuditKIT never
  imports SafeTune, and SafeTune does not push results to AuditKIT.
- **Outputs:** HF checkpoint + tokenizer + optional processor +
  `lexsi_provenance.json`; `results/summaries/results_<pillar>.json`; steering
  vectors `{layer_idx: tensor}` as `torch.save`; eval JSONL
  `<model>__base__<bench>_scored.jsonl` plus raw JSONL; active bundles.
- **Provenance:** `lexsi.provenance/1` with `library="safetune"`,
  `version`, `git_sha` (currently null), `created_at`, `base_model`, `method`,
  `inputs[]`, `params`; CuratorKIT/AlignTune lineage is nested.

## 7. Verification recipes

Run these from the repo root.

| Check | Command | Expected |
|---|---|---|
| Import/version | `python -c "import safetune; print(safetune.__version__)"` | `0.1.8`; seconds, CPU |
| Method catalog | `safetune list` | aliases grouped by pillar; seconds, no model |
| Steering quickstart | `python examples/quickstart/quickstart.py --device cpu` | refusal direction extracted and ablated on Qwen2.5-0.5B; ~30s CPU, ~1.5GB RAM |
| Recover quickstart | `python examples/quickstart/recover_quickstart.py` | RESTA on synthetic checkpoints; ~10s CPU |

## 8. Sources

Read for this brief: `README.md`, `pyproject.toml`, `CHANGELOG.md`,
`docs/reference/feature-map.md`, `docs/reference/LIBRARY_CHECKLIST.csv`,
`docs/community/scope.md`, `docs/getting-started/taxonomy.md`,
`docs/getting-started/installation.md`, `docs/user-guide/`, `docs/reference/`
(CLI, api-contract, hyperparameters), `docs/community/dev-runbook.md`,
`src/safetune/` (runner registry, provenance, model_utils, evaluate pipeline,
CircuitKIT adapter, vLLM backend), `tests/runtime/test_interop.py`, `examples/`.
