# CLAUDE.md — State Tracking Interpretability Research

## Project Overview

This repository implements the paper ["(How) Do Language Models Track State?"](https://arxiv.org/abs/2503.02854) (Li, Guo, Andreas 2025). It trains language models on synthetic permutation tasks and uses mechanistic interpretability techniques — linear probing and activation patching — to determine which internal algorithm a trained model uses to track state.

**Current branch `model-expansion-v3`** extends the original paper's scope (GPT-2, Pythia) to apply the same methodology to additional model families.

---

## Environment Setup

```bash
conda create -n lm_state_track python=3.12
conda activate lm_state_track
pip install -r requirements.txt
# Install PyTorch separately, matching your CUDA version:
# https://pytorch.org/get-started/locally/
```

**Key dependencies** (pinned in `requirements.txt`):
- `transformers==4.51.3` — model loading and tokenization (4.51+ required for Qwen3 classes)
- `nnsight==0.4.3` — activation tracing and patching
- `scikit-learn==1.6.1` — logistic regression for probing
- `accelerate==1.4.0` — distributed/mixed precision training
- `wandb==0.19.8` — experiment tracking
- `matplotlib==3.10.1`, `seaborn==0.13.2` — visualization

---

## Repository Layout

```
state-tracking/
├── permutation_task.py          # Task definition + data generation
├── make_topic_training_data.py  # NTP data via topic model
├── train.py                     # Training entry point
├── eval.py                      # Generalization evaluation
├── test_dataset.py              # Dataset sanity checks
├── paper_topic_mapping.json     # Pre-computed topic distributions for NTP
│
├── utils/
│   ├── model_utils.py           # Model + tokenizer setup, token expansion
│   ├── models.py                # Custom model classes with layer supervision
│   ├── data_loaders.py          # ChunkedDataset for large files
│   └── data_collators.py        # Batch collation per supervision type
│
├── interpret/
│   ├── main.py                  # Interpreter entry point (CLI)
│   ├── metadata_processors.py   # Extracts per-token state/parity labels
│   ├── visualization_manager.py # Plots probing and patching results
│   └── interpreters/
│       ├── base_interpreter.py              # Shared representation extraction
│       ├── probing_interpreter.py           # Linear probe + lengthwise probe
│       └── activation_patching_interpreter.py # Causal intervention
│
├── bash_scripts/
│   ├── train.sh                 # Wrapper: train.py
│   ├── train_NTP.sh             # Wrapper: NTP training
│   ├── eval.sh                  # Wrapper: eval.py
│   ├── run_probe.sh             # Wrapper: probing
│   ├── run_lengthwise_probe.sh  # Wrapper: lengthwise probing
│   └── run_intervention.sh      # Wrapper: activation patching
│
└── requirements.txt
```

**Generated directories** (gitignored):
- `checkpoints/` — model checkpoints from training
- `figures/gen_accuracy/`, `figures/probes/`, `figures/lengthwise_probe/`, `figures/intervene/` — output plots
- `probe_results/` — cached numpy probe scores
- `logs/` — training and analysis logs

---

## Core Concepts

### Permutation Task (`permutation_task.py`)

Models are trained on stories describing a sequence of actions applied to a permutation state. The task tests whether a model correctly tracks the current state of the permutation after each action.

- **S3**: 3-item permutations — 6 states, 6 actions (manageable for interpretability)
- **S5**: 5-item permutations — 120 states, 120 actions (harder, more realistic)
- **State**: current ordering of items (e.g., `[2, 0, 1]`)
- **Action**: a permutation composed onto the current state
- **Parity**: even/odd parity of the permutation (computed via inversion counting) — a coarser label used in probing and patching

### Supervision Types

| Type | Description |
|------|-------------|
| `direct_state` | Model predicts correct state token after each action (default) |
| `next_token` | Standard autoregressive next-token prediction on topic-model data |
| `direct_topic` | Topic-based supervision (experimental) |

---

## Full Workflow

### 1. Generate Data

```bash
# S3 data
python permutation_task.py --num_items 3 --data_dir S3_data --num_stories 100000

# S5 data
python permutation_task.py --num_items 5 --data_dir S5_data --num_stories 100000

# NTP topic-model data (to replicate paper)
python make_topic_training_data.py \
  --data_dir S3_NTP_data --num_topics 4 --num_stories 1000000 --num_items 3
```

### 2. Train

```bash
# Direct supervision (default)
bash bash_scripts/train.sh gpt2 3 S3_data/ checkpoints/gpt2_S3/

# NTP supervision
bash bash_scripts/train.sh EleutherAI/pythia-160M 3 S3_NTP_data/train \
  checkpoints/pythia_S3_NTP --supervision_type next_token

# Key optional flags
#   --no_pretrain                                  train from scratch
#   --use_bfloat16                                 mixed precision
#   --early_stopping                               stop on val loss plateau
#   --full_determinism                             reproducible runs
#   --seed 42
#   --lr_scheduler_type reduce_lr_on_plateau       decay LR on eval_loss plateau
#                                                  (forces step-cadence eval; combine with
#                                                   --early_stopping to also load best model)
#   --lr_scheduler_patience 2                      evals to wait before decaying
#   --lr_scheduler_factor 0.5                      LR multiplier on each decay
```

Checkpoints land in `<output_dir>/checkpoint-<num_steps>/`.

### 3. Evaluate Generalization

```bash
bash bash_scripts/eval.sh checkpoints/gpt2_S3/ 3 S3_data/
```

Produces `figures/gen_accuracy/<checkpoint>.png` — accuracy vs. sequence length.

### 4. Analysis

```bash
# Linear probing (layer-wise accuracy)
bash bash_scripts/run_probe.sh \
  --model_type gpt2 --checkpoint_dir checkpoints/gpt2_S3/ --num_items 3

# Lengthwise probing (accuracy vs. layer vs. sequence length — heatmap)
bash bash_scripts/run_lengthwise_probe.sh \
  --model_type gpt2 --checkpoint_dir checkpoints/gpt2_S3/ --num_items 3

# Activation patching (causal intervention heatmap)
bash bash_scripts/run_intervention.sh \
  --model_type gpt2 --checkpoint_dir checkpoints/gpt2_S3/ --num_items 3 \
  --patching_mode substitution --intervene_output_type state_by_parity
```

---

## Adding a New Model (Model Expansion Goal)

Currently supported families: `gpt2`, `pythia`, `qwen3`, `llama`, `rwkv`. Adding another is one registry entry plus one boilerplate class — no more if/elif chains scattered across files. The architecture metadata lives in `_FAMILY_REGISTRY` in `utils/model_utils.py`, and both training and interpret pipelines look it up at runtime.

### 1. `utils/models.py` — Add a `WithLayerTargets` class

Mirror an existing family (the Llama, Qwen3, and GPT-2 classes are nearly byte-for-byte identical):

```python
class FooModelWithLayerTargets(ModelWithLayerTargetsMixin, FooForCausalLM):
    """Foo model with layer-wise supervision."""
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)
    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)
```

### 2. `utils/model_utils.py` — Append a `_FAMILY_REGISTRY` entry

One dict with 11 fields covers training dispatch, interpret-side architecture paths, and activation-patching submodule names:

```python
{
    "family": "foo",                                              # CLI --model_type value
    "name_re": re.compile(r"^foo-org/Foo-", re.IGNORECASE),       # HF name regex (training)
    "config_class": FooConfig,
    "model_class": FooForCausalLM,
    "custom_class": FooModelWithLayerTargets,
    "inner_model_path": "model",                                  # under nnsight LanguageModel
    "layers_path":      "model.layers",
    "embeddings_path":  "model.embed_tokens",
    "lm_head_path":     "lm_head",
    "n_layers_attr":    "num_hidden_layers",                      # attr on config
    "submodules": {"ln1": "input_layernorm", "attn": "self_attn",
                   "ln2": "post_attention_layernorm", "mlp": "mlp"},
}
```

The `family` string is what users pass via `--model_type`; the `name_re` is what `train.py` matches against `--model`. Both are strict: `_resolve_family` raises `ValueError` if no name regex matches, so silent mis-routing is impossible.

### 3. `train.py` and `interpret/main.py` — Add to `choices` lists

Both files validate at parse time. Add the new HF name(s) to `train.py`'s `--model choices=[]` and the new short string to `interpret/main.py`'s `--model_type choices=[]`. If you forget, argparse will reject the CLI value with a clear error.

### What you do NOT need to edit anymore

- `setup_model()` in `utils/model_utils.py` — registry-driven via `_resolve_family`.
- `setup_model()` in `interpret/interpreters/base_interpreter.py` — registry-driven via `get_family_by_type` + `attrgetter`.
- `get_hs_logits()` in `interpret/interpreters/activation_patching_interpreter.py` — registry-driven via the `submodules` field.
- Optimizer / LR scheduler wiring in `train.py` (including `--lr_scheduler_type reduce_lr_on_plateau`) — handled by HF Trainer at the optimizer level, not per-architecture.

If a future transformers release renames internal HF attributes (e.g., `model.layers` → `model.transformer.layers`), only the affected registry entry needs updating — every dispatch site reads from there.

---

## Implementation Notes

### ChunkedDataset (`utils/data_loaders.py`)
Streams large JSON data files in chunks to avoid loading everything into memory. Uses binary search for O(log n) index mapping across chunks. Frees memory when switching chunks.

### Token Expansion (`utils/model_utils.py`)
`setup_tokenizer()` adds custom tokens for every possible state and action in the permutation group (6 for S3, 120 for S5), plus space-prefixed variants. `setup_model()` then calls `model.resize_token_embeddings(len(tokenizer))` to grow the embedding matrix (HF handles tied input/output embeddings transparently — relevant for Qwen3-0.6B).

### Metadata Extraction (`interpret/metadata_processors.py`)
Converts raw token sequences into per-token DataFrames with columns for current state, current parity, action taken, etc. These labels are used as probe targets.

### Probe Caching (`interpret/interpreters/probing_interpreter.py`)
Lengthwise probe results are cached to `probe_results/` as numpy arrays. Subsequent runs load from cache if the file exists — delete cache files to force recomputation.

### nnsight Tracing
Both probing and activation patching use `nnsight` to extract hidden states. The architecture-specific layer access pattern (e.g., `model.transformer.h` vs `model.gpt_neox.layers` vs `model.model.layers`) is resolved from `_FAMILY_REGISTRY` via `operator.attrgetter` — no per-family branches in the interpreter code.

---

## Experiment Tracking

Training is logged to Weights & Biases automatically. Set `WANDB_PROJECT` or configure via `wandb login` before training. To disable: set `--report_to none` in training args.

---

## Paper Reference

```bibtex
@misc{li2025howlanguagemodelstrack,
  title={(How) Do Language Models Track State?},
  author={Belinda Z. Li and Zifan Carl Guo and Jacob Andreas},
  year={2025},
  eprint={2503.02854},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2503.02854},
}
```
