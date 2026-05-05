"""
Common model utilities – tokenizer & model setup for all supported architectures.

Supported model_type strings
─────────────────────────────
  gpt2      – GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2)
  pythia    – EleutherAI Pythia (GPT-NeoX)
  llama     – LLaMA / LLaMA-2 / LLaMA-3
  bloom     – BigScience BLOOM
  opt       – Meta OPT
  falcon    – TII UAE Falcon
  mamba     – Mamba SSM (state-spaces/mamba-*)
  mistral   – Mistral AI Mistral / Mixtral
  phi       – Microsoft Phi-2 / Phi-3
  openelm   – Apple OpenELM (requires trust_remote_code)

The helpers infer the type from the model name string when *model_type* is
not provided explicitly, but callers that already know the type should pass
it in to avoid mis-detection.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomConfig,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    OPTConfig,
    OPTForCausalLM,
)
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

# Optional imports – wrapped so the repo still works if a package isn't installed
try:
    from transformers import FalconForCausalLM, FalconConfig
    _FALCON_AVAILABLE = True
except ImportError:
    FalconForCausalLM = None
    FalconConfig = None
    _FALCON_AVAILABLE = False

try:
    from transformers import MambaForCausalLM, MambaConfig
    _MAMBA_AVAILABLE = True
except ImportError:
    MambaForCausalLM = None
    MambaConfig = None
    _MAMBA_AVAILABLE = False

try:
    from transformers import PhiForCausalLM, PhiConfig
    _PHI_AVAILABLE = True
except ImportError:
    PhiForCausalLM = None
    PhiConfig = None
    _PHI_AVAILABLE = False

from utils.models import (
    GPT2ModelWithLayerTargets,
    LlamaModelWithLayerTargets,
    PythiaModelWithLayerTargets,
    BloomModelWithLayerTargets,
    OPTModelWithLayerTargets,
    MistralModelWithLayerTargets,
    FalconModelWithLayerTargets,
    MambaModelWithLayerTargets,
    PhiModelWithLayerTargets,
)

# LoRA target attention modules per architecture
_LORA_TARGET_MODULES: dict[str, list[str]] = {
    "gpt2":    ["c_attn"],
    "pythia":  ["query_key_value"],
    "llama":   ["q_proj", "v_proj"],
    "bloom":   ["query_key_value"],
    "opt":     ["q_proj", "v_proj"],
    "falcon":  ["query_key_value"],
    "mamba":   ["in_proj", "out_proj", "x_proj", "dt_proj"],
    "mistral": ["q_proj", "v_proj"],
    "phi":     ["q_proj", "v_proj"],
    "openelm": ["qkv_proj"],
}


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------
# Each entry maps a *model_type* key to:
#   base_class      – plain HF causal-LM class
#   custom_class    – our LayerTarget wrapper
#   config_class    – HF config class
#   keywords        – substrings in model-name that identify this architecture

_ARCH_REGISTRY: dict[str, dict] = {
    "gpt2": {
        "base_class": GPT2LMHeadModel,
        "custom_class": GPT2ModelWithLayerTargets,
        "config_class": GPT2Config,
        "keywords": ["gpt2", "distilgpt2"],
    },
    "pythia": {
        "base_class": GPTNeoXForCausalLM,
        "custom_class": PythiaModelWithLayerTargets,
        "config_class": AutoConfig,
        "keywords": ["pythia", "gpt-neox", "neox"],
    },
    "llama": {
        "base_class": LlamaForCausalLM,
        "custom_class": LlamaModelWithLayerTargets,
        "config_class": LlamaConfig,
        "keywords": ["llama", "llama2", "llama3", "vicuna", "alpaca", "mistral"],
    },
    "bloom": {
        "base_class": BloomForCausalLM,
        "custom_class": BloomModelWithLayerTargets,
        "config_class": BloomConfig,
        "keywords": ["bloom"],
    },
    "opt": {
        "base_class": OPTForCausalLM,
        "custom_class": OPTModelWithLayerTargets,
        "config_class": OPTConfig,
        "keywords": ["opt"],
    },
    "falcon": {
        "base_class": FalconForCausalLM,
        "custom_class": FalconModelWithLayerTargets,
        "config_class": FalconConfig,
        "keywords": ["falcon", "rw-"],
    },
    "mamba": {
        "base_class": MambaForCausalLM,
        "custom_class": MambaModelWithLayerTargets,
        "config_class": MambaConfig,
        "keywords": ["mamba"],
    },
    "mistral": {
        "base_class": MistralForCausalLM,
        "custom_class": MistralModelWithLayerTargets,
        "config_class": MistralConfig,
        "keywords": ["mistral", "mixtral"],
    },
    "phi": {
        "base_class": PhiForCausalLM,
        "custom_class": PhiModelWithLayerTargets,
        "config_class": PhiConfig,
        "keywords": ["phi"],
    },
    "openelm": {
        "base_class": AutoModelForCausalLM,
        "custom_class": None,
        "config_class": AutoConfig,
        "keywords": ["openelm"],
    },
}

# Remove entries whose optional dependencies aren't installed
for _key in ["falcon", "mamba", "phi"]:
    _avail = {"falcon": _FALCON_AVAILABLE, "mamba": _MAMBA_AVAILABLE, "phi": _PHI_AVAILABLE}
    if not _avail[_key]:
        _ARCH_REGISTRY[_key]["base_class"] = None
        _ARCH_REGISTRY[_key]["custom_class"] = None
        _ARCH_REGISTRY[_key]["config_class"] = None


def infer_model_type(model_name: str) -> str:
    """Guess model_type from *model_name* using the keyword table."""
    name_lower = model_name.lower()
    # gpt2 must come before llama so "mistral" in llama keywords doesn't win
    for mtype, entry in _ARCH_REGISTRY.items():
        for kw in entry["keywords"]:
            if kw in name_lower:
                return mtype
    return "auto"  # fallback – use AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def setup_tokenizer(model_name: str, state_tokens: dict, action_tokens: dict,
                    trust_remote_code: bool = False):
    """Load and configure a tokenizer, adding task-specific special tokens."""
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens(
        list(action_tokens.values())
        + [f" {a}" for a in action_tokens.values()]
    )
    tokenizer.add_tokens(
        list(state_tokens.values())
        + [f" {s}" for s in state_tokens.values()]
    )
    return tokenizer


def setup_model(
    tokenizer,
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    use_bfloat16: bool = False,
    no_pretrain: bool = False,
    output_dir: Optional[str] = None,
    use_custom_models: bool = False,
    layerwise_supervision_config=None,
    model_type: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """
    Load or initialise a causal-LM for any supported architecture.

    Resolution order for model_type:
      1. Explicit *model_type* argument
      2. Inferred from *model_name* via keyword matching
      3. 'auto' → AutoModelForCausalLM (best-effort)
    """
    print("Loading model")
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    # Resolve model_type
    if model_type is None and model_name:
        model_type = infer_model_type(model_name)
    elif model_type is None:
        model_type = "auto"

    entry = _ARCH_REGISTRY.get(model_type, {})
    base_class = entry.get("base_class") or AutoModelForCausalLM
    custom_class = entry.get("custom_class")
    config_class = entry.get("config_class") or AutoConfig

    # Check that optional deps are available
    if base_class is None:
        raise ImportError(
            f"Model type '{model_type}' requires an optional dependency that is "
            "not installed. Install it with `pip install transformers[{model_type}]` "
            "or see the model card for installation instructions."
        )

    model_class = custom_class if (use_custom_models and custom_class) else base_class

    # ── Find an existing checkpoint in output_dir (for training resumption) ──
    if not checkpoint_path and output_dir and os.path.exists(output_dir):
        subdirs = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]
        valid = [
            s for s in subdirs
            if s.startswith("checkpoint-") and s.split("-")[1].isdigit()
        ]
        if valid:
            checkpoint_path = os.path.join(
                output_dir, max(valid, key=lambda x: int(x.split("-")[1]))
            )

    # ── Load / create model ──────────────────────────────────────────────────
    common_from_pretrained_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        if use_custom_models and custom_class:
            model = model_class.from_pretrained(
                checkpoint_path,
                device_map="auto",
                layerwise_supervision_config=layerwise_supervision_config,
                **common_from_pretrained_kwargs,
            )
        else:
            model = model_class.from_pretrained(
                checkpoint_path, **common_from_pretrained_kwargs
            )

    elif no_pretrain:
        print("Initialising model with random weights")
        if config_class is AutoConfig:
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
        else:
            config = config_class.from_pretrained(model_name)

        if use_custom_models and custom_class:
            model = model_class(
                config,
                layerwise_supervision_config=layerwise_supervision_config,
            )
        else:
            model = model_class(config)

    else:
        print(f"Loading pre-trained model: {model_name}")
        if use_custom_models and custom_class:
            model = model_class.from_pretrained(
                model_name,
                device_map="auto",
                layerwise_supervision_config=layerwise_supervision_config,
                **common_from_pretrained_kwargs,
            )
        else:
            model = model_class.from_pretrained(
                model_name, **common_from_pretrained_kwargs
            )

    model.resize_token_embeddings(len(tokenizer))

    if use_bfloat16:
        model = model.to(torch.bfloat16)

    # Multi-GPU (training only)
    if use_custom_models and torch.cuda.device_count() > 1:
        model.gradient_checkpointing_enable()
        model = torch.nn.DataParallel(model)
        model.module.gradient_checkpointing_enable()

    model = model.to("cuda")
    return model


def apply_lora(
    model,
    model_name: str,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    model_type: Optional[str] = None,
):
    """Wrap *model* with LoRA adapters using the peft library.

    Only the LoRA parameters (and any custom layer_classifiers) are left
    trainable; all base-model weights are frozen.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    if model_type is None:
        model_type = infer_model_type(model_name)

    target_modules = _LORA_TARGET_MODULES.get(model_type, ["q_proj", "v_proj"])

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    # Re-enable gradients for newly added token embeddings (frozen by LoRA but
    # randomly initialised — the model can't learn task tokens without this)
    # and any custom classification heads from ModelWithLayerTargetsMixin.
    for name, param in model.named_parameters():
        if any(kw in name for kw in ("layer_classifiers", "embed", "lm_head")):
            param.requires_grad = True

    model.print_trainable_parameters()
    return model
