"""
Common model utilities for setting up tokenizers and models.
"""
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Config,
    LlamaForCausalLM,
    LlamaConfig,
    Qwen3ForCausalLM,
    Qwen3Config,
    RwkvForCausalLM,
    RwkvConfig,
)
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
import re
import torch
import os

# Import custom model classes
from utils.models import (
    GPT2ModelWithLayerTargets,
    LlamaModelWithLayerTargets,
    PythiaModelWithLayerTargets,
    Qwen3ModelWithLayerTargets,
    RwkvModelWithLayerTargets,
)


_FAMILY_REGISTRY = [
    {
        "family": "gpt2",
        "name_re": re.compile(r"^(distilgpt2|gpt2(-medium|-large|-xl)?)$", re.IGNORECASE),
        "config_class": GPT2Config,
        "model_class": GPT2LMHeadModel,
        "custom_class": GPT2ModelWithLayerTargets,
        "inner_model_path": "transformer",
        "layers_path":      "transformer.h",
        "embeddings_path":  "transformer.wte",
        "lm_head_path":     "lm_head",
        "n_layers_attr":    "n_layer",
        "submodules": {"ln1": "ln_1", "attn": "attn",
                       "ln2": "ln_2", "mlp": "mlp"},
    },
    {
        "family": "pythia",
        "name_re": re.compile(r"^EleutherAI/pythia-", re.IGNORECASE),
        "config_class": AutoConfig,
        "model_class": GPTNeoXForCausalLM,
        "custom_class": PythiaModelWithLayerTargets,
        "inner_model_path": "gpt_neox",
        "layers_path":      "gpt_neox.layers",
        "embeddings_path":  "gpt_neox.embed_in",
        "lm_head_path":     "embed_out",
        "n_layers_attr":    "num_hidden_layers",
        "submodules": {"ln1": "input_layernorm", "attn": "attention",
                       "ln2": "post_attention_layernorm", "mlp": "mlp"},
    },
    {
        "family": "qwen3",
        "name_re": re.compile(r"^Qwen/Qwen3-", re.IGNORECASE),
        "config_class": Qwen3Config,
        "model_class": Qwen3ForCausalLM,
        "custom_class": Qwen3ModelWithLayerTargets,
        "inner_model_path": "model",
        "layers_path":      "model.layers",
        "embeddings_path":  "model.embed_tokens",
        "lm_head_path":     "lm_head",
        "n_layers_attr":    "num_hidden_layers",
        "submodules": {"ln1": "input_layernorm", "attn": "self_attn",
                       "ln2": "post_attention_layernorm", "mlp": "mlp"},
    },
    {
        "family": "llama",
        "name_re": re.compile(r"^(meta-llama/|llama-)", re.IGNORECASE),
        "config_class": LlamaConfig,
        "model_class": LlamaForCausalLM,
        "custom_class": LlamaModelWithLayerTargets,
        "inner_model_path": "model",
        "layers_path":      "model.layers",
        "embeddings_path":  "model.embed_tokens",
        "lm_head_path":     "lm_head",
        "n_layers_attr":    "num_hidden_layers",
        "submodules": {"ln1": "input_layernorm", "attn": "self_attn",
                       "ln2": "post_attention_layernorm", "mlp": "mlp"},
    },
    {
        "family": "rwkv",
        "name_re": re.compile(r"^RWKV/rwkv-4-", re.IGNORECASE),
        "config_class": RwkvConfig,
        "model_class": RwkvForCausalLM,
        "custom_class": RwkvModelWithLayerTargets,
        "inner_model_path": "rwkv",
        "layers_path":      "rwkv.blocks",
        "embeddings_path":  "rwkv.embeddings",
        "lm_head_path":     "head",
        "n_layers_attr":    "num_hidden_layers",
        "submodules": {"ln1": "ln1", "attn": "attention",
                       "ln2": "ln2", "mlp": "feed_forward"},
    },
]


def _resolve_family(model_name: str) -> dict:
    """Look up the family spec for a given HF model name. Raises ValueError if no family matches."""
    for spec in _FAMILY_REGISTRY:
        if spec["name_re"].search(model_name):
            return spec
    families = ", ".join(s["family"] for s in _FAMILY_REGISTRY)
    raise ValueError(
        f"No model family matches {model_name!r}. "
        f"Supported families: {families}. "
        f"To add a new family, append an entry to _FAMILY_REGISTRY in utils/model_utils.py."
    )


def get_family_by_type(model_type: str) -> dict:
    """Look up registry entry by short family name (matches --model_type)."""
    for spec in _FAMILY_REGISTRY:
        if spec["family"] == model_type:
            return spec
    families = ", ".join(s["family"] for s in _FAMILY_REGISTRY)
    raise ValueError(
        f"Unknown model_type {model_type!r}. Supported: {families}. "
        f"Add an entry to _FAMILY_REGISTRY in utils/model_utils.py to support a new family."
    )


def setup_tokenizer(model_name, state_tokens, action_tokens):
    """
    Set up the tokenizer based on the model type.
    
    Args:
        model_name: Name of the model to load tokenizer for
        state_tokens: Dictionary of state -> state tokens
        action_tokens: Dictionary of action -> action tokens
        
    Returns:
        tokenizer: The tokenizer
    """
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for the task
    tokenizer.add_tokens(list(action_tokens.values()) + [f" {action}" for action in action_tokens.values()])
    tokenizer.add_tokens(list(state_tokens.values()) + [f" {state}" for state in state_tokens.values()])
    
    return tokenizer


def setup_model(tokenizer, model_name=None, checkpoint_path=None, use_bfloat16=False, 
                no_pretrain=False, output_dir=None, use_custom_models=False,
                layerwise_supervision_config=None):
    """
    Set up the model based on the model type and arguments.
    
    Args:
        model_name: Name of the model to load
        tokenizer: The tokenizer
        checkpoint_path: Path to checkpoint to load from
        use_bfloat16: Whether to use bfloat16 precision
        no_pretrain: Whether to initialize a new model with random weights
        output_dir: Output directory for checkpoints (for training)
        use_custom_models: Whether to use custom model classes
        layerwise_supervision_config: Configuration for layerwise supervision
        
    Returns:
        model: The language model
    """
    print("Loading model")
    
    if model_name:
        spec = _resolve_family(model_name)
        config_class = spec["config_class"]
        model_class = spec["custom_class"] if use_custom_models else spec["model_class"]
    else:
        model_class = AutoModelForCausalLM
        config_class = AutoConfig
    
    # Check for checkpoints in output directory (for training)
    if not checkpoint_path and output_dir and os.path.exists(output_dir):
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if len(subdirs) > 0:
            # Filter for subdirectories that match the expected pattern "checkpoint-NUMBER"
            valid_subdirs = []
            for subdir in subdirs:
                parts = subdir.split("-")
                if len(parts) >= 2 and parts[0] == "checkpoint" and parts[1].isdigit():
                    valid_subdirs.append(subdir)
            
            if valid_subdirs:
                min_subdir = min(valid_subdirs, key=lambda x: int(x.split("-")[1]))
                checkpoint_path = os.path.join(output_dir, min_subdir)
            else:
                print(f"Warning: Found subdirectories in {output_dir}, but none match the expected 'checkpoint-NUMBER' format.")
    
    # Load or create model
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        if use_custom_models:
            model = model_class.from_pretrained(
                checkpoint_path,
                device_map="auto",
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                layerwise_supervision_config=layerwise_supervision_config,
            )
        else:
            model = model_class.from_pretrained(checkpoint_path)
    elif no_pretrain:
        # Initialize a new model with random weights
        print("Initializing model with random weights")
        config = config_class.from_pretrained(model_name)
        if use_custom_models:
            config.torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
            model = model_class(config, layerwise_supervision_config=layerwise_supervision_config)
        else:
            model = model_class(config)
    else:
        # Load pre-trained model
        if use_custom_models:
            model = model_class.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                layerwise_supervision_config=layerwise_supervision_config,
            )
        else:
            model = model_class.from_pretrained(model_name)
    
    # Resize token embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Set up precision
    if use_bfloat16:
        model = model.to(torch.bfloat16)
    
    # Enable gradient checkpointing for multi-GPU training (for training)
    if use_custom_models and torch.cuda.device_count() > 1:
        model.gradient_checkpointing_enable()
        model = torch.nn.DataParallel(model)
        model.module.gradient_checkpointing_enable()
    
    model = model.to("cuda")
    return model 