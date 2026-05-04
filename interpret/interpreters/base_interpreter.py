"""
Base interpreter – loads a model via nnsight and exposes a uniform API
for hidden-state extraction across all supported architectures.

Architecture → internal path mapping
────────────────────────────────────
  gpt2    : transformer.wte   / transformer.h[i]      / lm_head
  pythia  : gpt_neox.embed_in / gpt_neox.layers[i]    / embed_out
  llama   : model.embed_tokens / model.layers[i]       / lm_head
  bloom   : transformer.word_embeddings / transformer.h[i] / lm_head
  opt     : model.decoder.embed_tokens / model.decoder.layers[i] / lm_head
  falcon  : transformer.word_embeddings / transformer.h[i] / lm_head
  mamba   : backbone.embeddings / backbone.layers[i]   / lm_head
  mistral : model.embed_tokens / model.layers[i]       / lm_head
  phi     : model.embed_tokens / model.layers[i]       / lm_head
"""

from __future__ import annotations

import json
import os
import random
from typing import Set

import numpy as np
import torch
from nnsight import LanguageModel
from torch.nn import LogSoftmax
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer

from interpret.visualization_manager import VisualizationManager
from permutation_task import PermutationTask, compute_parity


# ---------------------------------------------------------------------------
# Per-architecture internal path descriptors
# ---------------------------------------------------------------------------
# Each key maps to a dict with:
#   embed_path   – dotted attribute path to the embedding module
#   layers_path  – dotted attribute path to the list of transformer layers
#   lm_head_path – dotted attribute path to the LM head
#   n_layers_attr – config attribute that holds the number of layers
#   layer_components – list of sub-paths within each layer to trace
#                      (used for activation-patching)

_TRANSFORMER_TUPLE_OUTPUTS = ["output", "attn.output", "self_attn.output", "attention.output"]

_ARCH_MAP: dict[str, dict] = {
    "gpt2": {
        "embed_path": "transformer.wte",
        "layers_path": "transformer.h",
        "lm_head_path": "lm_head",
        "n_layers_attr": "n_layer",
        "layer_components": ["output"],          # residual stream after block
        "attn_path": "attn",
        "mlp_path": "mlp",
        "ln1_path": "ln_1",
        "ln2_path": "ln_2",
        "tuple_layer_components": _TRANSFORMER_TUPLE_OUTPUTS,
    },
    "pythia": {
        "embed_path": "gpt_neox.embed_in",
        "layers_path": "gpt_neox.layers",
        "lm_head_path": "embed_out",
        "n_layers_attr": "num_hidden_layers",
        "layer_components": ["output"],
        "attn_path": "attention",
        "mlp_path": "mlp",
        "ln1_path": "input_layernorm",
        "ln2_path": "post_attention_layernorm",
        "tuple_layer_components": _TRANSFORMER_TUPLE_OUTPUTS,
    },
    "llama": {
        "embed_path": "model.embed_tokens",
        "layers_path": "model.layers",
        "lm_head_path": "lm_head",
        "n_layers_attr": "num_hidden_layers",
        "layer_components": ["output"],
        "attn_path": "self_attn",
        "mlp_path": "mlp",
        "ln1_path": "input_layernorm",
        "ln2_path": "post_attention_layernorm",
        "tuple_layer_components": _TRANSFORMER_TUPLE_OUTPUTS,
    },
    "bloom": {
        "embed_path": "transformer.word_embeddings",
        "layers_path": "transformer.h",
        "lm_head_path": "lm_head",
        "n_layers_attr": "num_hidden_layers",
        "layer_components": ["output"],
        "attn_path": "self_attention",
        "mlp_path": "mlp",
        "ln1_path": "input_layernorm",
        "ln2_path": "post_attention_layernorm",
        "tuple_layer_components": _TRANSFORMER_TUPLE_OUTPUTS,
    },
    "opt": {
        "embed_path": "model.decoder.embed_tokens",
        "layers_path": "model.decoder.layers",
        "lm_head_path": "lm_head",
        "n_layers_attr": "num_hidden_layers",
        "layer_components": ["output"],
        "attn_path": "self_attn",
        "mlp_path": "fc2",          # OPT uses fc1/fc2 for MLP
        "ln1_path": "self_attn_layer_norm",
        "ln2_path": "final_layer_norm",
        "tuple_layer_components": _TRANSFORMER_TUPLE_OUTPUTS,
    },
    "falcon": {
        "embed_path": "transformer.word_embeddings",
        "layers_path": "transformer.h",
        "lm_head_path": "lm_head",
        "n_layers_attr": "num_hidden_layers",
        "layer_components": ["output"],
        "attn_path": "self_attention",
        "mlp_path": "mlp",
        "ln1_path": "ln_attn",
        "ln2_path": "ln_mlp",
        "tuple_layer_components": _TRANSFORMER_TUPLE_OUTPUTS,
    },
    "mamba": {
        # Mamba is an SSM – no attention or MLP sub-paths in the usual sense.
        # We expose the mixer (SSM block) and the residual output.
        # Mamba layers return a plain tensor (not a tuple), so tuple_layer_components is empty.
        "embed_path": "backbone.embeddings",
        "layers_path": "backbone.layers",
        "lm_head_path": "lm_head",
        "n_layers_attr": "num_hidden_layers",
        "layer_components": ["output"],  # residual after mixer
        "attn_path": "mixer",            # the SSM mixer block
        "mlp_path": "mixer",             # same – Mamba has no separate MLP
        "ln1_path": "norm",
        "ln2_path": "norm",
        "tuple_layer_components": [],
    },
    "mistral": {
        "embed_path": "model.embed_tokens",
        "layers_path": "model.layers",
        "lm_head_path": "lm_head",
        "n_layers_attr": "num_hidden_layers",
        "layer_components": ["output"],
        "attn_path": "self_attn",
        "mlp_path": "mlp",
        "ln1_path": "input_layernorm",
        "ln2_path": "post_attention_layernorm",
        "tuple_layer_components": _TRANSFORMER_TUPLE_OUTPUTS,
    },
    "phi": {
        "embed_path": "model.embed_tokens",
        "layers_path": "model.layers",
        "lm_head_path": "lm_head",
        "n_layers_attr": "num_hidden_layers",
        "layer_components": ["output"],
        "attn_path": "self_attn",
        "mlp_path": "mlp",
        "ln1_path": "input_layernorm",
        "ln2_path": "post_attention_layernorm",  # Phi-3 uses resid_attn_dropout but ln works
        "tuple_layer_components": _TRANSFORMER_TUPLE_OUTPUTS,
    },
}


def _get_nested_attr(obj, dotted_path: str):
    """Resolve a dotted attribute path on *obj* (e.g. 'model.layers')."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


class BaseInterpreter:
    """Base class for model interpretation with core functionality."""

    def __init__(
        self,
        num_items: int,
        checkpoint_dir: str,
        model_type: str = "pythia",
        device: str = "cuda:0",
        n_tokens: int = 100,
        data_dir: str = None,
        n_prompts: int = 100,
        **kwargs,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_type = model_type
        self.device = device
        self.num_items = num_items
        self.n_tokens = n_tokens
        self.data_dir = data_dir
        self.n_prompts = n_prompts

        # Permutation task setup
        perm_task = PermutationTask(num_items=self.num_items)
        self.action_to_nl = perm_task.action_to_nl
        self.nl_to_action = perm_task.nl_to_action
        self.state_to_nl = perm_task.state_to_nl
        self.nl_to_state = perm_task.nl_to_state

        if self.num_items == 3:
            self.INIT_STATE = np.array([1, 2, 3])
        elif self.num_items == 5:
            self.INIT_STATE = np.array([1, 2, 3, 4, 5])

        self.setup_model()
        self.sf = LogSoftmax(dim=-1)
        self.visualization_manager = VisualizationManager()

    # ── Architecture helpers ────────────────────────────────────────────────

    def _arch(self) -> dict:
        """Return the architecture descriptor for the current model_type."""
        arch = _ARCH_MAP.get(self.model_type)
        if arch is None:
            raise ValueError(
                f"Unsupported model_type='{self.model_type}'. "
                f"Supported: {list(_ARCH_MAP.keys())}"
            )
        return arch

    def _resolve_n_layers(self) -> int:
        arch = self._arch()
        attr = arch["n_layers_attr"]
        return getattr(self.model.config, attr)

    # ── Model setup ─────────────────────────────────────────────────────────

    def setup_model(self):
        """Initialise tokenizer + nnsight LanguageModel and resolve internal paths."""
        arch = self._arch()

        # Tokenizer
        if self.model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.checkpoint_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.add_tokens(
            list(self.action_to_nl.values())
            + [f" {a}" for a in self.action_to_nl.values()]
        )

        # nnsight model
        self.model = LanguageModel(
            self.checkpoint_dir, device_map=self.device, dispatch=True
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Resolve internal paths
        self.n_layers = self._resolve_n_layers()
        self.layer_names = [arch["layer_components"] for _ in range(self.n_layers)]

        self.inner_model = _get_nested_attr(self.model, arch["embed_path"].rsplit(".", 1)[0]) \
            if "." in arch["embed_path"] else self.model
        self.embeddings = _get_nested_attr(self.model, arch["embed_path"])
        self.model_layers = _get_nested_attr(self.model, arch["layers_path"])
        self.lm_head = _get_nested_attr(self.model, arch["lm_head_path"])

        # Also keep the arch descriptor for use in patching/probing
        self._arch_desc = arch

        # Action → token-id mapping
        self.action_to_token_ids = {
            action: self.tokenizer.convert_tokens_to_ids(" " + action)
            for action in self.nl_to_action
        }

    # ── Prompt generation ────────────────────────────────────────────────────

    def generate_prompts(self) -> Set[str]:
        """Generate prompts from data files or randomly."""
        if self.data_dir:
            prompts: Set[str] = set()
            for data_file in tqdm(os.listdir(self.data_dir), desc="Loading prompts from files"):
                with open(os.path.join(self.data_dir, data_file), "r") as f:
                    content = json.load(f)
                    prompts.add(" ".join(content["story"].split()[: self.n_tokens]))
                if len(prompts) >= self.n_prompts:
                    break
        else:
            prompts = set()
            for _ in tqdm(range(self.n_prompts), desc="Generating random prompts"):
                prompt = random.choices(list(self.action_to_nl.values()), k=self.n_tokens)
                prompts.add(" ".join(prompt))
        return prompts

    # ── State tracking ───────────────────────────────────────────────────────

    def cumulative_product(self, prompt: str):
        """Return cumulative permutation states for each token in *prompt*."""
        states = []
        curr_state = list(self.INIT_STATE)
        for token in self.tokenizer.tokenize(prompt):
            curr_action = self.nl_to_action[token.strip()]
            new_state = np.copy(curr_state)
            for old_pos, new_pos in enumerate(curr_action):
                new_state[old_pos] = curr_state[new_pos - 1]
            curr_state = new_state
            states.append(tuple(curr_state))
        return states

    # ── Generic component accessor ───────────────────────────────────────────

    def _get_layer_component(self, layer_idx: int, component_path: str):
        """
        Return the nnsight proxy for a sub-component within layer *layer_idx*.

        For standard residual ("output") components the first element of the
        tuple output is returned automatically.  For named sub-paths the raw
        attribute is returned.
        """
        layer = self.model_layers[layer_idx]
        parts = component_path.split(".")
        obj = layer
        for part in parts:
            obj = getattr(obj, part)

        # Transformer layers return tuples for certain outputs; Mamba layers do not.
        tuple_outputs = set(self._arch_desc.get("tuple_layer_components", []))
        if component_path in tuple_outputs:
            return obj[0]
        return obj
