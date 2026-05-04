"""
Model wrappers with layer-wise supervision support.

Supported architectures:
  - GPT-2        (gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2)
  - Pythia       (EleutherAI/pythia-*)
  - LLaMA / LLaMA-2 / LLaMA-3  (meta-llama/*)
  - Bloom        (bigscience/bloom-*)
  - OPT          (facebook/opt-*)
  - Falcon       (tiiuae/falcon-*)
  - Mamba        (state-spaces/mamba-*)
  - Mistral      (mistralai/Mistral-*)
  - Phi-2 / Phi-3 (microsoft/phi-*)
"""

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    GPT2LMHeadModel,
    LlamaForCausalLM,
    BloomForCausalLM,
    OPTForCausalLM,
    MistralForCausalLM,
)
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

# Falcon uses RWForCausalLM (older) or FalconForCausalLM (newer HF)
try:
    from transformers import FalconForCausalLM
    _FALCON_CLASS = FalconForCausalLM
except ImportError:
    _FALCON_CLASS = None

# Mamba uses a specialised causal-LM wrapper
try:
    from transformers import MambaForCausalLM
    _MAMBA_CLASS = MambaForCausalLM
except ImportError:
    _MAMBA_CLASS = None

# Phi models
try:
    from transformers import PhiForCausalLM
    _PHI_CLASS = PhiForCausalLM
except ImportError:
    _PHI_CLASS = None


# ---------------------------------------------------------------------------
# Shared masked-LM heads (for eval.py which applies selective masking)
# ---------------------------------------------------------------------------

class _MaskedLMHeadMixin:
    """Mixin: replaces the `forward` so cross-entropy is computed only over
    the action-token positions (via an optional *labels_mask* argument)."""

    def forward(self, input_ids, attention_mask=None, labels=None, labels_mask=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="none")
            if labels_mask is not None:
                total_loss = 0
                for i in range(shift_labels.size(0)):
                    loss = loss_fct(
                        shift_logits[i].view(-1, shift_logits[i].size(-1)),
                        shift_labels[i].view(-1),
                    )
                    total_loss += loss[labels_mask[i, 1:]].mean()
                total_loss /= shift_labels.size(0)
                outputs.logits = shift_logits
            else:
                total_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).mean()
            outputs.loss = total_loss
            outputs["loss"] = total_loss
        return outputs


class GPT2MaskedLMHeadModel(_MaskedLMHeadMixin, GPT2LMHeadModel):
    pass

class LlamaMaskedLMHeadModel(_MaskedLMHeadMixin, LlamaForCausalLM):
    pass

class BloomMaskedLMHeadModel(_MaskedLMHeadMixin, BloomForCausalLM):
    pass

class OPTMaskedLMHeadModel(_MaskedLMHeadMixin, OPTForCausalLM):
    pass

class MistralMaskedLMHeadModel(_MaskedLMHeadMixin, MistralForCausalLM):
    pass

if _FALCON_CLASS is not None:
    class FalconMaskedLMHeadModel(_MaskedLMHeadMixin, _FALCON_CLASS):
        pass
else:
    FalconMaskedLMHeadModel = None

if _PHI_CLASS is not None:
    class PhiMaskedLMHeadModel(_MaskedLMHeadMixin, _PHI_CLASS):
        pass
else:
    PhiMaskedLMHeadModel = None

# Mamba does not use an attention/MLP paradigm so it only supports the
# standard causal-LM head; we still expose it for completeness.
MambaMaskedLMHeadModel = _MAMBA_CLASS  # no masking variant needed


# ---------------------------------------------------------------------------
# Layer-target mixin (layerwise supervision used in train.py)
# ---------------------------------------------------------------------------

class ModelWithLayerTargetsMixin:
    """Mixin: adds a small per-layer classifier head on top of every hidden
    state so that intermediate representations can be directly supervised."""

    def init_layer_targets(self, config, layerwise_supervision_config=None):
        self.layerwise_supervision_config = layerwise_supervision_config
        if self.layerwise_supervision_config is not None:
            # hidden_size attribute name differs across architectures
            hidden_size = getattr(
                config,
                "hidden_size",
                getattr(config, "n_embd", getattr(config, "d_model", None)),
            )
            self.layer_classifiers = nn.ModuleDict({
                layer_idx: nn.Linear(
                    hidden_size,
                    self.layerwise_supervision_config[layer_idx]["n_label_classes"],
                )
                for layer_idx in self.layerwise_supervision_config
            })

    def forward_with_layer_targets(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        loss = 0
        loss_fct = nn.CrossEntropyLoss()

        for i in range(len(hidden_states)):
            layer_key = f"layer_{i - 1}_labels"
            if layer_key in kwargs:
                classification_logits = self.layer_classifiers[str(i - 1)](hidden_states[i])
                shift_logits = classification_logits[..., :-1, :].contiguous()
                shift_labels = kwargs[layer_key][..., 1:].contiguous()
                loss += loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

        if labels is not None:
            final_logits = outputs.logits
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss += loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        outputs.loss = loss
        return outputs


# ---------------------------------------------------------------------------
# Concrete layer-target model classes
# ---------------------------------------------------------------------------

class GPT2ModelWithLayerTargets(ModelWithLayerTargetsMixin, GPT2LMHeadModel):
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)

    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)


class LlamaModelWithLayerTargets(ModelWithLayerTargetsMixin, LlamaForCausalLM):
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)

    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)


class PythiaModelWithLayerTargets(ModelWithLayerTargetsMixin, GPTNeoXForCausalLM):
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)

    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)


class BloomModelWithLayerTargets(ModelWithLayerTargetsMixin, BloomForCausalLM):
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)

    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)


class OPTModelWithLayerTargets(ModelWithLayerTargetsMixin, OPTForCausalLM):
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)

    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)


class MistralModelWithLayerTargets(ModelWithLayerTargetsMixin, MistralForCausalLM):
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)

    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)


if _FALCON_CLASS is not None:
    class FalconModelWithLayerTargets(ModelWithLayerTargetsMixin, _FALCON_CLASS):
        def __init__(self, config, layerwise_supervision_config=None):
            super().__init__(config)
            self.init_layer_targets(config, layerwise_supervision_config)

        def forward(self, *args, **kwargs):
            return self.forward_with_layer_targets(*args, **kwargs)
else:
    FalconModelWithLayerTargets = None


if _PHI_CLASS is not None:
    class PhiModelWithLayerTargets(ModelWithLayerTargetsMixin, _PHI_CLASS):
        def __init__(self, config, layerwise_supervision_config=None):
            super().__init__(config)
            self.init_layer_targets(config, layerwise_supervision_config)

        def forward(self, *args, **kwargs):
            return self.forward_with_layer_targets(*args, **kwargs)
else:
    PhiModelWithLayerTargets = None


# Mamba does not have traditional "hidden states" in the transformer sense.
# We expose it for NTP training only; layerwise supervision is not supported.
if _MAMBA_CLASS is not None:
    class MambaModelWithLayerTargets(_MAMBA_CLASS):
        """Mamba wrapper stub: layerwise supervision is not applicable to SSMs.
        Falls back to standard causal-LM forward for all supervision types."""

        def __init__(self, config, layerwise_supervision_config=None):
            super().__init__(config)
            if layerwise_supervision_config is not None:
                import warnings
                warnings.warn(
                    "Layerwise supervision is not supported for Mamba. "
                    "The layerwise_supervision_config argument will be ignored.",
                    UserWarning,
                )

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            # Strip any layer_N_labels that the collator may have injected
            kwargs = {k: v for k, v in kwargs.items() if not k.startswith("layer_")}
            return super().forward(
                input_ids, attention_mask=attention_mask, labels=labels, use_cache = False, **kwargs
            )
else:
    MambaModelWithLayerTargets = None
