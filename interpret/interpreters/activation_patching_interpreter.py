"""
Activation-patching interpreter.

Uses the architecture-agnostic helpers in BaseInterpreter to trace hidden
states and perform substitution / deletion patching across all supported
model types.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from interpret.interpreters.base_interpreter import BaseInterpreter
from permutation_task import compute_parity


class ActivationPatchingInterpreter(BaseInterpreter):
    """Activation patching analysis for any supported architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervene_output_type = kwargs.get("intervene_output_type", "state")
        self.patching_mode = kwargs.get("patching_mode", "substitution")
        self.token_increments = kwargs.get("token_increments", 1)

    # ── Prompt pair generation ────────────────────────────────────────────────

    def generate_prompt_pairs(
        self, prompts: Set[str], diff_parity: bool = False
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Generate minimal pairs for patching experiments."""
        import random

        if diff_parity:
            pairs: Dict[str, List] = {"same_parity": [], "diff_parity": []}
        else:
            pairs = {"all": []}

        for prompt in prompts:
            item_to_replace = 0
            split_prompt = prompt.split()
            action_parity = compute_parity(
                self.nl_to_action[split_prompt[item_to_replace]]
            )

            for parity_type in pairs:
                sp = list(split_prompt)  # fresh copy per parity type
                if parity_type in ("same_parity", "all"):
                    candidates = [
                        a for a in self.action_to_nl
                        if compute_parity(a) == action_parity
                        and a != self.nl_to_action[sp[item_to_replace]]
                    ]
                    sp[item_to_replace] = self.action_to_nl[random.choice(candidates)]
                    pairs[parity_type].append((prompt, " ".join(sp)))

                elif parity_type == "diff_parity":
                    sp = list(split_prompt)
                    candidates = [
                        a for a in self.action_to_nl if compute_parity(a) != action_parity
                    ]
                    sp[item_to_replace] = self.action_to_nl[random.choice(candidates)]
                    pairs[parity_type].append((prompt, " ".join(sp)))

        return pairs

    # ── Hidden-state extraction ──────────────────────────────────────────────

    def get_hs_logits(
        self,
        prompt: str,
        layer_components=None,
    ):
        """
        Extract hidden states and final logits for *prompt* using nnsight.

        Returns a tuple (hidden_states_dict, logit_dict).
        """
        hidden_states: Dict[int, Dict[str, Any]] = {}
        arch = self._arch_desc

        with torch.no_grad():
            with self.model.trace() as tracer:
                with tracer.invoke(prompt):
                    # Embedding layer
                    hidden_states[-1] = {"embed": self.embeddings.output[0].cpu().save()}

                    # Per-layer components
                    for layer_idx in range(self.n_layers):
                        layer = self.model_layers[layer_idx]
                        hs: Dict[str, Any] = {}

                        # Always grab the residual stream output
                        hs["output"] = layer.output[0].cpu().save()

                        # Architecture-specific sub-components
                        attn_path = arch.get("attn_path")
                        mlp_path = arch.get("mlp_path")
                        ln1_path = arch.get("ln1_path")
                        ln2_path = arch.get("ln2_path")

                        if attn_path:
                            try:
                                attn_out = getattr(layer, attn_path).output
                                hs["attn"] = (
                                    attn_out[0] if isinstance(attn_out, tuple) else attn_out
                                ).cpu().save()
                            except Exception:
                                pass

                        if mlp_path and mlp_path != attn_path:
                            try:
                                mlp_out = getattr(layer, mlp_path).output
                                hs["mlp"] = (
                                    mlp_out[0] if isinstance(mlp_out, tuple) else mlp_out
                                ).cpu().save()
                            except Exception:
                                pass

                        if ln1_path:
                            try:
                                ln1_out = getattr(layer, ln1_path).output
                                hs["ln1"] = (
                                    ln1_out[0] if isinstance(ln1_out, tuple) else ln1_out
                                ).cpu().save()
                            except Exception:
                                pass

                        if ln2_path and ln2_path != ln1_path:
                            try:
                                ln2_out = getattr(layer, ln2_path).output
                                hs["ln2"] = (
                                    ln2_out[0] if isinstance(ln2_out, tuple) else ln2_out
                                ).cpu().save()
                            except Exception:
                                pass

                        hidden_states[layer_idx] = hs

                    # Extra components requested by the caller
                    if layer_components is not None:
                        for layer_idx, component in layer_components:
                            if component not in hidden_states.get(layer_idx, {}):
                                try:
                                    comp = self._get_layer_component(layer_idx, component)
                                    hidden_states.setdefault(layer_idx, {})[component] = (
                                        comp.cpu().save()
                                    )
                                except Exception:
                                    pass

                    # Logits
                    clean_logits_raw = self.sf(self.lm_head.output)
                    clean_logits = {
                        action: clean_logits_raw[-1, -1, self.action_to_token_ids[action]]
                        .cpu()
                        .to(dtype=float)
                        .item()
                        .save()
                        for action in self.action_to_token_ids
                    }

        clean_logits = {a: clean_logits[a] * 1.0 for a in clean_logits}
        return hidden_states, clean_logits

    # ── Patching ─────────────────────────────────────────────────────────────

    def get_patching_results_pair(
        self,
        minimal_pair: Tuple[str, str],
        layer_components: List[Tuple[int, str]],
        token_units: int = 1,
        replace_prev_tokens: bool = False,
        replace_next_tokens: bool = False,
        patching_mode: str = "substitution",
    ):
        """Run activation patching for a single prompt pair."""
        _, new_logits = self.get_hs_logits(minimal_pair[1], layer_components)
        _, base_logits = self.get_hs_logits(minimal_pair[0], layer_components)

        all_patching_logits = []
        prompt_tokens = self.tokenizer.tokenize(minimal_pair[0])

        with torch.no_grad():
            for layer_idx, component in layer_components:
                _patching_logits = []

                for token_idx in range(0, len(prompt_tokens), token_units):
                    if replace_prev_tokens:
                        token_range = np.arange(0, min(len(prompt_tokens), token_idx + token_units))
                    elif replace_next_tokens:
                        token_range = np.arange(token_idx, len(prompt_tokens))
                    else:
                        token_range = np.arange(
                            token_idx, min(len(prompt_tokens), token_idx + token_units)
                        )

                    with self.model.trace() as tracer:
                        # Get replacement values (substitution patching)
                        replace_val = None
                        if patching_mode == "substitution":
                            with tracer.invoke(minimal_pair[1]):
                                if layer_idx == -1:
                                    replace_val = self.embeddings.output[0]
                                else:
                                    replace_val = self._get_layer_component(layer_idx, component)

                        # Apply patch to base prompt
                        with tracer.invoke(minimal_pair[0]):
                            if layer_idx == -1:
                                tgt = self.embeddings.output[0]
                            else:
                                tgt = self._get_layer_component(layer_idx, component)

                            if patching_mode == "deletion":
                                tgt[token_range] = 0
                            else:
                                tgt[token_range] = replace_val[token_range]

                            patched_logits_raw = self.sf(self.lm_head.output)
                            patched_logits = {
                                action: patched_logits_raw[-1, -1, self.action_to_token_ids[action]]
                                .cpu()
                                .to(dtype=float)
                                .item()
                                .save()
                                for action in self.action_to_token_ids
                            }

                    _patching_logits.append({a: patched_logits[a] * 1.0 for a in patched_logits})

                all_patching_logits.append(_patching_logits)

        return all_patching_logits, base_logits, new_logits

    # ── Metrics ──────────────────────────────────────────────────────────────

    def get_metric_from_logits(
        self,
        patching_results,
        base_logits,
        new_logits,
        patching_mode: str = "substitution",
    ):
        """Normalised logit-difference metric from activation patching."""
        base_answer = max(base_logits, key=base_logits.get)
        new_answer = max(new_logits, key=new_logits.get)
        answer = new_answer if patching_mode == "substitution" else base_answer
        all_actions = list(self.action_to_nl.values())
        correct_actions = [answer]
        incorrect_actions = (
            [base_answer]
            if patching_mode == "substitution"
            else [a for a in all_actions if a != answer]
        )

        correct_logits = np.array([
            [[comp[a] for a in correct_actions] for comp in layer]
            for layer in patching_results
        ])
        incorrect_logits = np.array([
            [[comp[a] for a in incorrect_actions] for comp in layer]
            for layer in patching_results
        ])

        logit_diff = correct_logits.max(-1) - incorrect_logits.max(-1)

        base_correct = np.array([base_logits[a] for a in correct_actions]).max()
        base_incorrect = np.array([base_logits[a] for a in incorrect_actions]).max()
        base_logit_diff = base_correct - base_incorrect

        if patching_mode == "deletion":
            delta_metric = (logit_diff - base_logit_diff) / (0 - base_logit_diff)
        else:
            new_correct = np.array([new_logits[a] for a in correct_actions]).max()
            new_incorrect = np.array([new_logits[a] for a in incorrect_actions]).max()
            new_logit_diff = new_correct - new_incorrect
            delta_metric = (logit_diff - base_logit_diff) / (new_logit_diff - base_logit_diff)

        return np.clip(delta_metric, 0, 1)

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Run full activation-patching analysis."""
        prompts = self.generate_prompts()
        pairs = self.generate_prompt_pairs(
            prompts, diff_parity="parity" in self.intervene_output_type
        )

        layer_components = [(-1, "embed")]
        layer_names_plot = ["embed"]
        for layer_idx, layer in enumerate(self.layer_names):
            for component in layer:
                layer_components.append((layer_idx, component))
                component_name = "res" if component == "output" else component
                layer_names_plot.append(f"({layer_idx}, {component_name})")

        for intervene_type in ["prefix", "suffix", "window"]:
            for parity_type in pairs:
                output_fn = os.path.join(
                    self.checkpoint_dir,
                    f"{self.intervene_output_type}_{self.patching_mode}",
                )
                os.makedirs(f"figures/intervene/{output_fn}", exist_ok=True)
                print(f"Saving to figures/intervene/{output_fn}")

                all_logit_diffs = []
                for pair in tqdm(
                    pairs[parity_type],
                    desc=f"Processing {parity_type} pairs for {intervene_type}",
                ):
                    patching_results, base_logits, new_logits = (
                        self.get_patching_results_pair(
                            pair,
                            layer_components,
                            token_units=self.token_increments,
                            replace_prev_tokens=intervene_type == "prefix",
                            replace_next_tokens=intervene_type == "suffix",
                            patching_mode=self.patching_mode,
                        )
                    )
                    all_logit_diffs.append(
                        self.get_metric_from_logits(
                            patching_results, base_logits, new_logits, self.patching_mode
                        )
                    )

                for i, (_, logit_diff) in enumerate(
                    zip(pairs[parity_type], all_logit_diffs)
                ):
                    self.visualization_manager.plot_logits(
                        logit_diff,
                        np.arange(0, self.n_tokens, self.token_increments),
                        layer_names_plot,
                        plot_name=os.path.join(output_fn, f"{parity_type}/{i}/{intervene_type}"),
                    )

                plot_name = os.path.join(output_fn, f"{intervene_type}_{parity_type}")
                self.visualization_manager.plot_logits(
                    np.stack(all_logit_diffs).mean(0),
                    np.arange(0, self.n_tokens, self.token_increments),
                    layer_names_plot,
                    plot_name=plot_name,
                )
                print(f"Plot saved to figures/intervene/{plot_name}.png")
