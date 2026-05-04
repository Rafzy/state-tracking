"""
Probing interpreter – trains linear probes on layer representations.

The layer representation extraction loop uses the generic
self.model_layers / self.embeddings / self.layer_names / self.n_layers
attributes set by BaseInterpreter, so it works across all architectures
without any model-type branching.
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from interpret.interpreters.base_interpreter import BaseInterpreter
from interpret.metadata_processors import MetadataProcessor


class ProbeInterpreter(BaseInterpreter):
    """Linear probing analysis across all layer representations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_output_types = kwargs.get("probe_output_types", ["state", "state_parity"])

    # ── Metadata ─────────────────────────────────────────────────────────────

    def get_metadata_entries(
        self, prompts, end_idx, metadata_entries=None, token_positions=None, tqdm_desc=None
    ):
        if metadata_entries is None:
            metadata_entries = ["tokens", "state", "state_parity", "action_parity"]
        df = pd.DataFrame(columns=metadata_entries)

        for prompt in tqdm(prompts, desc=tqdm_desc):
            prompt_toks = self.tokenizer.tokenize(prompt)
            metadata_processor = MetadataProcessor(
                self, prompt_toks=prompt_toks, end_idx=end_idx
            )
            df = metadata_processor.add_all_entries_to_df(
                df, metadata_entries, token_positions=token_positions
            )
            if token_positions is None:
                token_positions = list(range(len(prompt_toks)))
        return df

    # ── Representation extraction ─────────────────────────────────────────────

    def get_layer_representations(self, prompts, token_positions=None, tqdm_desc=None):
        """Extract hidden-state representations for each prompt.

        Uses the generic self.model_layers / self.embeddings / self.layer_names
        so it is architecture-agnostic.
        """
        all_layer_reps: Dict[str, list] = {}

        for prompt in tqdm(prompts, desc=tqdm_desc):
            layer_reps: Dict[str, Any] = {}

            with torch.no_grad():
                with self.model.trace() as tracer:
                    with tracer.invoke(prompt):
                        # Embedding layer
                        layer_reps["embed"] = (
                            self.embeddings.output[:, token_positions, :].save()
                        )

                        # Transformer layers
                        for layer_idx in range(self.n_layers):
                            for layer_component_path in self.layer_names[layer_idx]:
                                layer = self.model_layers[layer_idx]
                                subcomponents = layer_component_path.split(".")
                                model_component = layer
                                for sc in subcomponents:
                                    model_component = getattr(model_component, sc)

                                tuple_outputs = set(self._arch_desc.get("tuple_layer_components", []))
                                key = f"{layer_idx}.{layer_component_path}"
                                if layer_component_path in tuple_outputs:
                                    layer_reps[key] = (
                                        model_component[0][:, token_positions, :].save()
                                    )
                                else:
                                    layer_reps[key] = (
                                        model_component[:, token_positions, :].save()
                                    )

                # Convert to numpy
                for item in layer_reps:
                    layer_reps[item] = layer_reps[item].cpu().numpy() * 1.0
                    if item not in all_layer_reps:
                        all_layer_reps[item] = []
                    all_layer_reps[item].append(layer_reps[item].squeeze())

        for item in tqdm(all_layer_reps, desc="Stacking"):
            all_layer_reps[item] = np.stack(all_layer_reps[item], axis=0)

        return all_layer_reps

    # ── Probes ────────────────────────────────────────────────────────────────

    def get_linear_probe_scores(self, reps, labels, num_labels: int, split: float = 0.8):
        if labels.dtype == np.dtype("O"):
            unique_labels = np.unique(labels)
            label_to_id = {lb: idx for idx, lb in enumerate(unique_labels)}
            labels = np.array([label_to_id[lb] for lb in labels])

        probe = LogisticRegression(max_iter=5000, n_jobs=1)
        split_idx = min(1000, int(split * len(reps)))
        train_reps, test_reps = reps[:split_idx], reps[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]

        probe.fit(train_reps, train_labels)
        probs = probe.predict_proba(test_reps)

        if probs.shape[1] < num_labels:
            full_probs = np.zeros((probs.shape[0], num_labels))
            for i, cls in enumerate(probe.classes_):
                full_probs[:, cls] = probs[:, i]
            probs = full_probs

        return probe.score(test_reps, test_labels), probs, test_labels, probe

    def train_probes(self, layer_reps, metadata_df, probe_output_types=None):
        if probe_output_types is None:
            probe_output_types = ["state_parity", "state"]

        probes: Dict[str, Dict] = {}
        layerwise_type_scores: Dict[str, list] = {}
        layerwise_type_probs: Dict[str, list] = {}
        layerwise_type_labels: Dict[str, list] = {}

        for probe_type in probe_output_types:
            probes[probe_type] = {}
            layerwise_type_scores[probe_type] = []
            layerwise_type_probs[probe_type] = []
            layerwise_type_labels[probe_type] = []

            y = metadata_df[probe_type]
            num_labels = len(y.unique())
            na_mask = y.isna()
            y = y[~na_mask]

            for layer_name in tqdm(layer_reps, desc=f"Training {probe_type} probes"):
                X = layer_reps[layer_name]
                X = X.reshape(-1, X.shape[-1])[~na_mask]
                score, probs, labels, probe = self.get_linear_probe_scores(
                    X, y, num_labels
                )
                probes[probe_type][layer_name] = probe
                layerwise_type_scores[probe_type].append(score)
                layerwise_type_probs[probe_type].append(probs)
                layerwise_type_labels[probe_type].append(labels)

        return probes, layerwise_type_scores, layerwise_type_probs, layerwise_type_labels

    def run_probe(self, prompts, end_idx, probe_output_types=None):
        if probe_output_types is None:
            probe_output_types = ["state", "state_parity"]

        layer_reps = self.get_layer_representations(
            prompts,
            token_positions=list(range(end_idx)),
            tqdm_desc="Extracting representations for probing",
        )
        metadata_df = self.get_metadata_entries(
            prompts, end_idx=end_idx, tqdm_desc=f"Getting metadata for length {end_idx}"
        )
        _, layerwise_type_scores, _, _ = self.train_probes(
            layer_reps, metadata_df, probe_output_types=probe_output_types
        )
        return layerwise_type_scores

    def run(self) -> None:
        prompts = self.generate_prompts()
        probe_results = self.run_probe(
            prompts,
            end_idx=self.n_tokens,
            probe_output_types=self.probe_output_types,
        )
        self.visualization_manager.plot_probes(
            probe_results, plot_name=f"{self.checkpoint_dir}"
        )


# ── Lengthwise probe ────────────────────────────────────────────────────────

class LengthwiseProbeInterpreter(ProbeInterpreter):
    """Runs probing across a sweep of token lengths."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_idx = kwargs.get("start_idx", 0)
        self.end_idx = kwargs.get("end_idx", self.n_tokens)
        self.token_increments = kwargs.get("token_increments", 1)

    def per_length_lm_representations(self, end_idx: int, n_prompts: int) -> Set[str]:
        prompts: Set[str] = set()
        for _ in tqdm(range(n_prompts), desc=f"Generating prompts for length {end_idx}"):
            prompt = random.choices(list(self.action_to_nl.values()), k=end_idx)
            prompts.add(" ".join(prompt))
        return prompts

    def per_length_lm_probes(self, prompts, token_idx: int, probe_output_types):
        layer_reps = self.get_layer_representations(
            prompts,
            token_positions=list(range(token_idx)),
            tqdm_desc=f"Getting representations for length {token_idx}",
        )
        metadata_df = self.get_metadata_entries(
            prompts,
            end_idx=token_idx,
            tqdm_desc=f"Getting metadata for length {token_idx}",
        )
        return self.train_probes(layer_reps, metadata_df, probe_output_types=probe_output_types)

    def sweep_length_probes(
        self,
        n_prompts: int,
        n_tokens: int,
        token_increments: int,
        checkpoint_name: str,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        probe_output_types=None,
    ):
        if end_idx is None:
            end_idx = n_tokens
        if probe_output_types is None:
            probe_output_types = ["state", "state_parity"]

        os.makedirs(f"probe_results/{checkpoint_name}", exist_ok=True)
        probe_results = {pt: {} for pt in probe_output_types}

        for length in range(start_idx, end_idx, token_increments):
            print(f"Running probes for length {length}")
            length_prompts = self.per_length_lm_representations(length, n_prompts)

            not_computed = []
            for pt in probe_output_types:
                save_file = f"probe_results/{checkpoint_name}/{length}_{pt}.npy"
                if os.path.exists(save_file):
                    probe_results[pt][length] = np.load(save_file)
                else:
                    not_computed.append(pt)

            if not_computed:
                _, layerwise_type_scores, _, _ = self.per_length_lm_probes(
                    length_prompts, token_idx=length, probe_output_types=not_computed
                )
                for pt in not_computed:
                    probe_results[pt][length] = layerwise_type_scores[pt]
                    np.save(
                        f"probe_results/{checkpoint_name}/{length}_{pt}.npy",
                        np.array(layerwise_type_scores[pt]),
                    )

        return probe_results

    def run(self):
        layer_names_plot = ["embed"]
        for layer_idx, layer in enumerate(self.layer_names):
            for component in layer:
                component_name = "res" if component == "output" else component
                layer_names_plot.append(f"({layer_idx}, {component_name})")

        probe_results = self.sweep_length_probes(
            self.n_prompts,
            self.n_tokens,
            self.token_increments,
            self.checkpoint_dir,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            probe_output_types=self.probe_output_types,
        )

        self.visualization_manager.plot_length_probe_heatmap(
            probe_results["state"],
            layer_names=layer_names_plot,
            plot_name=f"{self.checkpoint_dir}_{self.n_tokens}",
        )
