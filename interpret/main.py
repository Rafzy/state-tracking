"""
Entry point for model interpretation experiments.

Supported --model_type values
──────────────────────────────
  gpt2     – GPT-2 family
  pythia   – EleutherAI Pythia (GPT-NeoX)
  llama    – LLaMA / LLaMA-2 / LLaMA-3
  bloom    – BigScience BLOOM
  opt      – Meta OPT
  falcon   – TII UAE Falcon
  mamba    – Mamba SSM
  mistral  – Mistral AI Mistral / Mixtral
  phi      – Microsoft Phi-2 / Phi-3
"""

import argparse
import json
import os
import random
from typing import Set

from tqdm import tqdm

from interpret.interpreters import (
    ActivationPatchingInterpreter,
    BaseInterpreter,
    LengthwiseProbeInterpreter,
    ProbeInterpreter,
)
from interpret.visualization_manager import VisualizationManager

# All model_type keys that have an entry in the arch map
_SUPPORTED_MODEL_TYPES = [
    "gpt2",
    "pythia",
    "llama",
    "bloom",
    "opt",
    "falcon",
    "mamba",
    "mistral",
    "phi",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural network interpretation tools")

    # ── Model configuration ──────────────────────────────────────────────────
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        choices=_SUPPORTED_MODEL_TYPES,
        help=(
            "Architecture family of the model to interpret. "
            f"Supported: {_SUPPORTED_MODEL_TYPES}"
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoint_fromscratch_llama3_S3_state/checkpoint-7032",
        help="Directory containing the model checkpoint",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=3,
        help="Number of items in the permutation task (3 or 5)",
    )

    # ── Data settings ────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Directory containing data files (empty → generate random prompts)",
    )
    parser.add_argument(
        "--n_prompts", type=int, default=100, help="Number of prompts to generate/use"
    )
    parser.add_argument(
        "--n_tokens", type=int, default=100, help="Number of tokens per prompt"
    )

    # ── Interpretation type ──────────────────────────────────────────────────
    parser.add_argument(
        "--interpret_type",
        type=str,
        default="intervene",
        choices=["intervene", "probe", "lengthwise_probe"],
        help="Type of interpretation to perform",
    )

    # ── Intervention settings ────────────────────────────────────────────────
    parser.add_argument(
        "--intervene_output_type",
        type=str,
        default="state",
        choices=["state", "state_by_parity"],
        help="Output type for intervention analysis",
    )
    parser.add_argument(
        "--patching_mode",
        type=str,
        default="substitution",
        choices=["deletion", "substitution"],
        help="Activation patching mode",
    )

    # ── Probe settings ───────────────────────────────────────────────────────
    parser.add_argument(
        "--probe_output_types",
        type=str,
        default="state,state_parity",
        help="Comma-separated list of probe target types",
    )

    # ── Lengthwise probe / shared ────────────────────────────────────────────
    parser.add_argument(
        "--token_increments",
        type=int,
        default=5,
        help="Token increment size for lengthwise probing / patching",
    )
    parser.add_argument(
        "--start_idx", type=int, default=5, help="Start index for lengthwise probe"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="End index for lengthwise probe"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    common_kwargs = dict(
        num_items=args.num_items,
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
        device="cuda:0",
        data_dir=args.data_dir or None,
        n_prompts=args.n_prompts,
        n_tokens=args.n_tokens,
    )

    if args.interpret_type == "intervene":
        interpreter = ActivationPatchingInterpreter(
            **common_kwargs,
            intervene_output_type=args.intervene_output_type,
            patching_mode=args.patching_mode,
            token_increments=args.token_increments,
        )

    elif args.interpret_type == "probe":
        interpreter = ProbeInterpreter(
            **common_kwargs,
            probe_output_types=args.probe_output_types.split(","),
        )

    elif args.interpret_type == "lengthwise_probe":
        interpreter = LengthwiseProbeInterpreter(
            **common_kwargs,
            probe_output_types=args.probe_output_types.split(","),
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            token_increments=args.token_increments,
        )

    else:
        raise ValueError(f"Unknown interpret_type: {args.interpret_type}")

    interpreter.run()


if __name__ == "__main__":
    main()
