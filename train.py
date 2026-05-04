"""
Training script for state-tracking experiments.

Supports all model families via the --model argument.  The architecture is
auto-detected from the model name; you do NOT need a separate --model_type
flag here (that flag is only used by the interpret scripts).
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import wandb
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from permutation_task import PermutationTask, compute_parity
from utils.data_collators import (
    DataCollatorForLanguageModelingWithDirectSupervision,
    DataCollatorForLanguageModelingWithNextTokenSupervision,
)
from utils.data_loaders import ChunkedDataset
from utils.model_utils import setup_model, setup_tokenizer

# ── Supported model name list (for argparse choices) ──────────────────────
_GPT2_MODELS = [
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2",
]
_PYTHIA_MODELS = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    # Legacy capitalisation kept for backward compat
    "EleutherAI/pythia-70M", "EleutherAI/pythia-160M", "EleutherAI/pythia-410M",
    "EleutherAI/pythia-1B", "EleutherAI/pythia-1.4B", "EleutherAI/pythia-2.8B",
    "EleutherAI/pythia-6.9B", "EleutherAI/pythia-12B",
]
_LLAMA_MODELS = [
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3.1-8B",
]
_BLOOM_MODELS = [
    "bigscience/bloom-560m", "bigscience/bloom-1b1", "bigscience/bloom-1b7",
    "bigscience/bloom-3b", "bigscience/bloom-7b1", "bigscience/bloom",
]
_OPT_MODELS = [
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b", "facebook/opt-13b",
    "facebook/opt-30b", "facebook/opt-66b",
]
_FALCON_MODELS = [
    "tiiuae/falcon-7b", "tiiuae/falcon-40b",
    "tiiuae/falcon-rw-1b", "tiiuae/falcon-rw-7b",
]
_MAMBA_MODELS = [
    "state-spaces/mamba-130m-hf", "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf", "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b-hf",
]
_MISTRAL_MODELS = [
    "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-v0.1",
]
_PHI_MODELS = [
    "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
]

_ALL_MODELS = (
    _GPT2_MODELS + _PYTHIA_MODELS + _LLAMA_MODELS + _BLOOM_MODELS
    + _OPT_MODELS + _FALCON_MODELS + _MAMBA_MODELS + _MISTRAL_MODELS
    + _PHI_MODELS
)


# ── Argument parsing ────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        # Allow any string so users can pass custom HF repo names without
        # having to update this list; choices= would block that.
        help=(
            "Model name or HuggingFace repo id. "
            "Well-known options: " + ", ".join(_ALL_MODELS[:8]) + " …"
        ),
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--num_items", type=int, default=3, choices=[3, 5],
        help="Number of items for permutation task"
    )
    parser.add_argument(
        "--supervision_type", type=str, default="next_token",
        choices=["direct_state", "direct_topic", "next_token"],
    )
    parser.add_argument(
        "--layerwise_supervision_type", type=str, default=None,
        help="Path to JSON file with layerwise supervision config",
    )
    parser.add_argument("--no_pretrain", action="store_true", default=False)
    parser.add_argument("--from_checkpoint", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_bfloat16", action="store_true", default=False)
    parser.add_argument("--save_all_checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_determinism", action="store_true", default=False)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--is_parity_cur", action="store_true", default=False)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


# ── Helpers ─────────────────────────────────────────────────────────────────

def setup_data_collator(args, tokenizer, state_tokens, parity=None,
                        layerwise_supervision_config=None):
    if args.supervision_type == "next_token":
        return DataCollatorForLanguageModelingWithNextTokenSupervision(
            tokenizer=tokenizer, max_len=args.max_len
        )
    elif args.supervision_type == "direct_state":
        return DataCollatorForLanguageModelingWithDirectSupervision(
            tokenizer=tokenizer,
            max_len=args.max_len,
            STATE_TOKENS=state_tokens,
            label_key="state_seq",
            PARITY=parity if args.is_parity_cur else None,
            layerwise_intermediate_keys=layerwise_supervision_config,
        )
    else:
        raise ValueError(f"Unknown supervision type: {args.supervision_type}")


def prepare_dataset(args, tokenizer, state_tokens, data_collator, debug=False):
    print("Loading data")
    full_dataset = ChunkedDataset(args.data_dir, args.max_len, chunk_size=1, debug=debug)
    total_size = len(full_dataset)
    train_size = int(0.95 * total_size)

    if args.full_determinism or args.data_determinism:
        indices = list(range(total_size))
        train_dataset = torch.utils.data.Subset(full_dataset, indices[:train_size])
        eval_dataset = torch.utils.data.Subset(full_dataset, indices[train_size:])
    else:
        torch.manual_seed(args.seed)
        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, total_size - train_size]
        )

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator):
    print("Setting up trainer")
    wandb.init(project="state-tracking", name=args.output_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=2000 if not args.save_all_checkpoints else args.save_all_checkpoints,
        save_strategy="steps",
        save_total_limit=None if args.save_all_checkpoints else 1,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps" if args.early_stopping else "epoch",
        eval_steps=2000 if args.early_stopping else None,
        batch_eval_metrics=True,
        eval_on_start=args.early_stopping,
        remove_unused_columns=False,
        report_to="none" if args.disable_wandb else "wandb",
        dataloader_num_workers=1,
        bf16=args.use_bfloat16,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=args.full_determinism,
        metric_for_best_model="eval_loss" if args.early_stopping else None,
        greater_is_better=False if args.early_stopping else True,
        load_best_model_at_end=args.early_stopping,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if args.early_stopping:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    return trainer


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()

    if args.full_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    task = PermutationTask(num_items=args.num_items)
    state_tokens = {state.permutation: state.to_string() for state in task.states}
    action_tokens = {action.permutation: action.to_string() for action in task.actions}

    parity = None
    if args.is_parity_cur:
        parity = {state: compute_parity(state) for state in state_tokens}

    layerwise_supervision_config = None
    if args.layerwise_supervision_type and os.path.exists(args.layerwise_supervision_type):
        with open(args.layerwise_supervision_type) as f:
            layerwise_supervision_config = json.load(f)

    tokenizer = setup_tokenizer(args.model, state_tokens, action_tokens)

    model = setup_model(
        tokenizer=tokenizer,
        model_name=args.model,
        checkpoint_path=args.from_checkpoint or None,
        use_bfloat16=args.use_bfloat16,
        no_pretrain=args.no_pretrain,
        output_dir=args.output_dir,
        use_custom_models=True,
        layerwise_supervision_config=layerwise_supervision_config,
    )

    data_collator = setup_data_collator(
        args, tokenizer, state_tokens, parity, layerwise_supervision_config
    )
    train_dataset, eval_dataset = prepare_dataset(
        args, tokenizer, state_tokens, data_collator, debug=args.debug
    )
    trainer = setup_trainer(
        args, model, tokenizer, train_dataset, eval_dataset, data_collator
    )

    trainer.train(resume_from_checkpoint=args.from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
