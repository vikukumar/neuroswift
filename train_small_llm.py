from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from neuroswift.model import NeuroSwiftConfig, NeuroSwiftLM
from neuroswift.tokenizer import WordTokenizer


def build_examples(token_ids: torch.Tensor, seq_len: int, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    for start in range(0, token_ids.numel() - seq_len - 1, stride):
        chunk = token_ids[start : start + seq_len + 1]
        if chunk.numel() < seq_len + 1:
            break
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    if not inputs:
        raise RuntimeError("Dataset is too small for the chosen sequence length.")

    return torch.stack(inputs), torch.stack(targets)


def load_corpus_lines(data_path: Path, max_lines: int = 0) -> list[str]:
    if not data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    lines = [
        line.strip()
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_lines > 0:
        lines = lines[:max_lines]
    if len(lines) < 10:
        raise RuntimeError(
            f"Training corpus at {data_path} is too small. Expected at least 10 non-empty lines."
        )
    return lines


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train a NeuroSwift language model on an example corpus.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--seq-len", type=int, default=96, help="Training sequence length.")
    parser.add_argument("--stride", type=int, default=48, help="Sliding window stride.")
    parser.add_argument("--lr", type=float, default=2e-3, help="AdamW learning rate.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("examples/data/neuroswift_corpus.txt"),
        help="Path to a newline-delimited training corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/neuroswift-tiny"),
        help="Directory where modern model artifacts will be saved.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Optional cap on the number of corpus lines to load. Use 0 for all lines.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=1024,
        help="Maximum number of training sequences to keep after windowing. Use 0 for all sequences.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="neuroswift uses instruction data",
        help="Prompt used for the post-training generation demo.",
    )
    parser.add_argument(
        "--legacy-checkpoint",
        type=Path,
        default=None,
        help="Optional legacy .pt checkpoint path for compatibility.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cpu")

    corpus = load_corpus_lines(args.data_path, max_lines=args.max_lines)
    tokenizer = WordTokenizer.from_texts(corpus)
    token_ids = torch.tensor(tokenizer.encode_corpus(corpus), dtype=torch.long)

    inputs, targets = build_examples(token_ids, seq_len=args.seq_len, stride=args.stride)
    candidate_sequences = inputs.size(0)
    if args.max_sequences > 0 and candidate_sequences > args.max_sequences:
        keep = torch.randperm(candidate_sequences)[: args.max_sequences]
        inputs = inputs.index_select(0, keep)
        targets = targets.index_select(0, keep)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = NeuroSwiftConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        d_state=16,
        expansion=2,
        conv_kernel=4,
        num_experts=8,
        top_k=2,
        expert_hidden=256,
        plastic_dim=48,
        dropout=0.05,
        aux_loss_scale=1e-2,
    )
    print("NeuroSwift config:", asdict(config))
    print(f"Loaded {len(corpus)} training lines from {args.data_path}")
    print(f"Constructed {candidate_sequences} candidate sequences")
    print(f"Using {len(dataset)} training sequences")

    model = NeuroSwiftLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    last_loss = float("nan")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(
                batch_inputs,
                targets=batch_targets,
                update_plasticity=False,
            )
            loss = outputs["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        mean_loss = running_loss / len(loader)
        last_loss = mean_loss
        print(f"epoch {epoch + 1:02d} | loss={mean_loss:.4f}")

    model.eval()

    demo_text = "neuroswift uses instruction data"
    demo_ids = torch.tensor([tokenizer.encode(demo_text)], dtype=torch.long, device=device)

    with torch.no_grad():
        cold = model(demo_ids, update_plasticity=False)
        warmed = model(demo_ids, update_plasticity=True)
        adapted = model(
            demo_ids,
            plastic_states=warmed["plastic_states"],
            update_plasticity=False,
        )

        plastic_shift = (adapted["logits"] - cold["logits"]).abs().mean().item()
        plastic_norm = torch.stack([state.norm() for state in warmed["plastic_states"]]).mean().item()

    print(f"plasticity mean logit shift: {plastic_shift:.6f}")
    print(f"plastic state mean norm:    {plastic_norm:.6f}")

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=80,
        temperature=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    generation_config = {
        "prompt": args.prompt,
        "max_new_tokens": 80,
        "temperature": 0.9,
    }
    (args.output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2),
        encoding="utf-8",
    )
    training_summary = {
        "creator": "Vikash Kumar",
        "data_path": str(args.data_path),
        "num_lines": len(corpus),
        "candidate_sequences": candidate_sequences,
        "num_sequences": len(dataset),
        "final_loss": last_loss,
        "plasticity_mean_logit_shift": plastic_shift,
        "plasticity_mean_state_norm": plastic_norm,
        "config": config.to_dict(),
        "tokenizer_type": "word",
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )

    if args.legacy_checkpoint is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "corpus_path": str(args.data_path),
            "prompt": args.prompt,
        }
        args.legacy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.legacy_checkpoint)

    print("\nGenerated sample:")
    print(generated_text)
    print(f"\nSaved modern artifacts to: {args.output_dir}")
    if args.legacy_checkpoint is not None:
        print(f"Saved legacy checkpoint to: {args.legacy_checkpoint}")


if __name__ == "__main__":
    main()
