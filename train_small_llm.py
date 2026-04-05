from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from neuroswift.ingest import DatasetFolderReader
from neuroswift.model import NeuroSwiftConfig, NeuroSwiftLM
from neuroswift.tokenizer import WordTokenizer


def format_instruction_prompt(prompt: str) -> str:
    return f"user: {prompt}\nassistant:"


def extract_instruction_pair(payload: dict[str, object]) -> dict[str, str] | None:
    prompt = str(payload.get("prompt", "")).strip()
    response = str(payload.get("response", "")).strip()
    if prompt and response:
        return {"prompt": prompt, "response": response}

    messages = payload.get("messages")
    if isinstance(messages, list):
        user_messages = []
        assistant_messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                user_messages.append(content)
            elif role == "assistant":
                assistant_messages.append(content)

        if user_messages and assistant_messages:
            return {
                "prompt": user_messages[-1],
                "response": assistant_messages[-1],
            }

    return None


def load_instruction_pairs(data_path: Path, max_examples: int = 0) -> list[dict[str, str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {data_path}")

    pairs: list[dict[str, str]] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        pair = extract_instruction_pair(payload)
        if pair is not None:
            pairs.append(pair)

    if max_examples > 0 and len(pairs) > max_examples:
        keep = torch.randperm(len(pairs))[:max_examples].tolist()
        pairs = [pairs[idx] for idx in keep]

    if len(pairs) < 32:
        raise RuntimeError(
            f"Instruction dataset at {data_path} is too small. Expected at least 32 prompt/response pairs."
        )
    return pairs


def truncate_words(text: str, max_words: int = 120) -> str:
    words = text.split()
    return " ".join(words[:max_words]).strip()


def build_pairs_from_folder(
    data_dir: Path,
    max_examples: int = 0,
    fetch_urls: bool = False,
) -> tuple[list[dict[str, str]], int]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    reader = DatasetFolderReader(fetch_urls=fetch_urls)
    samples = []
    pairs: list[dict[str, str]] = []
    processed_files = 0

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        processed_files += 1

        if path.suffix.lower() == ".jsonl":
            direct_pairs = load_instruction_pairs(path, max_examples=0)
            if direct_pairs:
                pairs.extend(direct_pairs)
                continue

        sample_or_samples = reader.read_path(path)
        if sample_or_samples is None:
            continue

        sample_list = sample_or_samples if isinstance(sample_or_samples, list) else [sample_or_samples]
        samples.extend(sample_list)

        for sample in sample_list:
            source_name = Path(sample.source).name if "://" not in sample.source else sample.source
            summary = truncate_words(sample.to_rag_text())
            if not summary:
                continue

            response = f"The {sample.modality} source {source_name} contains: {summary}"
            pairs.append({"prompt": f"what is in {source_name}?", "response": response})
            pairs.append({"prompt": f"summarize {source_name}", "response": response})
            pairs.append(
                {
                    "prompt": f"describe the {sample.modality} input from {source_name}",
                    "response": response,
                }
            )

    if max_examples > 0 and len(pairs) > max_examples:
        keep = torch.randperm(len(pairs))[:max_examples].tolist()
        pairs = [pairs[idx] for idx in keep]

    if len(pairs) < 32:
        raise RuntimeError(
            f"Folder-derived dataset at {data_dir} is too small. Expected at least 32 prompt/response pairs."
        )
    return pairs, max(processed_files, len(samples))


def build_sft_examples(
    pairs: list[dict[str, str]],
    tokenizer: WordTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    input_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    skipped = 0
    pad_id = tokenizer.stoi[tokenizer.pad_token]

    for pair in pairs:
        prompt_ids = tokenizer.encode(format_instruction_prompt(pair["prompt"]), add_eos=False)
        response_ids = tokenizer.encode(pair["response"], add_eos=True)
        full_ids = prompt_ids + response_ids

        if len(full_ids) < 2:
            skipped += 1
            continue

        if len(full_ids) > seq_len + 1:
            full_ids = full_ids[: seq_len + 1]

        if len(prompt_ids) >= len(full_ids):
            skipped += 1
            continue

        input_ids = full_ids[:-1]
        labels: list[int] = []
        for target_pos in range(1, len(full_ids)):
            target_token = full_ids[target_pos]
            if target_pos < len(prompt_ids):
                labels.append(-100)
            else:
                labels.append(target_token)

        if not any(label != -100 for label in labels):
            skipped += 1
            continue

        pad_len = seq_len - len(input_ids)
        if pad_len < 0:
            skipped += 1
            continue

        input_ids = input_ids + ([pad_id] * pad_len)
        labels = labels + ([-100] * pad_len)

        input_rows.append(torch.tensor(input_ids, dtype=torch.long))
        label_rows.append(torch.tensor(labels, dtype=torch.long))

    if not input_rows:
        raise RuntimeError("No usable supervised examples were built from the instruction dataset.")

    return torch.stack(input_rows), torch.stack(label_rows), skipped


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train NeuroSwift on an instruction-style QA dataset.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--seq-len", type=int, default=96, help="Maximum sequence length.")
    parser.add_argument("--lr", type=float, default=2e-3, help="AdamW learning rate.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("examples/data/neuroswift_qa.jsonl"),
        help="Path to a JSONL file with prompt/response training pairs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional folder to auto-ingest into synthetic prompt/response pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/neuroswift-tiny"),
        help="Directory where modern model artifacts will be saved.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=2048,
        help="Maximum number of prompt/response pairs to sample from the dataset. Use 0 for all pairs.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="what is neuroswift?",
        help="Prompt used for the post-training generation demo.",
    )
    parser.add_argument(
        "--legacy-checkpoint",
        type=Path,
        default=None,
        help="Optional legacy .pt checkpoint path for compatibility.",
    )
    parser.add_argument(
        "--fetch-urls",
        action="store_true",
        help="Fetch remote contents while ingesting .url/.urls files from --data-dir.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cpu")

    folder_sample_count = 0
    if args.data_dir is not None:
        pairs, folder_sample_count = build_pairs_from_folder(
            args.data_dir,
            max_examples=args.max_examples,
            fetch_urls=args.fetch_urls,
        )
        data_source_label = str(args.data_dir)
    else:
        pairs = load_instruction_pairs(args.data_path, max_examples=args.max_examples)
        data_source_label = str(args.data_path)

    texts_for_vocab = []
    for pair in pairs:
        texts_for_vocab.append(format_instruction_prompt(pair["prompt"]))
        texts_for_vocab.append(pair["response"])

    tokenizer = WordTokenizer.from_texts(texts_for_vocab)
    inputs, targets, skipped = build_sft_examples(pairs, tokenizer, seq_len=args.seq_len)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = NeuroSwiftConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=160,
        n_layers=4,
        d_state=16,
        expansion=2,
        conv_kernel=4,
        num_experts=8,
        top_k=2,
        expert_hidden=320,
        plastic_dim=64,
        dropout=0.05,
        aux_loss_scale=1e-2,
    )
    print("NeuroSwift config:", config.to_dict())
    print(f"Loaded {len(pairs)} instruction pairs from {data_source_label}")
    if folder_sample_count:
        print(f"Indexed {folder_sample_count} multimodal source files from {args.data_dir}")
    print(f"Built {len(dataset)} supervised examples and skipped {skipped}")

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

    chat_prompt = format_instruction_prompt(args.prompt)
    prompt_ids = torch.tensor([tokenizer.encode(chat_prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        cold = model(prompt_ids, update_plasticity=False)
        warmed = model(prompt_ids, update_plasticity=True)
        adapted = model(
            prompt_ids,
            plastic_states=warmed["plastic_states"],
            update_plasticity=False,
        )

        plastic_shift = (adapted["logits"] - cold["logits"]).abs().mean().item()
        plastic_norm = torch.stack([state.norm() for state in warmed["plastic_states"]]).mean().item()

        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=48,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.05,
            adapt_during_generation=False,
        )

    answer_ids = generated_ids[0, prompt_ids.size(1) :].tolist()
    generated_text = tokenizer.decode(answer_ids).strip()

    print(f"plasticity mean logit shift: {plastic_shift:.6f}")
    print(f"plastic state mean norm:    {plastic_norm:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    generation_config = {
        "prompt_template": "user: {prompt}\nassistant:",
        "prompt": args.prompt,
        "max_new_tokens": 48,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "adapt_during_generation": False,
        "task_type": "instruction_qa",
        "answer_only": True,
    }
    (args.output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2),
        encoding="utf-8",
    )
    training_summary = {
        "creator": "Vikash Kumar",
        "data_path": str(args.data_path) if args.data_dir is None else None,
        "data_dir": str(args.data_dir) if args.data_dir is not None else None,
        "resolved_data_source": data_source_label,
        "folder_sample_count": folder_sample_count,
        "num_instruction_pairs": len(pairs),
        "num_examples": len(dataset),
        "skipped_examples": skipped,
        "final_loss": last_loss,
        "plasticity_mean_logit_shift": plastic_shift,
        "plasticity_mean_state_norm": plastic_norm,
        "config": config.to_dict(),
        "tokenizer_type": "word",
        "task_type": "instruction_qa",
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )

    if args.legacy_checkpoint is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "data_path": str(args.data_path) if args.data_dir is None else None,
            "data_dir": str(args.data_dir) if args.data_dir is not None else None,
            "prompt": args.prompt,
        }
        args.legacy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.legacy_checkpoint)

    print("\nGenerated answer:")
    print(generated_text)
    print(f"\nSaved modern artifacts to: {args.output_dir}")
    if args.legacy_checkpoint is not None:
        print(f"Saved legacy checkpoint to: {args.legacy_checkpoint}")


if __name__ == "__main__":
    main()
