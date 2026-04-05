from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

from neuroswift import NeuroSwiftAssistant


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


def load_instruction_pairs(data_path: Path) -> list[dict[str, str]]:
    if not data_path.exists():
        return []

    pairs: list[dict[str, str]] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        pair = extract_instruction_pair(payload)
        if pair is not None:
            pairs.append(pair)
    return pairs


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run the NeuroSwift CPU-first assistant with built-in RAG.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/neuroswift-tiny"),
        help="Directory produced by train_small_llm.py containing modern artifacts.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="what is neuroswift?",
        help="User question or instruction to answer.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional dataset folder to index for multimodal RAG.",
    )
    parser.add_argument(
        "--fetch-urls",
        action="store_true",
        help="Fetch remote URL contents while indexing .url/.urls files.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling cutoff.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty applied to previously generated tokens.",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=4,
        help="Number of similar examples or documents to retrieve.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found at {args.model_dir}. Run train_small_llm.py first."
        )

    device = torch.device("cpu")
    generation_config_path = args.model_dir / "generation_config.json"
    generation_config = (
        json.loads(generation_config_path.read_text(encoding="utf-8"))
        if generation_config_path.exists()
        else {}
    )
    training_summary_path = args.model_dir / "training_summary.json"
    training_summary = (
        json.loads(training_summary_path.read_text(encoding="utf-8"))
        if training_summary_path.exists()
        else {}
    )

    assistant = NeuroSwiftAssistant.from_pretrained(args.model_dir, device=device)
    indexed_project_docs = 0

    readme_path = Path("README.md")
    if readme_path.exists():
        assistant.rag.add_document(
            text=readme_path.read_text(encoding="utf-8", errors="ignore"),
            source=str(readme_path),
            modality="text",
            metadata={"kind": "project_readme"},
        )
        indexed_project_docs += 1

    docs_dir = Path("docs")
    if docs_dir.exists():
        indexed_project_docs += len(assistant.index_folder(docs_dir))

    data_path = training_summary.get("data_path")
    if data_path:
        assistant.index_instruction_pairs(load_instruction_pairs(Path(data_path)))

    indexed_samples = []
    effective_data_dir = args.data_dir or (
        Path(training_summary["data_dir"]) if training_summary.get("data_dir") else None
    )
    if effective_data_dir is not None:
        indexed_samples = assistant.index_folder(effective_data_dir, fetch_urls=args.fetch_urls)

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(generation_config.get("max_new_tokens", 48))

    temperature = args.temperature
    if temperature is None:
        temperature = float(generation_config.get("temperature", 0.0))

    top_k = args.top_k
    if top_k is None:
        top_k = int(generation_config.get("top_k", 0))

    top_p = args.top_p
    if top_p is None:
        top_p = float(generation_config.get("top_p", 1.0))

    repetition_penalty = args.repetition_penalty
    if repetition_penalty is None:
        repetition_penalty = float(generation_config.get("repetition_penalty", 1.05))

    result = assistant.answer(
        args.prompt,
        retrieve_k=args.retrieve_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    prompt_ids = torch.tensor(
        [assistant.tokenizer.encode(result["compiled_prompt"])],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        cold = assistant.model(prompt_ids, update_plasticity=False)
        warmed = assistant.model(prompt_ids, update_plasticity=True)
        adapted = assistant.model(
            prompt_ids,
            plastic_states=warmed["plastic_states"],
            update_plasticity=False,
        )
        mean_logit_shift = (adapted["logits"] - cold["logits"]).abs().mean().item()
        mean_plastic_norm = torch.stack(
            [state.norm() for state in warmed["plastic_states"]]
        ).mean().item()

    print(f"Loaded model directory: {args.model_dir}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Indexed project docs: {indexed_project_docs}")
    print(f"Indexed multimodal samples: {len(indexed_samples)}")
    print(f"Retrieved examples: {len(result['retrieval_hits'])}")
    print(f"Answer source: {result['answer_source']}")
    print(f"Plasticity mean logit shift: {mean_logit_shift:.6f}")
    print(f"Plastic state mean norm:    {mean_plastic_norm:.6f}")
    print("\nGenerated answer:")
    print(safe_console_text(result["answer"]))


if __name__ == "__main__":
    main()
