from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from neuroswift.model import NeuroSwiftLM
from neuroswift.tokenizer import CharTokenizer


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Load a trained NeuroSwift model and run inference.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/neuroswift-tiny"),
        help="Directory produced by train_small_llm.py containing modern artifacts.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to generate from. Defaults to the training script prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found at {args.model_dir}. Run train_small_llm.py first."
        )

    device = torch.device("cpu")
    generation_config_path = args.model_dir / "generation_config.json"
    if generation_config_path.exists():
        generation_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
    else:
        generation_config = {"prompt": "neuroswift ", "max_new_tokens": 80, "temperature": 0.9}

    model = NeuroSwiftLM.from_pretrained(args.model_dir, device=device)
    tokenizer = CharTokenizer.from_pretrained(args.model_dir)
    prompt = args.prompt if args.prompt is not None else generation_config.get("prompt", "neuroswift ")
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        cold = model(prompt_ids, update_plasticity=False)
        warmed = model(prompt_ids, update_plasticity=True)
        adapted = model(
            prompt_ids,
            plastic_states=warmed["plastic_states"],
            update_plasticity=False,
        )
        mean_logit_shift = (adapted["logits"] - cold["logits"]).abs().mean().item()
        mean_plastic_norm = torch.stack(
            [state.norm() for state in warmed["plastic_states"]]
        ).mean().item()

        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens or generation_config.get("max_new_tokens", 80),
            temperature=args.temperature if args.temperature is not None else generation_config.get("temperature", 0.9),
        )

    print(f"Loaded model directory: {args.model_dir}")
    print(f"Prompt: {prompt!r}")
    print(f"Plasticity mean logit shift: {mean_logit_shift:.6f}")
    print(f"Plastic state mean norm:    {mean_plastic_norm:.6f}")
    print("\nGenerated sample:")
    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
