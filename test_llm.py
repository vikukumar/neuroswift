"""
test_llm.py
===========
Inference / evaluation script for a trained NeuroSwiftLM or NeuroSwiftOmni model.

Features
--------
* Auto-ingest --data-dir or --data-path for RAG context (all file types)
* Rich plasticity analysis: logit-shift + state-norm display
* Batch evaluation mode: score a JSONL file of prompt/response pairs
* Interactive REPL mode (--interactive)
* Auto-detect model type (LM vs Omni)
* Post-answer generation accuracy metrics (token match rate)

Usage
-----
    # Single prompt
    python test_llm.py --prompt "what is neuroswift?"

    # With data-dir for RAG indexing
    python test_llm.py --data-dir my_data/ --prompt "explain quantum physics"

    # Interactive REPL
    python test_llm.py --interactive

    # Batch evaluation
    python test_llm.py --eval-file data/qa.jsonl --model-dir artifacts/neuroswift-tiny

    # Generate an image (Omni model)
    python test_llm.py --model-dir artifacts/neuroswift-omni --modality image
"""
from __future__ import annotations

import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch

from neuroswift.data_pipeline import run_pipeline
from neuroswift.layers import auto_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("NeuroSwift.Test")


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------


def _safe(text: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _print_header(model_dir: Path, args: Any) -> None:
    print("\n" + "═" * 56)
    print("  NeuroSwift Inference")
    print(f"  Model   : {model_dir}")
    print(f"  Device  : {args.device or 'auto'}")
    print("═" * 56)


def _token_match_rate(pred: str, ref: str) -> float:
    """Simple token-overlap accuracy (unigram F1)."""
    pred_toks = set(pred.lower().split())
    ref_toks = set(ref.lower().split())
    if not ref_toks:
        return 0.0
    overlap = pred_toks & ref_toks
    precision = len(overlap) / max(len(pred_toks), 1)
    recall = len(overlap) / max(len(ref_toks), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> ArgumentParser:
    p = ArgumentParser(
        description="Test / evaluate a trained NeuroSwift model.",
        epilog="""
Examples:
  python test_llm.py --prompt "what is neuroswift?"
  python test_llm.py --data-dir my_data/ --prompt "summarize the data"
  python test_llm.py --interactive
  python test_llm.py --eval-file data/qa.jsonl
  python test_llm.py --model-dir artifacts/neuroswift-omni --modality image
""",
    )
    p.add_argument("--model-dir", type=Path, default=Path("artifacts/neuroswift-tiny"),
                   help="Model directory from train_small_llm.py or train_omni.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt to answer.")
    p.add_argument("--device", type=str, default=None,
                   help="cpu / cuda / mps (default: auto).")

    # Data context for RAG
    rag_grp = p.add_argument_group("RAG / Data Context")
    rag_grp.add_argument("--data-dir", type=Path, default=None,
                         help="Folder to ingest for RAG context (all file types).")
    rag_grp.add_argument("--data-path", type=Path, default=None,
                         help="Single file to ingest for RAG context.")
    rag_grp.add_argument("--fetch-urls", action="store_true")
    rag_grp.add_argument("--retrieve-k", type=int, default=4,
                         help="Number of RAG documents to retrieve.")

    # Generation params
    gen_grp = p.add_argument_group("Generation")
    gen_grp.add_argument("--max-new-tokens", type=int, default=None)
    gen_grp.add_argument("--temperature", type=float, default=None)
    gen_grp.add_argument("--top-k", type=int, default=None)
    gen_grp.add_argument("--top-p", type=float, default=None)
    gen_grp.add_argument("--repetition-penalty", type=float, default=None)
    gen_grp.add_argument("--modality", type=str, default=None,
                         choices=["image", "audio", "video", "all"],
                         help="Generate multimodal artifact (Omni model only).")
    gen_grp.add_argument("--out-dir", type=str, default="artifacts/generated",
                         help="Output directory for generated artifact files.")

    # Modes
    mode_grp = p.add_argument_group("Modes")
    mode_grp.add_argument("--interactive", action="store_true",
                          help="Run interactive chat REPL.")
    mode_grp.add_argument("--eval-file", type=Path, default=None,
                          help="JSONL file of {prompt, response} pairs to batch-evaluate.")
    mode_grp.add_argument("--eval-max", type=int, default=200,
                          help="Max examples to evaluate from --eval-file.")
    mode_grp.add_argument("--show-plasticity", action="store_true", default=True,
                          help="Show plasticity logit shift and state norm.")
    return p


# ---------------------------------------------------------------------------
# Load model helper
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_dir: Path, device: torch.device):
    """Auto-detect and load NeuroSwiftLM or NeuroSwiftOmni."""
    from neuroswift.tokenizer import load_tokenizer

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_dir}. Run train_small_llm.py first."
        )
    config_meta = json.loads(config_path.read_text(encoding="utf-8"))
    is_omni = "text_vocab_size" in config_meta

    tokenizer = load_tokenizer(model_dir)

    if is_omni:
        from neuroswift.omni import NeuroSwiftOmni
        model = NeuroSwiftOmni.from_pretrained(model_dir, device=device)
        return model, tokenizer, "omni"
    else:
        from neuroswift.model import NeuroSwiftLM
        model = NeuroSwiftLM.from_pretrained(model_dir, device=device)
        return model, tokenizer, "lm"


# ---------------------------------------------------------------------------
# Index data for RAG
# ---------------------------------------------------------------------------


def index_data_for_rag(assistant, data_source: Path | None) -> int:
    """Ingest data_source into assistant's RAG. Returns num samples indexed."""
    if data_source is None:
        return 0
    try:
        samples = assistant.index_folder(data_source)
        return len(samples)
    except Exception as exc:
        logger.warning(f"RAG indexing failed: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Plasticity analysis
# ---------------------------------------------------------------------------


@torch.no_grad()
def analyze_plasticity(model, prompt_ids: torch.Tensor, model_type: str) -> dict[str, float]:
    """Run plasticity forward pass and compute logit shift + state norm."""
    if model_type == "omni":
        cold = model(text_input_ids=prompt_ids, update_plasticity=False)
        warmed = model(text_input_ids=prompt_ids, update_plasticity=True)
        c_logits = cold.get("text_logits", cold.get("logits"))
        w_logits = warmed.get("text_logits", warmed.get("logits"))
        plastic_states = warmed.get("plastic_states", [])
    else:
        cold = model(prompt_ids, update_plasticity=False)
        warmed = model(prompt_ids, update_plasticity=True)
        c_logits = cold["logits"]
        w_logits = warmed["logits"]
        plastic_states = warmed["plastic_states"]

    if c_logits is None or w_logits is None:
        return {"logit_shift": 0.0, "state_norm": 0.0}

    logit_shift = (w_logits.float() - c_logits.float()).abs().mean().item()
    state_norm = (
        torch.stack([s.norm() for s in plastic_states]).mean().item()
        if plastic_states
        else 0.0
    )
    return {"logit_shift": logit_shift, "state_norm": state_norm}


# ---------------------------------------------------------------------------
# Multimodal generation
# ---------------------------------------------------------------------------


def generate_modality_artifact(model, tokenizer, prompt: str, modality: str, out_dir: str, device: torch.device) -> str:
    """Generate image/audio/video from an Omni model and return summary."""
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    try:
        artifact = model.generate_artifact(
            prompt_ids,
            prompt_text=prompt,
            modality=modality,
            out_dir=out_dir,
        )
        return artifact.summary()
    except Exception as exc:
        return f"Generation failed: {exc}"


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def batch_evaluate(
    model,
    tokenizer,
    eval_file: Path,
    device: torch.device,
    model_type: str,
    max_examples: int = 200,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
) -> dict[str, float]:
    """
    Evaluate on a JSONL file of {prompt, response} pairs.

    Returns metrics: avg_token_f1, min_f1, max_f1, pct_nonzero.
    """
    # Use data_pipeline to ingest eval pairs
    train_pairs, _, _ = run_pipeline(
        source=eval_file,
        max_total_pairs=max_examples,
        dedup_exact=False,
        dedup_near=False,
        val_fraction=0.0,
        verbose=False,
    )
    pairs = train_pairs[:max_examples]
    if not pairs:
        logger.warning(f"No pairs found in {eval_file}")
        return {}

    eos_id = tokenizer.eos_token_id
    f1_scores: list[float] = []

    for pair in pairs:
        prompt_text = f"user: {pair.prompt}\nassistant:"
        prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

        with torch.no_grad():
            if model_type == "omni":
                # Use backbone text generation
                from neuroswift.model import NeuroSwiftLM
                # Omni text head
                outputs = model(text_input_ids=prompt_ids, update_plasticity=False)
                logits = outputs.get("text_logits", None)
                if logits is None:
                    continue
                # Greedy decode for eval
                gen_ids = [logits[0, -1].argmax().item()]
                pred = tokenizer.decode(gen_ids).strip()
            else:
                gen_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_id=eos_id,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    adapt_during_generation=False,
                )
                answer_ids = gen_ids[0, prompt_ids.size(1):].tolist()
                pred = tokenizer.decode(answer_ids).strip()

        f1 = _token_match_rate(pred, pair.response)
        f1_scores.append(f1)

    if not f1_scores:
        return {}

    avg_f1 = sum(f1_scores) / len(f1_scores)
    pct_nonzero = sum(1 for s in f1_scores if s > 0.0) / len(f1_scores)
    return {
        "num_examples": len(f1_scores),
        "avg_token_f1": avg_f1,
        "min_f1": min(f1_scores),
        "max_f1": max(f1_scores),
        "pct_nonzero": pct_nonzero,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args().parse_args()

    device = torch.device(args.device) if args.device else auto_device()
    _print_header(args.model_dir, args)

    # Load model
    logger.info(f"Loading model from {args.model_dir} …")
    try:
        model, tokenizer, model_type = load_model_and_tokenizer(args.model_dir, device)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.info(f"Model type: {model_type}  |  Vocab: {tokenizer.vocab_size:,}")

    # Load generation config
    gen_cfg_path = args.model_dir / "generation_config.json"
    gen_cfg = json.loads(gen_cfg_path.read_text(encoding="utf-8")) if gen_cfg_path.exists() else {}
    max_new_tokens = args.max_new_tokens or int(gen_cfg.get("max_new_tokens", 64))
    temperature = args.temperature if args.temperature is not None else float(gen_cfg.get("temperature", 0.7))
    top_k = args.top_k if args.top_k is not None else int(gen_cfg.get("top_k", 40))
    top_p = args.top_p if args.top_p is not None else float(gen_cfg.get("top_p", 0.9))
    rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else float(gen_cfg.get("repetition_penalty", 1.1))

    # Load the NeuroSwiftAssistant for RAG
    from neuroswift.omni import NeuroSwiftAssistant
    try:
        assistant = NeuroSwiftAssistant.from_pretrained(args.model_dir, device=device)
    except Exception:
        assistant = None

    # Index data for RAG context
    data_source = args.data_dir or args.data_path
    if data_source and assistant is not None:
        logger.info(f"Indexing {data_source} for RAG context …")
        n = index_data_for_rag(assistant, data_source)
        logger.info(f"Indexed {n} samples into RAG.")

    # Also index README + docs
    if assistant is not None:
        for extra in [Path("README.md"), Path("docs")]:
            if extra.exists():
                try:
                    if extra.is_dir():
                        assistant.index_folder(extra)
                    else:
                        assistant.rag.add_document(
                            text=extra.read_text(encoding="utf-8", errors="ignore"),
                            source=str(extra),
                            modality="text",
                        )
                except Exception:
                    pass

    # ── Batch evaluation mode ──────────────────────────────────────────────
    if args.eval_file is not None:
        if not args.eval_file.exists():
            logger.error(f"Eval file not found: {args.eval_file}")
            sys.exit(1)
        logger.info(f"Batch evaluating on {args.eval_file} (max {args.eval_max} examples) …")
        metrics = batch_evaluate(
            model, tokenizer, args.eval_file, device, model_type,
            max_examples=args.eval_max,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=rep_penalty,
        )
        print("\n── Evaluation Metrics ─────────────────")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<20} {v:.4f}")
            else:
                print(f"  {k:<20} {v}")
        print("──────────────────────────────────────\n")
        return

    # ── Multimodal generation mode (Omni only) ─────────────────────────────
    if args.modality is not None:
        prompt = args.prompt or "generate a realistic scene"
        if model_type != "omni":
            logger.error("--modality requires an Omni model (trained with train-omni).")
            sys.exit(1)
        logger.info(f"Generating {args.modality} artifact for: {prompt!r}")
        summary = generate_modality_artifact(model, tokenizer, prompt, args.modality, args.out_dir, device)
        print("\n── Artifact ───────────────────────────")
        print(summary)
        print("──────────────────────────────────────\n")
        return

    # ── Default prompt or interactive mode ────────────────────────────────
    prompt = args.prompt or ("what is neuroswift?" if not args.interactive else None)

    def _answer_one(user_prompt: str) -> None:
        if assistant is not None:
            result = assistant.answer(
                user_prompt,
                retrieve_k=args.retrieve_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
            )
            answer = result["answer"]
            answer_source = result["answer_source"]
            retrieval_hits = len(result.get("retrieval_hits", []))
            compiled_prompt = result.get("compiled_prompt", user_prompt)
        else:
            # Bare model without NeuroSwiftAssistant
            prompt_text = f"user: {user_prompt}\nassistant:"
            prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
            with torch.no_grad():
                gen_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    adapt_during_generation=False,
                )
            answer_ids = gen_ids[0, prompt_ids.size(1):].tolist()
            answer = tokenizer.decode(answer_ids).strip()
            answer_source = "model_generation"
            retrieval_hits = 0
            compiled_prompt = prompt_text

        print(f"\n  Answer [{answer_source}]:")
        print(f"  {_safe(answer)}")
        print(f"  (RAG hits: {retrieval_hits})")

        # Plasticity analysis
        if args.show_plasticity and model_type == "lm":
            p_ids = torch.tensor([tokenizer.encode(compiled_prompt)], dtype=torch.long, device=device)
            p_stats = analyze_plasticity(model, p_ids, model_type)
            print(f"  Plasticity shift: {p_stats['logit_shift']:.6f}  |  State norm: {p_stats['state_norm']:.6f}")
        print()

    if args.interactive:
        print("\nNeuroSwift Chat — type 'quit' to exit\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not user_input or user_input.lower() in {"quit", "exit", "bye", "q"}:
                print("Goodbye!")
                break
            _answer_one(user_input)
    elif prompt:
        print(f"\n  Prompt: {prompt!r}")
        _answer_one(prompt)
    else:
        print("No prompt provided. Use --prompt 'your question' or --interactive.")


if __name__ == "__main__":
    main()
