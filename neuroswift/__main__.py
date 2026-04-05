"""
neuroswift.__main__
===================
Unified CLI entry point for NeuroSwift.

Usage::

    python -m neuroswift <subcommand> [options]

Subcommands
-----------
train         Train NeuroSwiftLM (text-only, CPU-first)
train-omni    Train NeuroSwiftOmni (full multimodal: text, image, audio, video)
chat          Interactive REPL using NeuroSwiftAssistant
index         Index a folder into the RAG store  (saves rag.json)
generate      Generate image / audio / video from a text prompt
auto-train    Continuous background training loop (WorldAutoTrain)

Examples::

    python -m neuroswift train --data-dir data/ --epochs 5
    python -m neuroswift train-omni --data-dir data/ --epochs 3
    python -m neuroswift chat --model artifacts/neuroswift-tiny
    python -m neuroswift index --source data/ --output artifacts/rag.json
    python -m neuroswift generate --model artifacts/neuroswift-omni --prompt "a sunrise" --modality image
    python -m neuroswift auto-train --data-dir data/ --interval 120
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def _info(text: str) -> None:
    print(f"  {text}")


def _banner() -> None:
    print(_bold("\n  ██╗     ██╗███████╗██╗    ██╗██████╗    "))
    print(_bold("  ██║     ██║██╔════╝██║    ██║██╔══██╗   "))
    print(_bold("  ██║     ██║█████╗  ██║ █╗ ██║╚█████╔╝   "))
    print(_bold("  ██║     ██║██╔══╝  ██║███╗██║██╔══██╗   "))
    print(_bold("  ███████╗██║██║     ╚███╔███╔╝██║  ██║   "))
    print(_bold("  ╚══════╝╚═╝╚═╝      ╚══╝╚══╝ ╚═╝  ╚═╝   "))
    print()
    print("  NeuroSwift — World-class Multimodal Deep Learning")
    print("  CPU-first • SSM+MoE+Plasticity • Omni generation")
    print()


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> None:
    """Delegates to train_small_llm logic inline (NeuroSwiftLM, text-only)."""
    import torch
    from .layers import auto_device
    from .model import NeuroSwiftConfig, NeuroSwiftLM
    from .tokenizer import WordTokenizer

    # Re-use all logic from train_small_llm.py
    # We import it as a module and forward args
    try:
        from importlib import import_module
        train_mod = import_module("neuroswift.train_small_llm" if False else "train_small_llm")
    except ModuleNotFoundError:
        pass

    # Direct inline call to avoid path ambiguity
    import importlib.util, types

    train_path = Path(__file__).parent.parent / "train_small_llm.py"
    if not train_path.exists():
        # Installed package: fall back to inline
        _train_inline(args)
        return

    spec = importlib.util.spec_from_file_location("train_small_llm", train_path)
    if spec is None or spec.loader is None:
        _train_inline(args)
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    # Build a fake argparse namespace compatible with train_small_llm.parse_args()
    train_args = mod.parse_args().parse_args([])
    train_args.epochs = args.epochs
    train_args.batch_size = args.batch_size
    train_args.seq_len = args.seq_len
    train_args.lr = args.lr
    train_args.output_dir = args.output_dir
    train_args.data_path = args.data_path
    train_args.data_dir = args.data_dir
    train_args.max_examples = args.max_examples
    train_args.max_per_file = args.max_per_file
    train_args.no_dedup = args.no_dedup
    train_args.augment = args.augment
    train_args.val_fraction = args.val_fraction
    train_args.early_stop_patience = args.early_stop_patience
    train_args.d_model = args.d_model
    train_args.n_layers = args.n_layers
    train_args.num_experts = args.num_experts
    train_args.expert_hidden = args.expert_hidden
    train_args.prompt = args.prompt
    train_args.fetch_urls = args.fetch_urls
    train_args.device = args.device
    train_args.resume = args.resume
    train_args.legacy_checkpoint = None

    mod.main.__globals__["sys"] = sys
    try:
        # Call main() with overridden args
        _original_parse = mod.parse_args
        def _patched_parse():
            class _P:
                def parse_args(self, _=None):
                    return train_args
            return _P()
        mod.parse_args = _patched_parse
        mod.main()
    finally:
        mod.parse_args = _original_parse


def _train_inline(args: argparse.Namespace) -> None:
    """Minimal inline train fallback when train_small_llm.py is not on disk."""
    print("train_small_llm.py not found; running inline training …")
    from .auto_trainer import AutoTrainer
    trainer = AutoTrainer(
        data_dir=args.data_dir or Path("examples/data"),
        output_dir=args.output_dir,
    )
    trainer.train_once()


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train NeuroSwiftLM (text-only)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--max-examples", type=int, default=50000)
    p.add_argument("--max-per-file", type=int, default=10000)
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--d-model", type=int, default=0)
    p.add_argument("--n-layers", type=int, default=0)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--expert-hidden", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/neuroswift-tiny"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prompt", type=str, default="what is neuroswift?")
    p.add_argument("--fetch-urls", action="store_true")
    p.set_defaults(func=_cmd_train)


# ---------------------------------------------------------------------------
# Sub-command: train-omni
# ---------------------------------------------------------------------------


def _cmd_train_omni(args: argparse.Namespace) -> None:
    from .train_omni import main as omni_main
    # Build argv list for train_omni.parse_args()
    argv = [
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--seq-len", str(args.seq_len),
        "--lr", str(args.lr),
        "--image-size", str(args.image_size),
        "--max-video-frames", str(args.max_video_frames),
        "--video-frame-size", str(args.video_frame_size),
        "--d-model", str(args.d_model),
        "--n-layers", str(args.n_layers),
        "--num-experts", str(args.num_experts),
        "--max-examples", str(args.max_examples),
        "--max-per-file", str(args.max_per_file),
        "--val-fraction", str(args.val_fraction),
        "--text-weight", str(args.text_weight),
        "--image-weight", str(args.image_weight),
        "--audio-weight", str(args.audio_weight),
        "--video-weight", str(args.video_weight),
        "--save-every", str(args.save_every),
        "--prompt", args.prompt,
    ]
    if args.batch_size > 0:
        argv += ["--batch-size", str(args.batch_size)]
    if args.device:
        argv += ["--device", args.device]
    if args.auto_train:
        argv.append("--auto-train")
        argv += ["--auto-interval", str(args.auto_interval)]
    omni_main(argv)


def _add_train_omni_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train-omni", help="Train NeuroSwiftOmni (multimodal: text+image+audio+video)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/neuroswift-omni"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-video-frames", type=int, default=8)
    p.add_argument("--video-frame-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--max-examples", type=int, default=5000)
    p.add_argument("--max-per-file", type=int, default=1000)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--text-weight", type=float, default=1.0)
    p.add_argument("--image-weight", type=float, default=0.5)
    p.add_argument("--audio-weight", type=float, default=0.5)
    p.add_argument("--video-weight", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--prompt", type=str, default="describe this scene")
    p.add_argument("--auto-train", action="store_true")
    p.add_argument("--auto-interval", type=float, default=300.0)
    p.set_defaults(func=_cmd_train_omni)


# ---------------------------------------------------------------------------
# Sub-command: chat
# ---------------------------------------------------------------------------


def _cmd_chat(args: argparse.Namespace) -> None:
    import torch
    from .layers import auto_device
    from .omni import NeuroSwiftAssistant

    device = torch.device(args.device) if args.device else auto_device()
    print(f"Loading model from {args.model} on {device} …")

    try:
        assistant = NeuroSwiftAssistant.from_pretrained(args.model, device=device)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        print("Tip: train a model first with: python -m neuroswift train --output-dir artifacts/neuroswift-tiny")
        sys.exit(1)

    if args.index_dir:
        print(f"Indexing {args.index_dir} into RAG …")
        samples = assistant.index_folder(args.index_dir)
        print(f"Indexed {len(samples)} samples.")

    if args.prompt:
        # Non-interactive single-shot
        result = assistant.answer(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\n[{result['answer_source']}] {result['answer']}\n")
        return

    # Interactive REPL
    print("\nNeuroSwift Chat — type 'quit' or Ctrl+C to exit\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input or user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break
        result = assistant.answer(
            user_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"NeuroSwift [{result['answer_source']}]: {result['answer']}\n")


def _add_chat_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("chat", help="Interactive chat REPL using NeuroSwiftAssistant")
    p.add_argument("--model", type=Path, default=Path("artifacts/neuroswift-tiny"),
                   help="Path to saved model directory.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--index-dir", type=Path, default=None,
                   help="Optional folder to index into RAG before chatting.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single-shot prompt (non-interactive mode).")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.set_defaults(func=_cmd_chat)


# ---------------------------------------------------------------------------
# Sub-command: index
# ---------------------------------------------------------------------------


def _cmd_index(args: argparse.Namespace) -> None:
    from .ingest import DatasetFolderReader
    from .rag import NeuroSwiftRAG

    print(f"Indexing {args.source} …")
    rag = NeuroSwiftRAG.from_folder(args.source, fetch_urls=args.fetch_urls)
    print(f"Indexed {len(rag)} documents.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rag.save(args.output)
    print(f"RAG index saved to: {args.output}")


def _add_index_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("index", help="Index a folder into the RAG store")
    p.add_argument("--source", type=Path, required=True,
                   help="Folder to ingest.")
    p.add_argument("--output", type=Path, default=Path("artifacts/rag.json"),
                   help="Path to save the RAG index JSON.")
    p.add_argument("--fetch-urls", action="store_true")
    p.set_defaults(func=_cmd_index)


# ---------------------------------------------------------------------------
# Sub-command: generate
# ---------------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> None:
    import torch
    from .layers import auto_device
    from .omni import NeuroSwiftOmni
    from .tokenizer import load_tokenizer

    device = torch.device(args.device) if args.device else auto_device()
    print(f"Loading Omni model from {args.model} on {device} …")

    try:
        model = NeuroSwiftOmni.from_pretrained(args.model, device=device)
        tokenizer = load_tokenizer(args.model)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        print("Tip: train an Omni model first with: python -m neuroswift train-omni --data-dir data/")
        sys.exit(1)

    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    out_dir = Path(args.out_dir)
    print(f"Generating {args.modality} for prompt: {args.prompt!r}")

    artifact = model.generate_artifact(
        prompt_ids,
        prompt_text=args.prompt,
        modality=args.modality,
        out_dir=out_dir,
        sample_rate=args.sample_rate,
        video_fps=args.fps,
        assemble_mp4=not args.no_mp4,
    )
    print("\n── Artifact Summary ──")
    print(artifact.summary())
    print()


def _add_generate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("generate", help="Generate image / audio / video from a text prompt")
    p.add_argument("--model", type=Path, required=True,
                   help="Path to a saved NeuroSwiftOmni model directory.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--modality", type=str, default="image",
                   choices=["image", "audio", "video", "all"])
    p.add_argument("--out-dir", type=str, default="artifacts/generated")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--no-mp4", action="store_true",
                   help="Skip MP4 assembly (save frames as PNGs only).")
    p.set_defaults(func=_cmd_generate)


# ---------------------------------------------------------------------------
# Sub-command: auto-train
# ---------------------------------------------------------------------------


def _cmd_auto_train(args: argparse.Namespace) -> None:
    import torch
    from .auto_trainer import AutoTrainer
    from .layers import auto_device

    device = torch.device(args.device) if args.device else auto_device()

    print(f"Starting World Auto-Training")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Device     : {device}")
    print(f"  Interval   : {args.interval}s")
    print(f"  Max cycles : {args.max_cycles or 'unlimited'}")
    print()

    trainer = AutoTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        epochs_per_cycle=args.epochs,
        interval=args.interval,
        min_pairs=args.min_pairs,
        save_every=args.save_every,
        plasticity_warmup=not args.no_plasticity,
        freeze_first_n_layers=args.freeze_layers,
        verbose=True,
    )

    if args.once:
        loss = trainer.train_once()
        print(f"Single-cycle done. Loss: {loss:.4f}")
    else:
        trainer.run(max_cycles=args.max_cycles)


def _add_auto_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("auto-train", help="Continuous World Auto-Training loop")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/auto-trained"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=2,
                   help="Epochs per training cycle.")
    p.add_argument("--interval", type=float, default=300.0,
                   help="Seconds between polling cycles.")
    p.add_argument("--min-pairs", type=int, default=4)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--max-cycles", type=int, default=0,
                   help="Maximum cycles (0 = run forever).")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="Freeze first N backbone layers for faster incremental updates.")
    p.add_argument("--no-plasticity", action="store_true",
                   help="Disable plasticity warm-up after each cycle.")
    p.add_argument("--once", action="store_true",
                   help="Run only one training cycle and exit.")
    p.set_defaults(func=_cmd_auto_train)


# ---------------------------------------------------------------------------
# Root parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroswift",
        description="NeuroSwift — World-class Multimodal Deep Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neuroswift train --epochs 3 --data-dir data/
  python -m neuroswift train-omni --data-dir data/ --epochs 5
  python -m neuroswift chat --model artifacts/neuroswift-tiny
  python -m neuroswift index --source data/ --output artifacts/rag.json
  python -m neuroswift generate --model artifacts/neuroswift-omni --prompt "a sunrise" --modality image
  python -m neuroswift auto-train --data-dir data/ --interval 60 --once
""",
    )
    parser.add_argument("--version", action="version",
                        version="%(prog)s " + _get_version())
    sub = parser.add_subparsers(title="subcommands", dest="subcommand")
    sub.required = True
    _add_train_parser(sub)
    _add_train_omni_parser(sub)
    _add_chat_parser(sub)
    _add_index_parser(sub)
    _add_generate_parser(sub)
    _add_auto_train_parser(sub)
    return parser


def _get_version() -> str:
    try:
        from ._version import __version__
        return __version__
    except Exception:
        return "0.0.0"


def main() -> None:
    _banner()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
