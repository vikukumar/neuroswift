from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CharTokenizer:
    def __init__(self, stoi: dict[str, int], itos: dict[int, str]) -> None:
        self.stoi = dict(stoi)
        self.itos = {int(k): v for k, v in itos.items()}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        text_list = list(texts)
        vocab = sorted(set("\n".join(text_list)))
        stoi = {ch: idx for idx, ch in enumerate(vocab)}
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            raise ValueError(
                "Input contains characters outside the tokenizer vocabulary: "
                + repr(sorted(set(unknown)))
            )
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_payload = {
            "tokenizer_class": "CharTokenizer",
            "tokenizer_type": "character",
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        (save_dir / "tokenizer.json").write_text(
            json.dumps(tokenizer_payload, indent=2),
            encoding="utf-8",
        )
        (save_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "CharTokenizer",
                    "model_max_length": 1024,
                    "clean_up_tokenization_spaces": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, save_directory: str | Path) -> "CharTokenizer":
        save_dir = Path(save_directory)
        tokenizer_payload = json.loads((save_dir / "tokenizer.json").read_text(encoding="utf-8"))
        return cls(stoi=tokenizer_payload["stoi"], itos=tokenizer_payload["itos"])


__all__ = ["CharTokenizer"]
