from __future__ import annotations

import json
import re
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


# Multilingual Word Tokenizer supporting Devanagari, English, and more.
# Uses \w (alphanumeric) with Unicode support if available.
TOKEN_PATTERN = re.compile(r"[\w]+(?:'[\w]+)?|[^\w\s]", re.IGNORECASE | re.UNICODE)


class WordTokenizer:
    pad_token = "<pad>"
    unk_token = "<unk>"
    eos_token = "<eos>"

    def __init__(self, stoi: dict[str, int], itos: dict[int, str], lowercase: bool = True) -> None:
        self.stoi = dict(stoi)
        self.itos = {int(k): v for k, v in itos.items()}
        self.lowercase = lowercase

    @classmethod
    def from_texts(cls, texts: Iterable[str], lowercase: bool = True) -> "WordTokenizer":
        text_list = list(texts)
        specials = [cls.pad_token, cls.unk_token, cls.eos_token]
        vocab = list(specials)
        unique_tokens = set()
        for text in text_list:
            normalized = text.lower() if lowercase else text
            # Captures Hindi/Sanskrit words correctly via Unicode-aware pattern
            tokens = TOKEN_PATTERN.findall(normalized)
            unique_tokens.update(tokens)

        for token in sorted(unique_tokens):
            if token not in vocab:
                vocab.append(token)

        stoi = {token: idx for idx, token in enumerate(vocab)}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos, lowercase=lowercase)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def eos_token_id(self) -> int:
        return self.stoi[self.eos_token]

    def _normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def tokenize(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        return TOKEN_PATTERN.findall(normalized)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        tokens = self.tokenize(text)
        ids = [self.stoi.get(token, self.stoi[self.unk_token]) for token in tokens]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode_corpus(self, texts: Iterable[str]) -> list[int]:
        stream: list[int] = []
        for text in texts:
            stream.extend(self.encode(text, add_eos=True))
        return stream

    def decode(self, ids: list[int]) -> str:
        words: list[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            if token == self.pad_token:
                continue
            if token == self.eos_token:
                if words and words[-1] != "\n":
                    words.append("\n")
                continue
            if token == self.unk_token:
                token = "?"

            if not words or words[-1] == "\n":
                words.append(token)
            elif token in {".", ",", "!", "?", ";", ":", ")", "]", "}"}:
                words[-1] = words[-1] + token
            elif words[-1] in {"(", "[", "{"}:
                words[-1] = words[-1] + token
            else:
                words.append(" " + token)

        return "".join(words).strip()

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_payload = {
            "tokenizer_class": "WordTokenizer",
            "tokenizer_type": "word",
            "lowercase": self.lowercase,
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "special_tokens": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "eos_token": self.eos_token,
            },
        }
        (save_dir / "tokenizer.json").write_text(
            json.dumps(tokenizer_payload, indent=2),
            encoding="utf-8",
        )
        (save_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "WordTokenizer",
                    "model_max_length": 1024,
                    "lowercase": self.lowercase,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, save_directory: str | Path) -> "WordTokenizer":
        save_dir = Path(save_directory)
        tokenizer_payload = json.loads((save_dir / "tokenizer.json").read_text(encoding="utf-8"))
        return cls(
            stoi=tokenizer_payload["stoi"],
            itos=tokenizer_payload["itos"],
            lowercase=tokenizer_payload.get("lowercase", True),
        )


def load_tokenizer(save_directory: str | Path) -> CharTokenizer | WordTokenizer:
    save_dir = Path(save_directory)
    tokenizer_payload = json.loads((save_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_class = tokenizer_payload.get("tokenizer_class", "CharTokenizer")
    if tokenizer_class == "WordTokenizer":
        return WordTokenizer.from_pretrained(save_dir)
    return CharTokenizer.from_pretrained(save_dir)


__all__ = ["CharTokenizer", "WordTokenizer", "load_tokenizer"]
