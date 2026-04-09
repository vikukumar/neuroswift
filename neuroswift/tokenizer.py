from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Iterable, Any

logger = logging.getLogger("NeuroSwift.Train")


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

    def get_vocab(self) -> dict[str, Any]:
        return {"stoi": self.stoi, "itos": self.itos, "type": "char"}

    def load_vocab(self, vocab: dict[str, Any]) -> None:
        self.stoi = dict(vocab["stoi"])
        self.itos = {int(k): v for k, v in vocab["itos"].items()}

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
    def from_texts(
        cls, 
        texts: Iterable[str], 
        lowercase: bool = True, 
        max_vocab: int = 12000, 
        min_freq: int = 2
    ) -> "WordTokenizer":
        # Point 3: Normalize Tokenizer (lowercase + consistent build)
        text_list = list(texts)
        specials = [cls.pad_token, cls.unk_token, cls.eos_token]
        
        # Point 1 & 3: Filter rare tokens and target 8k-16k vocab
        from collections import Counter
        counts = Counter()
        for text in text_list:
            normalized = text.lower() if lowercase else text
            tokens = TOKEN_PATTERN.findall(normalized)
            counts.update(tokens)
            
        # Point 3: Remove rare tokens (appearing only once)
        valid_tokens = [t for t, freq in counts.items() if freq >= min_freq]
        
        # Point 1: Target 8k-16k (defaulting to 12,000 for stability)
        # Sort by frequency, then take top N
        sorted_tokens = sorted(valid_tokens, key=lambda x: counts[x], reverse=True)
        final_tokens = sorted_tokens[:max_vocab - len(specials)]
        
        # Re-sort alphabetically for consistent build across identical data
        vocab = specials + sorted(final_tokens)

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

    def get_vocab(self) -> dict[str, Any]:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "type": "word",
            "lowercase": self.lowercase,
            "special_tokens": {
                "pad": self.pad_token,
                "unk": self.unk_token,
                "eos": self.eos_token,
            }
        }

    def load_vocab(self, vocab: dict[str, Any]) -> None:
        self.stoi = dict(vocab["stoi"])
        self.itos = {int(k): v for k, v in vocab["itos"].items()}
        self.lowercase = vocab.get("lowercase", True)
        if "special_tokens" in vocab:
            s = vocab["special_tokens"]
            self.pad_token = s.get("pad", "<pad>")
            self.unk_token = s.get("unk", "<unk>")
            self.eos_token = s.get("eos", "<eos>")

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


class HybridTokenizer:
    pad_token = "<pad>"
    unk_token = "<unk>"
    eos_token = "<eos>"
    lowercase = True

    def __init__(self, sp_model=None, word_tokenizer=None, spm_model_path=None):
        self.sp_model = sp_model
        self.word_tokenizer = word_tokenizer
        self.spm_model_path = spm_model_path

    @classmethod
    def from_texts(
        cls, 
        texts: Iterable[str], 
        vocab_size: int = 12000, 
        output_dir: str | Path | None = None
    ) -> "HybridTokenizer":
        if output_dir is None:
            output_dir = Path("artifacts/neuroswift-tiny")
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        spm_model_path = output_dir / "tokenizer.model"

        # If SPM model exists, load it
        if spm_model_path.exists():
            try:
                import sentencepiece as spm
                sp_model = spm.SentencePieceProcessor()
                # Use str() for Windows path compatibility
                sp_model.load(str(spm_model_path))
                obj = cls(sp_model=sp_model, spm_model_path=str(spm_model_path))
                logger.info("Tokenizer: SentencePiece (BPE) model loaded")
                # Persist metadata (tokenizer.json)
                obj.save_pretrained(output_dir)
                return obj
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece model from {spm_model_path}: {e}")

        # Fallback to WordTokenizer if no SPM model or loading failed
        logger.info("Tokenizer: WordTokenizer fallback")
        word_tok = WordTokenizer.from_texts(texts, max_vocab=vocab_size)
        obj = cls(word_tokenizer=word_tok)
        obj.save_pretrained(output_dir)
        return obj

    @property
    def vocab_size(self) -> int:
        if self.sp_model:
            return self.sp_model.get_piece_size()
        return self.word_tokenizer.vocab_size

    @property
    def stoi(self) -> dict[str, int]:
        if self.word_tokenizer:
            return self.word_tokenizer.stoi
        return {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.get_piece_size())}

    @property
    def itos(self) -> dict[int, str]:
        if self.word_tokenizer:
            return self.word_tokenizer.itos
        return {i: self.sp_model.id_to_piece(i) for i in range(self.sp_model.get_piece_size())}

    @property
    def eos_token_id(self) -> int:
        if self.sp_model:
            return self.sp_model.eos_id()
        return self.word_tokenizer.eos_token_id
        
    @property
    def pad_token_id(self) -> int:
        if self.sp_model:
            return self.sp_model.pad_id()
        return self.word_tokenizer.stoi.get(self.pad_token, 0)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        if self.sp_model:
            ids = self.sp_model.encode(text)
            if add_eos:
                ids.append(self.sp_model.eos_id())
            return ids
        return self.word_tokenizer.encode(text, add_eos=add_eos)

    def decode(self, ids: list[int]) -> str:
        if self.sp_model:
            valid_ids = []
            for i in ids:
                if i in (self.sp_model.pad_id(), self.sp_model.unk_id()):
                    continue
                piece = self.sp_model.id_to_piece(i)
                # v29: Enhanced defensive decoding against RedPajama pollution
                clean_piece = piece.replace(" ", "").strip()
                if not clean_piece:
                    continue
                # Reject standalone numbers > 2 digits or symbols that look like logs
                if clean_piece.isnumeric() and len(clean_piece) > 2:
                    continue
                if any(x in clean_piece for x in ("/", "\\", "0x", "de_middle", "en_middle")):
                    continue
                valid_ids.append(i)
            text = self.sp_model.decode(valid_ids)
            return text
        else:
            words = []
            for i in ids:
                token = self.word_tokenizer.itos.get(int(i), "")
                if token in (self.pad_token, self.unk_token):
                    continue
                # v29: Enhanced defensive decoding
                if token.isnumeric() and len(token) > 2:
                    continue
                if any(x in token for x in ("/", "\\", "0x", "de_middle", "en_middle")):
                    continue
                words.append(token)
            
            res = []
            for w in words:
                if w == self.eos_token:
                    res.append("\n")
                elif w in {".", ",", "!", "?", ";", ":", ")", "]", "}"}:
                    if res and res[-1] != "\n":
                        res[-1] = res[-1] + w
                    else:
                        res.append(w)
                else:
                    if not res or res[-1] == "\n":
                        res.append(w)
                    else:
                        res.append(" " + w)
            return "".join(res).strip()

    def get_vocab(self) -> dict[str, Any]:
        if self.sp_model:
            return {
                "stoi": self.stoi,
                "itos": self.itos,
                "type": "bpe",
                "lowercase": self.lowercase,
                "special_tokens": {
                    "pad": self.pad_token,
                    "unk": self.unk_token,
                    "eos": self.eos_token,
                }
            }
        return self.word_tokenizer.get_vocab()

    def load_vocab(self, vocab: dict[str, Any]) -> None:
        if self.word_tokenizer:
            self.word_tokenizer.load_vocab(vocab)

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.sp_model:
            tokenizer_payload = {
                "tokenizer_class": "HybridTokenizer",
                "tokenizer_type": "bpe",
                "vocab_size": self.vocab_size,
                "stoi": self.stoi,
                "itos": {str(k): v for k, v in self.itos.items()},
                "spm_model_path": self.spm_model_path,
            }
            (save_dir / "tokenizer.json").write_text(
                json.dumps(tokenizer_payload, indent=2), encoding="utf-8"
            )
            (save_dir / "tokenizer_vocab.json").write_text(
                json.dumps(self.stoi, indent=2), encoding="utf-8"
            )
        else:
            self.word_tokenizer.save_pretrained(save_directory)
            (save_dir / "tokenizer_vocab.json").write_text(
                json.dumps(self.word_tokenizer.stoi, indent=2), encoding="utf-8"
            )

    @classmethod
    def from_pretrained(cls, save_directory: str | Path) -> "HybridTokenizer":
        save_dir = Path(save_directory)
        tokenizer_json = save_dir / "tokenizer.json"
        
        if not tokenizer_json.exists():
            # Fallback to WordTokenizer if metadata is missing
            word_tok = WordTokenizer.from_pretrained(save_dir)
            return cls(word_tokenizer=word_tok)

        tokenizer_payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        if tokenizer_payload.get("tokenizer_class") == "HybridTokenizer" and "spm_model_path" in tokenizer_payload:
            import sentencepiece as spm
            spm_path = tokenizer_payload["spm_model_path"]
            if spm_path and Path(spm_path).exists():
                sp_model = spm.SentencePieceProcessor(model_file=spm_path)
                return cls(sp_model=sp_model, spm_model_path=spm_path)
            else:
                logger.warning(f"SPM model file missing at {spm_path}. Falling back to Word.")
                word_tok = WordTokenizer.from_pretrained(save_dir)
                return cls(word_tokenizer=word_tok)
        else:
            word_tok = WordTokenizer.from_pretrained(save_dir)
            return cls(word_tokenizer=word_tok)


def load_tokenizer(save_directory: str | Path) -> Any:
    save_dir = Path(save_directory)
    t_json = save_dir / "tokenizer.json"
    
    if not t_json.exists():
        # Minimal fallback
        from neuroswift.tokenizer import WordTokenizer
        try:
            return WordTokenizer.from_pretrained(save_dir)
        except:
            from neuroswift.tokenizer import CharTokenizer
            return CharTokenizer.from_pretrained(save_dir)

    tokenizer_payload = json.loads(t_json.read_text(encoding="utf-8"))
    tokenizer_class = tokenizer_payload.get("tokenizer_class", "CharTokenizer")
    if tokenizer_class == "HybridTokenizer":
        return HybridTokenizer.from_pretrained(save_dir)
    if tokenizer_class == "WordTokenizer":
        return WordTokenizer.from_pretrained(save_dir)
    return CharTokenizer.from_pretrained(save_dir)


__all__ = ["CharTokenizer", "WordTokenizer", "HybridTokenizer", "load_tokenizer"]
