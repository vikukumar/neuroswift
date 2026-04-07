from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .ingest import DatasetFolderReader, MultimodalSample
from .data_pipeline import WebScraper
import datetime


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


@dataclass
class RAGDocument:
    doc_id: str
    text: str
    source: str
    modality: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    score: float
    document: RAGDocument
    matched_terms: tuple[str, ...] = ()


class FuturePredictor:
    """God-level temporal awareness for accurate future prediction."""
    @staticmethod
    def get_context() -> str:
        now = datetime.datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}. Use this for real-time awareness and future-oriented predictions."


class NeuroSwiftRAG:
    """
    Small CPU-friendly lexical retriever.

    The index stays lightweight by using BM25-style term statistics instead of
    dense embeddings, which keeps lookup fast and easy to ship alongside the
    core NeuroSwift model.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        lowercase: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
        max_text_chars: int = 6000,
    ) -> None:
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.k1 = k1
        self.b = b
        self.max_text_chars = max_text_chars

        self.documents: list[RAGDocument] = []
        self._doc_tokens: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._doc_frequencies: Counter[str] = Counter()
        self._avg_doc_len: float = 1.0

    def __len__(self) -> int:
        return len(self.documents)

    def _normalize(self, text: str) -> str:
        normalized = text.strip()
        return normalized.lower() if self.lowercase else normalized

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is not None and hasattr(self.tokenizer, "tokenize"):
            return list(self.tokenizer.tokenize(text))
        return TOKEN_PATTERN.findall(self._normalize(text))

    @staticmethod
    def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
            else:
                safe[key] = str(value)
        return safe

    def _recompute_average_length(self) -> None:
        if not self._doc_tokens:
            self._avg_doc_len = 1.0
            return
        total = sum(len(tokens) for tokens in self._doc_tokens)
        self._avg_doc_len = max(total / len(self._doc_tokens), 1.0)

    def add_document(
        self,
        text: str,
        source: str,
        modality: str = "text",
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> None:
        cleaned_text = text.strip()
        if not cleaned_text:
            return

        truncated_text = cleaned_text[: self.max_text_chars]
        tokens = self._tokenize(truncated_text)
        if not tokens:
            return

        document = RAGDocument(
            doc_id=doc_id or f"doc_{len(self.documents)}",
            text=truncated_text,
            source=source,
            modality=modality,
            metadata=self._json_safe_metadata(metadata or {}),
        )
        term_frequency = Counter(tokens)

        self.documents.append(document)
        self._doc_tokens.append(tokens)
        self._term_frequencies.append(term_frequency)
        self._doc_frequencies.update(set(tokens))
        self._recompute_average_length()

    def add_samples(self, samples: Iterable[MultimodalSample]) -> None:
        for sample_idx, sample in enumerate(samples):
            self.add_document(
                text=sample.to_rag_text(),
                source=sample.source,
                modality=sample.modality,
                metadata=sample.metadata,
                doc_id=f"{sample.modality}_{sample_idx}",
            )

    def add_instruction_pairs(
        self,
        pairs: Iterable[dict[str, str]],
        source: str = "instruction_memory",
    ) -> None:
        for pair_idx, pair in enumerate(pairs):
            prompt = str(pair.get("prompt", "")).strip()
            response = str(pair.get("response", "")).strip()
            if not prompt or not response:
                continue
            self.add_document(
                text=f"user: {prompt}\nassistant: {response}",
                source=source,
                modality="qa",
                metadata={"prompt": prompt, "response": response},
                doc_id=f"qa_{pair_idx}",
            )

    @classmethod
    def from_folder(
        cls,
        root: str | Path,
        *,
        reader: Optional[DatasetFolderReader] = None,
        tokenizer: Optional[Any] = None,
        fetch_urls: bool = False,
    ) -> "NeuroSwiftRAG":
        retriever = cls(tokenizer=tokenizer)
        folder_reader = reader or DatasetFolderReader(fetch_urls=fetch_urls)
        retriever.add_samples(folder_reader.read_folder(root))
        return retriever

    def query(
        self,
        query: str,
        top_k: int = 4,
        preferred_modality: Optional[str] = None,
    ) -> list[RetrievalHit]:
        query = query.strip()
        if not query or not self.documents or top_k <= 0:
            return []

        query_tokens = self._tokenize(query)
        content_tokens = [
            token
            for token in query_tokens
            if token not in STOPWORDS and any(char.isalnum() for char in token) and len(token) > 1
        ]
        if content_tokens:
            query_tokens = content_tokens
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        query_lower = self._normalize(query)
        total_docs = max(len(self.documents), 1)
        hits: list[RetrievalHit] = []

        for document, doc_tokens, term_frequency in zip(
            self.documents,
            self._doc_tokens,
            self._term_frequencies,
        ):
            score = 0.0
            matched_terms: list[str] = []
            doc_length = max(len(doc_tokens), 1)

            for term, term_count in query_counts.items():
                frequency = term_frequency.get(term, 0)
                if frequency <= 0:
                    continue

                matched_terms.append(term)
                doc_freq = self._doc_frequencies.get(term, 1)
                idf = math.log(1.0 + ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5)))
                denom = frequency + self.k1 * (
                    1.0 - self.b + self.b * (doc_length / self._avg_doc_len)
                )
                score += idf * (((self.k1 + 1.0) * frequency) / max(denom, 1e-9)) * term_count

            normalized_text = self._normalize(document.text)
            prompt_memory = self._normalize(str(document.metadata.get("prompt", "")))

            if prompt_memory and prompt_memory == query_lower:
                score += 12.0
            elif normalized_text == query_lower:
                score += 8.0
            elif query_lower and query_lower in normalized_text:
                score += 3.0

            if preferred_modality and document.modality == preferred_modality:
                score += 0.5

            if score <= 0.0:
                continue

            hits.append(
                RetrievalHit(
                    score=score,
                    document=document,
                    matched_terms=tuple(sorted(set(matched_terms))),
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def web_query(self, query: str, top_k: int = 2) -> list[RetrievalHit]:
        """
        God-level web search fallback. 
        Scrapes search results for real-time grounding.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Initiating Web-Search RAG for: {query}")
        
        # Simulated DuckDuckGo scraping (generic pattern)
        search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        raw_text = WebScraper.scrape(search_url)
        
        if not raw_text:
            return []
            
        # Create a temporary document from the web result
        web_doc = RAGDocument(
            doc_id=f"web_{hash(query)}",
            text=raw_text[:2000], 
            source=search_url,
            modality="web",
            metadata={"timestamp": str(datetime.datetime.now())}
        )
        
        return [RetrievalHit(score=10.0, document=web_doc)]

    def format_hits(self, hits: Iterable[RetrievalHit], max_chars: int = 3200) -> str:
        remaining = max_chars
        sections: list[str] = []
        for hit_idx, hit in enumerate(hits, start=1):
            if remaining <= 0:
                break

            snippet = hit.document.text.replace("\r", " ").strip()
            snippet = " ".join(snippet.split())
            snippet = snippet[: min(remaining, 1000)]
            # Clean formatting for small models (v25)
            # Remove 'modality=' and 'source=' etc. to prevent the model from memorizing metadata
            source_name = Path(hit.document.source).name
            section = f"[Context from {source_name}]\n{snippet}"
            
            sections.append(section)
            remaining -= len(section)
        return "\n\n".join(sections)

    def save(self, path: str | Path) -> None:
        payload = {
            "documents": [asdict(document) for document in self.documents],
            "lowercase": self.lowercase,
            "k1": self.k1,
            "b": self.b,
            "max_text_chars": self.max_text_chars,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        tokenizer: Optional[Any] = None,
    ) -> "NeuroSwiftRAG":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        retriever = cls(
            tokenizer=tokenizer,
            lowercase=payload.get("lowercase", True),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
            max_text_chars=int(payload.get("max_text_chars", 6000)),
        )
        for document_payload in payload.get("documents", []):
            retriever.add_document(
                text=str(document_payload["text"]),
                source=str(document_payload["source"]),
                modality=str(document_payload.get("modality", "text")),
                metadata=document_payload.get("metadata", {}),
                doc_id=str(document_payload.get("doc_id", f"doc_{len(retriever.documents)}")),
            )
        return retriever


__all__ = ["NeuroSwiftRAG", "RAGDocument", "RetrievalHit"]
