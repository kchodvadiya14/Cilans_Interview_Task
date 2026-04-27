"""Module 3: Search and question answering over the chunks.

This module keeps the first prototype simple:
- semantic retrieval using multilingual embeddings,
- keyword retrieval as a cheap and reliable baseline,
- optional LLM answer synthesis when a client is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Callable

import numpy as np

try:  # Optional acceleration if FAISS is available.
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None

from .chunking import Chunk, MultilingualEmbeddingModel


@dataclass
class RetrievalResult:
    query: str
    retrieved_chunks: list[Chunk]
    scores: list[float] = field(default_factory=list)
    answer: str | None = None
    method: str = "semantic"


class ChunkRetriever:
    """A small vector index with a keyword fallback."""

    def __init__(self, embedding_model: MultilingualEmbeddingModel | None = None, use_faiss: bool = True) -> None:
        self.embedding_model = embedding_model or MultilingualEmbeddingModel()
        self.use_faiss = use_faiss and faiss is not None
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.index = None

    def fit(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        vectors = np.asarray(self.embedding_model.embed(texts), dtype="float32")
        if vectors.size == 0:
            self.embeddings = None
            self.index = None
            return

        vectors = self._l2_normalize(vectors)
        self.embeddings = vectors
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)

    def semantic_search(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self.chunks:
            return RetrievalResult(query=query, retrieved_chunks=[], scores=[], method="semantic")

        query_vector = np.asarray(self.embedding_model.embed([query])[0], dtype="float32")[None, :]
        query_vector = self._l2_normalize(query_vector)

        if self.index is not None:
            scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
            ranked = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        else:
            assert self.embeddings is not None
            scores = (self.embeddings @ query_vector.T).reshape(-1)
            ranked = sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)[:top_k]

        retrieved_chunks = [self.chunks[index] for index, _ in ranked]
        retrieval_scores = [score for _, score in ranked]
        return RetrievalResult(query=query, retrieved_chunks=retrieved_chunks, scores=retrieval_scores, method="semantic")

    def keyword_search(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self.chunks:
            return RetrievalResult(query=query, retrieved_chunks=[], scores=[], method="keyword")

        query_terms = {term for term in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if len(term) > 2}
        scored: list[tuple[int, float]] = []
        for index, chunk in enumerate(self.chunks):
            chunk_terms = re.findall(r"\w+", chunk.text.lower(), flags=re.UNICODE)
            overlap = sum(1 for term in chunk_terms if term in query_terms)
            score = overlap / max(len(query_terms), 1)
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:top_k]
        return RetrievalResult(
            query=query,
            retrieved_chunks=[self.chunks[index] for index, _ in selected],
            scores=[score for _, score in selected],
            method="keyword",
        )

    def answer(
        self,
        query: str,
        top_k: int = 4,
        llm_client: Callable[[str], str] | None = None,
    ) -> RetrievalResult:
        result = self.semantic_search(query, top_k=top_k)
        context = self._build_context(result.retrieved_chunks)

        if llm_client is not None:
            prompt = self._build_prompt(query, context)
            result.answer = llm_client(prompt)
            result.method = "semantic+llm"
            return result

        result.answer = self._extractive_answer(query, result.retrieved_chunks)
        result.method = "semantic+extractive"
        return result

    def _build_context(self, chunks: list[Chunk]) -> str:
        return "\n\n".join(chunk.text for chunk in chunks)

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "You are a careful multilingual textbook assistant.\n"
            "Answer only from the provided context. If the answer is not present, say so.\n\n"
            f"Question: {query}\n\nContext:\n{context}\n"
        )

    def _extractive_answer(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "I could not find relevant passages in the document."

        query_terms = {term for term in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if len(term) > 2}
        scored_sentences: list[tuple[float, str]] = []
        for chunk in chunks:
            for sentence in self._split_sentences(chunk.text):
                sentence_terms = set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))
                overlap = len(query_terms & sentence_terms)
                if overlap:
                    scored_sentences.append((overlap / max(math.sqrt(len(sentence_terms)), 1.0), sentence.strip()))

        if not scored_sentences:
            return chunks[0].text[:500]

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in scored_sentences[:3]]
        return " ".join(selected)

    def _split_sentences(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        return re.split(r"(?<=[.!?।॥])\s+", cleaned)

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
