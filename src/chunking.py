"""Module 2: Chunking strategy for retrieval.

The default strategy is structure-aware chunking:
- preserve headings as anchors,
- keep paragraphs together until a token budget is hit,
- optionally create smaller semantic windows when a section is long.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

try:  # Optional dependency for a stronger semantic splitter.
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_path: list[str] = field(default_factory=list)
    source_name: str = "document"
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ChunkConfig:
    max_tokens: int = 350
    overlap_tokens: int = 50
    semantic_window_sentences: int = 3
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class StructureAwareChunker:
    """Convert Markdown-like document text into retrieval-ready chunks."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    def chunk_blocks(self, blocks: Iterable[dict] | Iterable[object], source_name: str = "document") -> list[Chunk]:
        section_path: list[str] = []
        buffer: list[str] = []
        chunks: list[Chunk] = []
        chunk_counter = 1
        page_numbers: list[int] = []

        for block in blocks:
            block_type = getattr(block, "block_type", None) if not isinstance(block, dict) else block.get("block_type")
            text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")
            level = getattr(block, "level", None) if not isinstance(block, dict) else block.get("level")
            page_number = getattr(block, "page_number", None) if not isinstance(block, dict) else block.get("page_number")

            if page_number is not None and page_number not in page_numbers:
                page_numbers.append(page_number)

            if block_type == "heading":
                self._flush_buffer(chunks, buffer, section_path, source_name, chunk_counter, page_numbers)
                if text:
                    section_path = self._update_section_path(section_path, text, level or 1)
                continue

            if block_type == "page_break":
                self._flush_buffer(chunks, buffer, section_path, source_name, chunk_counter, page_numbers)
                continue

            if text:
                buffer.append(text)
                if self._estimate_tokens(" ".join(buffer)) >= self.config.max_tokens:
                    chunk_counter = self._flush_buffer(chunks, buffer, section_path, source_name, chunk_counter, page_numbers)

        self._flush_buffer(chunks, buffer, section_path, source_name, chunk_counter, page_numbers)
        return chunks

    def chunk_markdown(self, markdown: str, source_name: str = "document") -> list[Chunk]:
        """Convenience wrapper for already-cleaned Markdown."""

        blocks = []
        for line in markdown.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                blocks.append({"block_type": "heading", "text": stripped[level:].strip(), "level": level, "page_number": 1})
            elif stripped.startswith("- "):
                blocks.append({"block_type": "bullet", "text": stripped[2:], "page_number": 1})
            else:
                blocks.append({"block_type": "paragraph", "text": stripped, "page_number": 1})
        return self.chunk_blocks(blocks, source_name=source_name)

    def semantic_chunks(self, markdown: str, source_name: str = "document") -> list[Chunk]:
        """Optional advanced path: sentence windows with small overlap."""

        sentences = self._split_sentences(markdown)
        chunks: list[Chunk] = []
        window = self.config.semantic_window_sentences
        overlap = max(1, self.config.semantic_window_sentences - 1)
        start = 0
        chunk_id = 1
        while start < len(sentences):
            end = min(start + window, len(sentences))
            text = " ".join(sentences[start:end]).strip()
            if text:
                chunks.append(Chunk(chunk_id=f"{source_name}-sem-{chunk_id}", text=text, source_name=source_name, metadata={"strategy": "semantic"}))
                chunk_id += 1
            if end == len(sentences):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _flush_buffer(
        self,
        chunks: list[Chunk],
        buffer: list[str],
        section_path: list[str],
        source_name: str,
        chunk_counter: int,
        page_numbers: list[int],
    ) -> int:
        if not buffer:
            return chunk_counter

        text = " ".join(buffer).strip()
        if text:
            for part in self._split_by_token_budget(text):
                if part:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{source_name}-chunk-{chunk_counter}",
                            text=part,
                            section_path=section_path.copy(),
                            source_name=source_name,
                            page_numbers=page_numbers.copy(),
                            metadata={"strategy": "structure", "max_tokens": str(self.config.max_tokens)},
                        )
                    )
                    chunk_counter += 1
        buffer.clear()
        return chunk_counter

    def _update_section_path(self, section_path: list[str], heading_text: str, level: int) -> list[str]:
        trimmed = section_path[: max(level - 1, 0)]
        return trimmed + [heading_text]

    def _split_by_token_budget(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        if len(tokens) <= self.config.max_tokens:
            return [text]

        chunks: list[str] = []
        start = 0
        step = max(self.config.max_tokens - self.config.overlap_tokens, 1)
        while start < len(tokens):
            end = min(start + self.config.max_tokens, len(tokens))
            chunk_text = " ".join(tokens[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end == len(tokens):
                break
            start += step
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        return re.split(r"(?<=[.!?।॥])\s+", cleaned)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _estimate_tokens(self, text: str) -> int:
        return len(self._tokenize(text))


class MultilingualEmbeddingModel:
    """Thin wrapper around sentence-transformers with a graceful fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or ChunkConfig().embedding_model_name
        self._model = SentenceTransformer(self.model_name) if SentenceTransformer is not None else None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            return [self._hash_embed(text) for text in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]

    def _hash_embed(self, text: str, dimension: int = 128) -> list[float]:
        vector = [0.0] * dimension
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
            index = hash(token) % dimension
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]
