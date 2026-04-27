"""Small executable demo for the AI-only prototype.

The demo intentionally avoids any web UI or backend service. It shows the pipeline
as it would appear in an interview: document understanding -> chunking -> retrieval.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import ChunkConfig, StructureAwareChunker
from src.document_understanding import DocumentProcessor, OCRConfig
from src.retrieval import ChunkRetriever


SAMPLE_TEXT = """
Chapter 1: परिचय
Definitions
The textbook explains multilingual knowledge extraction.
- Point 1: Clean the OCR text.
- Point 2: Preserve headings and bullet points.

વિષયવસ્તુ
આ પુસ્તકમાં Gujarati, English, અને Sanskrit mixed content છે.
- મહત્વપૂર્ણ વિચાર 1
- મહત્વપૂર્ણ વિચાર 2
""".strip()


def main() -> None:
    processor = DocumentProcessor(OCRConfig(backend="text"))
    document = processor.process_text(SAMPLE_TEXT, source_name="sample_textbook")

    print("=== MODULE 1: STRUCTURED MARKDOWN ===")
    print(document.markdown)
    print()

    chunker = StructureAwareChunker(ChunkConfig(max_tokens=80, overlap_tokens=20))
    chunks = chunker.chunk_markdown(document.markdown, source_name=document.source_name)

    print("=== MODULE 2: CHUNKS ===")
    for chunk in chunks:
        print(f"[{chunk.chunk_id}] {chunk.section_path} -> {chunk.text[:120]}")
    print()

    retriever = ChunkRetriever()
    retriever.fit(chunks)

    query = "What should we do to clean OCR text?"
    keyword_result = retriever.keyword_search(query)
    semantic_result = retriever.answer(query)

    print("=== MODULE 3: KEYWORD SEARCH ===")
    for score, chunk in zip(keyword_result.scores, keyword_result.retrieved_chunks):
        print(f"{score:.3f} | {chunk.chunk_id} | {chunk.text[:120]}")
    print()

    print("=== MODULE 3: ANSWER ===")
    print(semantic_result.answer)


if __name__ == "__main__":
    main()
