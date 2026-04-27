"""Query a textbook PDF with Gujlish questions and return verbatim evidence.

Usage examples:
    python scripts/query_data_gujlish.py --query "aama 1 shlok kayo che ?"
    python scripts/query_data_gujlish.py --interactive
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import ChunkConfig, StructureAwareChunker
from src.document_understanding import DocumentProcessor, OCRConfig
from src.retrieval import ChunkRetriever

GUJARATI_DIGITS = "૦૧૨૩૪૫૬૭૮૯"
DEVANAGARI_DIGITS = "०१२३४५६७८९"

NUMBER_WORDS = {
    "ek": 1,
    "one": 1,
    "be": 2,
    "two": 2,
    "tran": 3,
    "three": 3,
    "char": 4,
    "four": 4,
    "panch": 5,
    "five": 5,
    "chh": 6,
    "six": 6,
    "sat": 7,
    "seven": 7,
    "aath": 8,
    "eight": 8,
    "nav": 9,
    "nine": 9,
    "das": 10,
    "ten": 10,
    "pratham": 1,
    "bijo": 2,
    "trijo": 3,
}

GUJLISH_TERM_ALIASES = {
    "aama": "આમાં",
    "ama": "આમાં",
    "kayo": "કયો",
    "kai": "કઈ",
    "che": "છે",
    "shlok": "શ્લોક",
    "slok": "શ્લોક",
    "shloka": "श्लोक",
    "adhyay": "અધ્યાય",
    "adhyaya": "अध्याय",
    "geeta": "ગીતા",
    "gita": "गीता",
}

SHLOK_TERMS = {
    "shlok",
    "slok",
    "shloka",
    "શ્લોક",
    "श्लोक",
    "verse",
}


def normalize_digits(text: str) -> str:
    translated = text
    for idx, digit in enumerate(GUJARATI_DIGITS):
        translated = translated.replace(digit, str(idx))
    for idx, digit in enumerate(DEVANAGARI_DIGITS):
        translated = translated.replace(digit, str(idx))
    return translated


def extract_requested_number(query: str) -> int | None:
    normalized = normalize_digits(query.lower())
    match = re.search(r"\b(\d{1,3})\b", normalized)
    if match:
        return int(match.group(1))

    tokens = re.findall(r"[a-zA-Z]+", normalized)
    for token in tokens:
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    return None


def expand_gujlish_query(query: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", query.lower(), flags=re.UNICODE)
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        alias = GUJLISH_TERM_ALIASES.get(token)
        if alias:
            expanded.append(alias)
    return " ".join(expanded)


def line_stream(markdown_text: str) -> list[str]:
    lines = [line.strip() for line in markdown_text.splitlines()]
    return [line for line in lines if line and line != "---"]


def contains_shlok_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in SHLOK_TERMS)


def find_exact_shlok_lines(lines: Iterable[str], requested_number: int | None) -> list[str]:
    if requested_number is None:
        return []

    hits: list[str] = []
    for line in lines:
        normalized = normalize_digits(line)
        if not contains_shlok_term(normalized):
            continue

        has_number = bool(re.search(rf"\b{requested_number}\b", normalized))
        if has_number:
            hits.append(line)

    return hits


def retrieve_verbatim_snippets(markdown_text: str, query: str, top_k: int = 3) -> list[str]:
    chunker = StructureAwareChunker(ChunkConfig(max_tokens=180, overlap_tokens=30))
    chunks = chunker.chunk_markdown(markdown_text, source_name="data_file")
    if not chunks:
        return []

    retriever = ChunkRetriever()
    retriever.fit(chunks)

    expanded_query = expand_gujlish_query(query)
    result = retriever.semantic_search(expanded_query, top_k=top_k)

    snippets: list[str] = []
    for chunk in result.retrieved_chunks:
        snippets.append(chunk.text)
    return snippets


def load_pdf_markdown(pdf_path: Path, ocr_backend: str) -> str:
    direct_text = extract_text_directly_from_pdf(pdf_path)
    if direct_text.strip() and not looks_garbled(direct_text):
        return direct_text

    if ocr_backend == "tesseract" and find_tesseract_binary() is None:
        if direct_text.strip():
            raise RuntimeError(
                "PDF text layer appears garbled and Tesseract is not installed. "
                "Install Tesseract OCR to extract clean Gujarati/Sanskrit text."
            )
        raise RuntimeError(
            "No usable text was found and Tesseract is not installed. "
            "Install Tesseract OCR and try again."
        )

    processor = DocumentProcessor(
        OCRConfig(
            backend=ocr_backend,
            language_hints=["eng", "guj", "san"],
            render_dpi=220,
        )
    )
    result = processor.process_file(pdf_path)
    if result.markdown.strip():
        return result.markdown

    if direct_text.strip():
        # Keep a last-resort path to at least return source text when OCR fails.
        return direct_text
    return ""


def extract_text_directly_from_pdf(pdf_path: Path) -> str:
    """Read embedded PDF text before trying OCR.

    Many scanned textbooks are already OCR-ed and contain a text layer.
    This path keeps answers closer to the original source text.
    """

    if fitz is None:
        return ""

    pages: list[str] = []
    document = fitz.open(pdf_path)
    for page_index in range(document.page_count):
        page_text = document.load_page(page_index).get_text("text")
        if page_text and page_text.strip():
            pages.append(page_text.strip())
    document.close()
    return "\n\n".join(pages).strip()


def looks_garbled(text: str) -> bool:
    """Heuristic check for broken font-encoded PDF text layers."""

    if not text.strip():
        return True

    weird_chars = re.findall(r"[Ðð×ØÆ¤™¢£¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ƒ‚„…†‡ˆ‰‹›œžŸ]", text)
    replacement_like = text.count("�")
    total_chars = max(len(text), 1)
    weird_ratio = (len(weird_chars) + replacement_like) / total_chars

    # Many Latin-only glyph artifacts from bad font mapping push this ratio high.
    return weird_ratio > 0.03


def find_tesseract_binary() -> str | None:
    in_path = shutil.which("tesseract")
    if in_path:
        return in_path

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def detect_data_file(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(f"No PDF file found in: {data_dir}")
    return candidates[0]


def answer_query(markdown_text: str, query: str) -> dict[str, list[str] | str]:
    lines = line_stream(markdown_text)
    number = extract_requested_number(query)

    if contains_shlok_term(query.lower()):
        shlok_lines = find_exact_shlok_lines(lines, number)
        if shlok_lines:
            return {
                "mode": "exact-shlok-line",
                "responses": shlok_lines,
            }

    snippets = retrieve_verbatim_snippets(markdown_text, query, top_k=3)
    if snippets:
        return {
            "mode": "semantic-verbatim-snippet",
            "responses": snippets,
        }

    return {
        "mode": "not-found",
        "responses": ["No matching text found in the file."],
    }


def print_answer(answer: dict[str, list[str] | str]) -> None:
    print(f"Mode: {answer['mode']}")
    print("Output from file:")
    for item in answer["responses"]:
        print("-" * 40)
        print(item)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Query the textbook in Gujlish and return file-grounded text.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Directory that contains the PDF file")
    parser.add_argument("--query", type=str, default="", help="Single query to run")
    parser.add_argument("--ocr-backend", type=str, default="tesseract", choices=["tesseract", "easyocr", "paddleocr"], help="OCR engine")
    parser.add_argument("--interactive", action="store_true", help="Start interactive query mode")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_file = detect_data_file(args.data_dir)
    print(f"Using file: {data_file.name}")
    markdown_text = load_pdf_markdown(data_file, ocr_backend=args.ocr_backend)

    if not markdown_text.strip():
        raise RuntimeError(
            "Could not extract text from the PDF. Ensure OCR dependencies are installed and the scan is readable."
        )

    if args.query:
        print_answer(answer_query(markdown_text, args.query))
        return

    if args.interactive:
        print("Interactive mode started. Type 'exit' to quit.")
        while True:
            user_query = input("\nAsk in Gujlish> ").strip()
            if not user_query:
                continue
            if user_query.lower() in {"exit", "quit"}:
                break
            print_answer(answer_query(markdown_text, user_query))
        return

    parser.error("Provide --query or use --interactive")


if __name__ == "__main__":
    main()
