"""Query a textbook PDF with Gujlish questions and return verbatim evidence.

Usage examples:
    python scripts/query_data_gujlish.py --query "aama 1 shlok kayo che ?"
    python scripts/query_data_gujlish.py --interactive
"""

from __future__ import annotations

from argparse import ArgumentParser
import importlib
from pathlib import Path
import os
import re
import shutil
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

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

ORDINAL_WORDS = {
    "pehlo": 1,
    "phelo": 1,
    "pratham": 1,
    "prathama": 1,
    "bijo": 2,
    "bijo": 2,
    "bijo": 2,
    "trijo": 3,
    "chotho": 4,
    "panchmo": 5,
    "chhatho": 6,
    "satmo": 7,
    "aathmo": 8,
    "navmo": 9,
    "dasmo": 10,
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

CHAPTER_TERMS = {
    "adhyay",
    "adhyaya",
    "chapter",
    "પ્રકરણ",
    "અધ્યાય",
}

QUESTION_INTENT_PATTERNS = {
    "chapter": [
        r"\badyay\b",
        r"\badhyay\b",
        r"\badhyaya\b",
        r"\badyaya\b",
        r"\bchapter\b",
        r"પ્રકરણ",
        r"અધ્યાય",
    ],
    "meaning": [
        r"su\s+kehva\s+mange\s+che",
        r"shu\s+kehva\s+mange\s+che",
        r"શું\s+કહેવા\s+માટે",
        r"શું\s+કહેવા\s+માંગે\s+છે",
        r"meaning",
        r"matlab",
    ],
}


def _load_fitz_module():
    try:
        return importlib.import_module("fitz")
    except Exception:  # pragma: no cover - optional dependency
        return None


def _load_gemini_module():
    try:
        return importlib.import_module("google.generativeai")
    except Exception:  # pragma: no cover - optional dependency
        return None


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


def extract_requested_chapter_number(query: str) -> int | None:
    normalized = normalize_digits(query.lower())
    match = re.search(r"\b(?:chapter|adyay|adhyay|adhyaya|adyaya|અધ્યાય|પ્રકરણ)\s*(\d{1,3})\b", normalized)
    if match:
        return int(match.group(1))

    tokens = re.findall(r"[a-zA-Z]+", normalized)
    for token in tokens:
        if token in ORDINAL_WORDS:
            return ORDINAL_WORDS[token]
    return None


def is_chapter_query(query: str) -> bool:
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in QUESTION_INTENT_PATTERNS["chapter"]) \
        and any(re.search(pattern, lowered) for pattern in QUESTION_INTENT_PATTERNS["meaning"])


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


def find_exact_shlok_blocks(lines: list[str], requested_number: int | None) -> list[str]:
    if requested_number is None:
        return []

    blocks: list[str] = []
    for index, line in enumerate(lines):
        normalized = normalize_digits(line)
        if not contains_shlok_term(normalized):
            continue
        if not re.search(rf"\b{requested_number}\b", normalized):
            continue

        buffer = [line]
        next_index = index + 1
        while next_index < len(lines):
            candidate = lines[next_index]
            candidate_norm = normalize_digits(candidate)
            # Stop at next shlok marker or clear section break.
            if contains_shlok_term(candidate_norm) and re.search(r"\b\d+\b", candidate_norm):
                break
            if candidate.startswith("#"):
                break
            buffer.append(candidate)
            # Keep block compact for CLI readability.
            if len(buffer) >= 4:
                break
            next_index += 1

        blocks.append("\n".join(buffer))

    return blocks


def find_exact_chapter_block(lines: list[str], requested_number: int | None) -> list[str]:
    if requested_number is None:
        return []

    blocks: list[str] = []
    for index, line in enumerate(lines):
        normalized = normalize_digits(line)
        lower = normalized.lower()
        if not any(term in lower for term in CHAPTER_TERMS):
            continue
        if not re.search(rf"\b{requested_number}\b", normalized):
            continue

        buffer = [line]
        next_index = index + 1
        while next_index < len(lines):
            candidate = lines[next_index]
            candidate_norm = normalize_digits(candidate)
            candidate_lower = candidate_norm.lower()
            if any(term in candidate_lower for term in CHAPTER_TERMS) and re.search(r"\b\d+\b", candidate_norm):
                break
            buffer.append(candidate)
            if len(buffer) >= 8:
                break
            next_index += 1

        blocks.append("\n".join(buffer))

    return blocks


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


def keyword_line_search(markdown_text: str, query: str, top_k: int = 5) -> list[str]:
    lines = line_stream(markdown_text)
    if not lines:
        return []

    expanded_query = expand_gujlish_query(query)
    query_terms = set(re.findall(r"\w+", expanded_query.lower(), flags=re.UNICODE))
    query_terms = {term for term in query_terms if len(term) > 1}
    if not query_terms:
        return []

    scored: list[tuple[float, str]] = []
    for line in lines:
        line_terms = set(re.findall(r"\w+", line.lower(), flags=re.UNICODE))
        overlap = len(query_terms & line_terms)
        if overlap == 0:
            continue
        score = overlap / max(len(query_terms), 1)
        scored.append((score, line))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [line for _, line in scored[:top_k]]


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

    if direct_text.strip() and not looks_garbled(direct_text):
        # Keep a last-resort path only when the embedded layer is clean.
        return direct_text
    return ""


def extract_text_directly_from_pdf(pdf_path: Path) -> str:
    """Read embedded PDF text before trying OCR.

    Many scanned textbooks are already OCR-ed and contain a text layer.
    This path keeps answers closer to the original source text.
    """

    fitz = _load_fitz_module()
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

    weird_chars = re.findall(r"[Ðð×ØÆ¤™¢£¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ƒ‚„…†‡ˆ‰‹›œžŸ]|[\u00C0-\u024F]", text)
    replacement_like = text.count("�")
    indic_chars = len(re.findall(r"[\u0A80-\u0AFF\u0900-\u097F]", text))
    total_chars = max(len(text), 1)
    weird_ratio = (len(weird_chars) + replacement_like) / total_chars
    indic_ratio = indic_chars / total_chars

    # Many Latin-only glyph artifacts from bad font mapping push this ratio high.
    return weird_ratio > 0.03 or (indic_ratio < 0.01 and weird_ratio > 0.01)


def is_preferred_script_text(text: str) -> bool:
    has_gujarati = bool(re.search(r"[\u0A80-\u0AFF]", text))
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    return (has_gujarati or has_devanagari) and not looks_garbled(text)


def filter_clean_script_lines(lines: Iterable[str]) -> list[str]:
    return [line for line in lines if is_preferred_script_text(line)]


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


def detect_transcript_file(data_dir: Path, pdf_file: Path | None = None) -> Path | None:
    supported = ["*.txt", "*.md"]
    candidates: list[Path] = []
    for pattern in supported:
        candidates.extend(sorted(data_dir.glob(pattern)))

    if not candidates:
        return None

    if pdf_file is not None:
        pdf_stem = pdf_file.stem.lower()
        for candidate in candidates:
            if pdf_stem in candidate.stem.lower() or candidate.stem.lower() in pdf_stem:
                return candidate

    for candidate in candidates:
        if "transcript" in candidate.stem.lower() or "clean" in candidate.stem.lower():
            return candidate

    return candidates[0]


def load_transcript_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def answer_query(markdown_text: str, query: str) -> dict[str, list[str] | str]:
    lines = line_stream(markdown_text)
    number = extract_requested_number(query)
    chapter_number = extract_requested_chapter_number(query)

    if is_chapter_query(query):
        chapter_blocks = filter_clean_script_lines(find_exact_chapter_block(lines, chapter_number))
        if chapter_blocks:
            return {
                "mode": "exact-chapter-block",
                "responses": chapter_blocks,
            }

        return {
            "mode": "chapter-not-found",
            "responses": [
                "માફ કરશો, આ પ્રશ્ન માટે માગેલો અધ્યાય ફાઇલમાં મળ્યો નથી.",
                "જો તમે સાચો ટ્રાન્સક્રિપ્ટ ઉમેરશો તો હું અધ્યાય પ્રમાણે જવાબ આપી શકીશ.",
            ],
        }

    if contains_shlok_term(query.lower()):
        shlok_blocks = filter_clean_script_lines(find_exact_shlok_blocks(lines, number))
        if shlok_blocks:
            return {
                "mode": "exact-shlok-block",
                "responses": shlok_blocks,
            }

        shlok_lines = filter_clean_script_lines(find_exact_shlok_lines(lines, number))
        if shlok_lines:
            return {
                "mode": "exact-shlok-line",
                "responses": shlok_lines,
            }

    snippets = filter_clean_script_lines(retrieve_verbatim_snippets(markdown_text, query, top_k=3))
    if snippets:
        return {
            "mode": "semantic-verbatim-snippet",
            "responses": snippets,
        }

    keyword_hits = filter_clean_script_lines(keyword_line_search(markdown_text, query, top_k=5))
    if keyword_hits:
        return {
            "mode": "keyword-line-match",
            "responses": keyword_hits,
        }

    return {
        "mode": "not-found-or-low-quality",
        "responses": [
            "માફ કરશો, આ PDF માંથી હાલમાં સાફ ગુજરાતી/દેવનાગરી ટેક્સ્ટ મળ્યો નથી.",
            "કૃપા કરીને વધુ સારી સ્કેન ફાઇલ આપો અથવા OCR backend easyocr/paddleocr થી ફરી ચલાવો.",
        ],
    }


def print_answer(answer: dict[str, list[str] | str]) -> None:
    llm_answer = answer.get("llm_answer")
    if isinstance(llm_answer, str) and llm_answer.strip():
        print("LLM Answer:")
        print("----------------------------------------")
        print(llm_answer.strip())
        print()

    print(f"Mode: {answer['mode']}")
    print("Output from file:")
    for item in answer["responses"]:
        print("-" * 40)
        print(item)


def generate_gemini_answer(
    query: str,
    evidence_lines: list[str],
    api_key: str,
    model_name: str,
) -> str:
    genai = _load_gemini_module()
    if genai is None:
        raise RuntimeError(
            "Gemini dependency missing. Install with: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    evidence = "\n\n".join(evidence_lines)

    prompt = (
        "તમે બહુ કાળજીપૂર્વક જવાબ આપતા પુસ્તક સહાયક છો.\n"
        "ફક્ત આપેલ Evidence પરથી જ જવાબ આપો.\n"
        "જો જવાબ ન મળે તો સ્પષ્ટ કહો: 'આ માહિતી આપવામાં આવેલી ફાઇલમાં ઉપલબ્ધ નથી.'\n"
        "જવાબ ગુજરાતી માં આપો.\n\n"
        f"પ્રશ્ન: {query}\n\n"
        f"Evidence:\n{evidence}\n"
    )

    response = model.generate_content(prompt)
    text = getattr(response, "text", "") or ""
    return text.strip()


def maybe_apply_llm(
    answer: dict[str, list[str] | str],
    query: str,
    use_llm: bool,
    api_key: str,
    model_name: str,
) -> dict[str, list[str] | str]:
    if not use_llm:
        return answer

    evidence = answer.get("responses", [])
    if not isinstance(evidence, list) or not evidence:
        return answer

    if not api_key:
        answer["llm_error"] = "GEMINI_API_KEY missing. Showing file-grounded output only."
        return answer

    try:
        llm_answer = generate_gemini_answer(query, [str(item) for item in evidence], api_key, model_name)
        if llm_answer:
            answer["llm_answer"] = llm_answer
            answer["mode"] = f"{answer['mode']}+gemini"
    except Exception as exc:
        answer["llm_error"] = f"Gemini failed: {exc}"

    return answer


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Query the textbook in Gujlish and return file-grounded text.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Directory that contains the PDF file")
    parser.add_argument("--query", type=str, default="", help="Single query to run")
    parser.add_argument("--ocr-backend", type=str, default="tesseract", choices=["tesseract", "easyocr", "paddleocr"], help="OCR engine")
    parser.add_argument("--use-llm", action="store_true", help="Generate final Gujarati answer with Gemini using retrieved evidence")
    parser.add_argument("--llm-model", type=str, default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--gemini-api-key", type=str, default="", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--interactive", action="store_true", help="Start interactive query mode")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_file = detect_data_file(args.data_dir)
    transcript_file = detect_transcript_file(args.data_dir, pdf_file=data_file)

    if transcript_file is not None:
        print(f"Using transcript: {transcript_file.name}")
        markdown_text = load_transcript_text(transcript_file)
    else:
        print(f"Using file: {data_file.name}")
        markdown_text = load_pdf_markdown(data_file, ocr_backend=args.ocr_backend)

    if not markdown_text.strip():
        print("Mode: extraction-failed")
        print("Output from file:")
        print("----------------------------------------")
        print("માફ કરશો, આ ફાઇલમાંથી વાંચી શકાય એવો સાફ લખાણ મળ્યો નથી.")
        print("કૃપા કરીને /data માં UTF-8 transcript.txt અથવા .md ઉમેરો, અથવા વધુ સ્પષ્ટ PDF/ઈમેજ આપો.")
        return

    gemini_api_key = (args.gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()

    if args.query:
        answer = answer_query(markdown_text, args.query)
        answer = maybe_apply_llm(
            answer=answer,
            query=args.query,
            use_llm=args.use_llm,
            api_key=gemini_api_key,
            model_name=args.llm_model,
        )
        if "llm_error" in answer:
            print(f"Note: {answer['llm_error']}")
            print()
        print_answer(answer)
        return

    if args.interactive:
        print("Interactive mode started. Type 'exit' to quit.")
        while True:
            user_query = input("\nAsk in Gujlish> ").strip()
            if not user_query:
                continue
            if user_query.lower() in {"exit", "quit"}:
                break
            answer = answer_query(markdown_text, user_query)
            answer = maybe_apply_llm(
                answer=answer,
                query=user_query,
                use_llm=args.use_llm,
                api_key=gemini_api_key,
                model_name=args.llm_model,
            )
            if "llm_error" in answer:
                print(f"Note: {answer['llm_error']}")
                print()
            print_answer(answer)
        return

    parser.error("Provide --query or use --interactive")


if __name__ == "__main__":
    main()
