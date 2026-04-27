"""Module 1: OCR, cleanup, and hierarchy reconstruction.

The implementation is intentionally pragmatic:
- Use OCR when the real libraries are installed.
- Fall back to text-only parsing for notebook demos and lightweight testing.
- Reconstruct a Markdown-like hierarchy with heuristics that are easy to explain in an interview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
import re
import shutil
import os
import unicodedata
from typing import Any

from PIL import Image, ImageOps, ImageFilter
import numpy as np

try:  # Optional dependency.
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:  # Optional dependency.
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None
    Output = None

try:  # Optional dependency.
    import easyocr
except Exception:  # pragma: no cover - optional dependency
    easyocr = None

try:  # Optional dependency.
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - optional dependency
    PaddleOCR = None


LANGUAGE_ALIASES = {
    "eng": "eng",
    "en": "eng",
    "guj": "guj",
    "gu": "guj",
    "san": "san",
    "sa": "san",
}


@dataclass
class OCRLine:
    text: str
    bbox: list[int] = field(default_factory=list)
    confidence: float | None = None
    page_number: int = 1


@dataclass
class DocumentBlock:
    block_type: str
    text: str
    level: int | None = None
    page_number: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentResult:
    source_name: str
    page_count: int
    markdown: str
    blocks: list[DocumentBlock]
    ocr_backend: str
    notes: list[str] = field(default_factory=list)


@dataclass
class OCRConfig:
    backend: str = "tesseract"
    language_hints: list[str] = field(default_factory=lambda: ["eng", "guj", "san"])
    render_dpi: int = 220


class DocumentProcessor:
    """End-to-end processor that turns scanned pages into structured Markdown."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()

    def process_file(self, file_path: str | Path) -> DocumentResult:
        path = Path(file_path)
        return self.process_bytes(path.read_bytes(), source_name=path.name)

    def process_bytes(self, file_bytes: bytes, source_name: str = "uploaded_document") -> DocumentResult:
        if source_name.lower().endswith(".pdf"):
            pages = self._extract_pdf_pages(file_bytes)
        else:
            pages = [Image.open(BytesIO(file_bytes)).convert("RGB")]

        all_blocks: list[DocumentBlock] = []
        notes: list[str] = []

        for page_number, image in enumerate(pages, start=1):
            lines = self._ocr_image(image, page_number)
            cleaned_lines = self._clean_lines(lines)
            page_blocks = self._structure_lines(cleaned_lines)
            all_blocks.extend(page_blocks)
            if not lines:
                notes.append(f"Page {page_number}: no OCR lines found; check scan quality or OCR backend availability.")

        markdown = self._blocks_to_markdown(all_blocks)
        return DocumentResult(
            source_name=source_name,
            page_count=len(pages),
            markdown=markdown,
            blocks=all_blocks,
            ocr_backend=self.config.backend,
            notes=notes,
        )

    def process_text(self, text: str, source_name: str = "sample_text") -> DocumentResult:
        """Text-only path used in the notebook demo and for smoke testing."""

        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines = [OCRLine(text=line, page_number=1) for line in raw_lines]
        cleaned_lines = self._clean_lines(lines)
        blocks = self._structure_lines(cleaned_lines)
        markdown = self._blocks_to_markdown(blocks)
        return DocumentResult(
            source_name=source_name,
            page_count=1,
            markdown=markdown,
            blocks=blocks,
            ocr_backend="text-fallback",
            notes=["Used text-only fallback. Swap in OCR backends for real scans."],
        )

    def _ocr_image(self, image: Image.Image, page_number: int) -> list[OCRLine]:
        processed = self._preprocess_image(image)
        backend = self.config.backend.lower().strip()

        if backend == "easyocr":
            return self._ocr_with_easyocr(processed, page_number)
        if backend == "paddleocr":
            return self._ocr_with_paddleocr(processed, page_number)
        return self._ocr_with_tesseract(processed, page_number)

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        gray = ImageOps.grayscale(image)
        denoised = gray.filter(ImageFilter.MedianFilter(size=3))
        enhanced = ImageOps.autocontrast(denoised)
        return enhanced.point(lambda px: 255 if px > 170 else 0)

    def _extract_pdf_pages(self, pdf_bytes: bytes) -> list[Image.Image]:
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for PDF input. Install pymupdf.")

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[Image.Image] = []
        zoom = self.config.render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB"))
        document.close()
        return pages

    def _ocr_with_tesseract(self, image: Image.Image, page_number: int) -> list[OCRLine]:
        if pytesseract is None or Output is None:
            return []

        self._configure_tesseract_executable()
        tessdata_dir = self._resolve_tessdata_dir()

        requested_languages = self._normalize_languages(self.config.language_hints)
        available_languages = self._available_tesseract_languages()
        selected_languages = [lang for lang in requested_languages if lang in available_languages]
        if not selected_languages:
            selected_languages = ["eng"] if "eng" in available_languages else requested_languages[:1]

        lang = "+".join(selected_languages) or "eng"
        try:
            ocr_config = "--oem 3 --psm 6"
            if tessdata_dir is not None:
                ocr_config += f' --tessdata-dir "{tessdata_dir}"'

            data: dict[str, list[Any]] = pytesseract.image_to_data(
                image,
                lang=lang,
                config=ocr_config,
                output_type=Output.DICT,
            )
        except Exception:
            return []

        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        total_items = len(data.get("text", []))
        for index in range(total_items):
            text = str(data["text"][index]).strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except Exception:
                confidence = -1.0
            if confidence < 0:
                continue

            key = (
                int(data.get("block_num", [0])[index]),
                int(data.get("par_num", [0])[index]),
                int(data.get("line_num", [0])[index]),
            )
            grouped.setdefault(key, []).append(
                {
                    "text": text,
                    "left": int(data.get("left", [0])[index]),
                    "top": int(data.get("top", [0])[index]),
                    "width": int(data.get("width", [0])[index]),
                    "height": int(data.get("height", [0])[index]),
                    "confidence": confidence,
                }
            )

        lines: list[OCRLine] = []
        for key in sorted(grouped):
            words = sorted(grouped[key], key=lambda item: item["left"])
            text = " ".join(word["text"] for word in words).strip()
            if not text:
                continue
            left = min(word["left"] for word in words)
            top = min(word["top"] for word in words)
            right = max(word["left"] + word["width"] for word in words)
            bottom = max(word["top"] + word["height"] for word in words)
            confidence = sum(word["confidence"] for word in words) / len(words)
            lines.append(OCRLine(text=text, bbox=[left, top, right, bottom], confidence=confidence, page_number=page_number))
        return lines

    def _configure_tesseract_executable(self) -> None:
        """Set pytesseract executable path when Tesseract is installed outside PATH."""

        if pytesseract is None:
            return

        if shutil.which("tesseract"):
            return

        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = candidate
                break

    def _available_tesseract_languages(self) -> set[str]:
        local_tessdata = self._resolve_tessdata_dir()
        if local_tessdata is not None:
            local_langs = {
                file.stem for file in local_tessdata.glob("*.traineddata") if file.is_file()
            }
            if local_langs:
                return local_langs

        if pytesseract is None:
            return {"eng"}
        try:
            return set(pytesseract.get_languages(config=""))
        except Exception:
            return {"eng"}

    def _resolve_tessdata_dir(self) -> Path | None:
        env_dir = os.environ.get("TESSDATA_PREFIX")
        if env_dir:
            path = Path(env_dir)
            if path.exists():
                return path

        project_local = Path(__file__).resolve().parents[1] / ".tessdata"
        if project_local.exists():
            return project_local
        return None

    def _ocr_with_easyocr(self, image: Image.Image, page_number: int) -> list[OCRLine]:
        if easyocr is None:
            return []

        reader = easyocr.Reader(self._normalize_easyocr_languages(self.config.language_hints), gpu=False)
        image_array = np.array(image)
        lines: list[OCRLine] = []
        for bbox, text, confidence in reader.readtext(image_array):
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            lines.append(
                OCRLine(
                    text=str(text).strip(),
                    bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    confidence=float(confidence),
                    page_number=page_number,
                )
            )
        return [line for line in lines if line.text]

    def _ocr_with_paddleocr(self, image: Image.Image, page_number: int) -> list[OCRLine]:
        if PaddleOCR is None:
            return []

        reader = PaddleOCR(use_angle_cls=True, lang=self._choose_paddle_language(self.config.language_hints))
        raw = reader.ocr(image, cls=True)
        lines: list[OCRLine] = []
        if not raw:
            return lines
        for page in raw:
            for box, (text, confidence) in page:
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]
                lines.append(
                    OCRLine(
                        text=str(text).strip(),
                        bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                        confidence=float(confidence),
                        page_number=page_number,
                    )
                )
        return [line for line in lines if line.text]

    def _normalize_languages(self, languages: list[str]) -> list[str]:
        normalized: list[str] = []
        for language in languages:
            alias = LANGUAGE_ALIASES.get(language.lower())
            if alias and alias not in normalized:
                normalized.append(alias)
        return normalized or ["eng"]

    def _normalize_easyocr_languages(self, languages: list[str]) -> list[str]:
        # EasyOCR does not support Gujarati directly in all builds,
        # so Gujarati/Sanskrit are mapped to Hindi as a practical fallback.
        mapping = {
            "en": "en",
            "eng": "en",
            "gu": "hi",
            "guj": "hi",
            "sa": "hi",
            "san": "hi",
            "hi": "hi",
            "hin": "hi",
        }
        normalized = [mapping.get(language.lower(), "en") for language in languages]
        return list(dict.fromkeys(normalized)) or ["en"]

    def _choose_paddle_language(self, languages: list[str]) -> str:
        # PaddleOCR language support is narrower, so we keep the fallback simple.
        normalized = {language.lower() for language in languages}
        if "guj" in normalized or "gu" in normalized:
            return "en"
        if "san" in normalized or "sa" in normalized:
            return "en"
        return "en"

    def _clean_lines(self, lines: list[OCRLine]) -> list[OCRLine]:
        cleaned: list[OCRLine] = []
        for line in lines:
            text = self._normalize_text(line.text)
            if text:
                cleaned.append(OCRLine(text=text, bbox=line.bbox, confidence=line.confidence, page_number=line.page_number))
        return cleaned

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\xa0", " ")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
        return text.strip()

    def _structure_lines(self, lines: list[OCRLine]) -> list[DocumentBlock]:
        if not lines:
            return []

        blocks: list[DocumentBlock] = []
        paragraph_buffer: list[str] = []
        bullet_buffer: list[str] = []
        current_page = lines[0].page_number
        page_heights = self._page_median_heights(lines)

        for line in sorted(lines, key=lambda item: (item.page_number, item.bbox[1] if item.bbox else 0, item.bbox[0] if item.bbox else 0)):
            if line.page_number != current_page:
                self._flush_buffers(blocks, paragraph_buffer, bullet_buffer, current_page)
                blocks.append(DocumentBlock(block_type="page_break", text=f"Page {current_page}", page_number=current_page))
                current_page = line.page_number

            line_type, level = self._classify_line(line, page_heights.get(line.page_number, 0.0))
            if line_type == "heading":
                self._flush_buffers(blocks, paragraph_buffer, bullet_buffer, current_page)
                blocks.append(DocumentBlock(block_type="heading", text=line.text, level=level, page_number=line.page_number, metadata={"bbox": line.bbox, "confidence": line.confidence}))
                continue
            if line_type == "bullet":
                self._flush_paragraph(blocks, paragraph_buffer, current_page)
                bullet_buffer.append(self._strip_bullet_prefix(line.text))
                continue
            if self._looks_like_table_row(line.text):
                self._flush_buffers(blocks, paragraph_buffer, bullet_buffer, current_page)
                blocks.append(DocumentBlock(block_type="table", text=line.text, page_number=line.page_number, metadata={"bbox": line.bbox, "confidence": line.confidence}))
                continue

            self._flush_bullet_list(blocks, bullet_buffer, current_page)
            paragraph_buffer.append(line.text)

        self._flush_buffers(blocks, paragraph_buffer, bullet_buffer, current_page)
        return blocks

    def _classify_line(self, line: OCRLine, page_median_height: float) -> tuple[str, int | None]:
        text = line.text.strip()
        if self._is_bullet(text):
            return "bullet", None
        if self._is_heading_candidate(text, line, page_median_height):
            return "heading", self._heading_level(text)
        return "paragraph", None

    def _is_bullet(self, text: str) -> bool:
        stripped = text.strip()
        return bool(re.match(r"^([-*•·–—]|\d+[.)]|[a-zA-Z][.)])\s+", stripped))

    def _strip_bullet_prefix(self, text: str) -> str:
        return re.sub(r"^([-*•·–—]|\d+[.)]|[a-zA-Z][.)])\s+", "", text.strip())

    def _is_heading_candidate(self, text: str, line: OCRLine, page_median_height: float) -> bool:
        if not text or len(text) > 140:
            return False
        has_numbering = bool(re.match(r"^(chapter\s+)?\d+(\.\d+)*[:.)-]?\s+", text.lower()))
        title_like = text == text.title() and len(text.split()) <= 8
        all_caps_short = text.upper() == text and len(text.split()) <= 10 and len(text) <= 90
        ends_with_colon = text.endswith(":")
        line_height = (line.bbox[3] - line.bbox[1]) if len(line.bbox) == 4 else 0
        taller_than_body = page_median_height > 0 and line_height >= page_median_height * 1.2
        return has_numbering or title_like or all_caps_short or ends_with_colon or taller_than_body

    def _heading_level(self, text: str) -> int:
        normalized = text.lower().strip()
        if normalized.startswith("chapter"):
            return 1
        numeric_match = re.match(r"^(\d+(?:\.\d+)*)", normalized)
        if numeric_match:
            depth = numeric_match.group(1).count(".") + 1
            return min(depth + 1, 4)
        return 2

    def _looks_like_table_row(self, text: str) -> bool:
        return "|" in text or "\t" in text or re.search(r"\s{3,}", text) is not None

    def _page_median_heights(self, lines: list[OCRLine]) -> dict[int, float]:
        page_heights: dict[int, list[float]] = {}
        for line in lines:
            if len(line.bbox) == 4:
                height = max(line.bbox[3] - line.bbox[1], 1)
                page_heights.setdefault(line.page_number, []).append(float(height))
        medians: dict[int, float] = {}
        for page_number, heights in page_heights.items():
            heights = sorted(heights)
            mid = len(heights) // 2
            medians[page_number] = heights[mid] if len(heights) % 2 else (heights[mid - 1] + heights[mid]) / 2
        return medians

    def _flush_paragraph(self, blocks: list[DocumentBlock], paragraph_buffer: list[str], page_number: int) -> None:
        if paragraph_buffer:
            blocks.append(DocumentBlock(block_type="paragraph", text=" ".join(paragraph_buffer).strip(), page_number=page_number))
            paragraph_buffer.clear()

    def _flush_bullet_list(self, blocks: list[DocumentBlock], bullet_buffer: list[str], page_number: int) -> None:
        for bullet in bullet_buffer:
            if bullet:
                blocks.append(DocumentBlock(block_type="bullet", text=bullet, page_number=page_number))
        bullet_buffer.clear()

    def _flush_buffers(self, blocks: list[DocumentBlock], paragraph_buffer: list[str], bullet_buffer: list[str], page_number: int) -> None:
        self._flush_paragraph(blocks, paragraph_buffer, page_number)
        self._flush_bullet_list(blocks, bullet_buffer, page_number)

    def _blocks_to_markdown(self, blocks: list[DocumentBlock]) -> str:
        parts: list[str] = []
        for block in blocks:
            if block.block_type == "page_break":
                parts.append(f"\n---\n\n> {block.text}")
            elif block.block_type == "heading":
                level = max(1, min(block.level or 1, 6))
                parts.append(f"{'#' * level} {block.text}")
            elif block.block_type == "bullet":
                parts.append(f"- {block.text}")
            else:
                parts.append(block.text)
        return "\n\n".join(parts).strip()
