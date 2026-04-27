# Multilingual Knowledge Extraction & Exploration Assistant

An AI-only pipeline prototype for querying a scanned, multilingual textbook (Bhagavad Gita) using Gujlish (transliterated Gujarati) questions. The system returns verbatim, file-grounded answers in Gujarati or Devanagari script — no frontend, no backend service.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Modules](#modules)
- [Key Design Decisions](#key-design-decisions)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Example Output](#example-output)
- [Limitations](#limitations)

---

## Overview

The source PDF is a scanned Gujarati/Sanskrit textbook with an unreliable embedded text layer. This project solves the problem with a layered, explainable pipeline:

1. Prefers a clean UTF-8 transcript when available.
2. Falls back to direct PDF text extraction (PyMuPDF).
3. Uses OCR (Tesseract / EasyOCR / PaddleOCR) as a last resort.
4. Cleans and structures text into a Markdown-like hierarchy.
5. Chunks content for semantic retrieval using multilingual embeddings.
6. Routes Gujlish queries (e.g. `aama 1 shlok kayo che ?`) to the correct block.
7. Returns verbatim Gujarati/Devanagari text from the source file.
8. Optionally synthesizes a clean final answer using Gemini.

---

## Project Structure

```
multilingual_knowledge_assistant/
├── data/
│   ├── GEN AI TASK REF FILE Geeta-demo-1-10.pdf   # Source textbook
│   └── transcript_template_gu.txt                  # Clean UTF-8 transcript (preferred input)
├── notebooks/
│   └── Multilingual_Knowledge_Assistant.ipynb      # End-to-end walkthrough
├── scripts/
│   ├── demo_pipeline.py                            # Minimal pipeline demo (no PDF needed)
│   └── query_data_gujlish.py                       # Main query script for the interview demo
├── src/
│   ├── __init__.py
│   ├── document_understanding.py                   # Module 1: OCR, cleanup, hierarchy
│   ├── chunking.py                                 # Module 2: Structure-aware chunking + embeddings
│   └── retrieval.py                                # Module 3: Semantic/keyword search + LLM synthesis
├── .tessdata/
│   ├── eng.traineddata
│   ├── guj.traineddata
│   ├── osd.traineddata
│   └── san.traineddata
├── .env                                            # GEMINI_API_KEY (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline Architecture

```
Input: PDF / Transcript
         ↓
Text Extraction (PyMuPDF direct → OCR fallback)
         ↓
Text Cleaning & Unicode Normalization
         ↓
Markdown-like Hierarchy Reconstruction
         ↓
Structure-Aware Chunking (RecursiveCharacterTextSplitter style)
         ↓
Multilingual Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
         ↓
Vector Index (FAISS / in-memory cosine fallback)
         ↓
User Query (Gujlish / Gujarati / English)
         ↓
Query Routing → Shlok / Chapter / Semantic / Keyword
         ↓
Verbatim Evidence Retrieval (Gujarati / Devanagari lines)
         ↓
Optional: Gemini LLM Final Answer Synthesis
         ↓
Output: Grounded Answer
```

---

## Modules

### `src/document_understanding.py`

Handles the full document ingestion pipeline:

- **PDF loading** via PyMuPDF (`fitz`), rendering pages at configurable DPI.
- **Image preprocessing**: EXIF correction, grayscale, median denoising, autocontrast, binarization.
- **OCR backends**: Tesseract (default), EasyOCR, PaddleOCR — all optional, with graceful fallbacks.
- **Tesseract auto-configuration**: detects the executable on Windows without requiring PATH setup.
- **Local `.tessdata/` support**: uses project-local language packs (Gujarati, Sanskrit, English, OSD).
- **Text cleanup**: Unicode NFKC normalization, hyphenation repair, whitespace normalization, page-number stripping.
- **Hierarchy reconstruction**: classifies lines as headings, bullets, paragraphs, or table rows using font-height heuristics and text patterns.
- **Output**: a `DocumentResult` with structured `DocumentBlock` list and a Markdown string.

Key classes: `DocumentProcessor`, `OCRConfig`, `DocumentResult`, `DocumentBlock`, `OCRLine`

---

### `src/chunking.py`

Converts structured document blocks into retrieval-ready chunks:

- **Structure-aware chunking**: preserves heading context as `section_path`, respects token budgets (default 350 tokens), applies overlap (default 50 tokens).
- **Semantic chunking**: optional sentence-window mode with configurable overlap.
- **Markdown convenience wrapper**: `chunk_markdown()` parses raw Markdown strings directly.
- **Multilingual embedding model**: thin wrapper around `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with a hash-based fallback when the library is unavailable.

Key classes: `StructureAwareChunker`, `ChunkConfig`, `Chunk`, `MultilingualEmbeddingModel`

---

### `src/retrieval.py`

Provides search and answer generation over the chunk index:

- **Semantic search**: L2-normalized FAISS inner-product search; falls back to in-memory cosine similarity when FAISS is unavailable.
- **Keyword search**: term-overlap scoring, language-agnostic, works without embeddings.
- **Extractive answer**: scores sentences by query-term overlap and returns the top 3.
- **LLM synthesis**: accepts any callable `llm_client(prompt) -> str`; formats a grounded prompt that instructs the model to answer only from retrieved context.

Key classes: `ChunkRetriever`, `RetrievalResult`

---

### `scripts/query_data_gujlish.py`

The main interview demo script. Full feature set:

- Auto-detects the PDF and transcript in `data/`.
- Prefers the clean transcript over OCR output.
- Detects garbled PDF text layers using a heuristic ratio check (`looks_garbled`).
- Parses Gujlish queries: extracts shlok numbers, chapter numbers, ordinal words (`pehlo`, `bijo`, `trijo`…), and Gujarati/Devanagari digits.
- Routes queries to four modes: `exact-shlok-block`, `exact-chapter-block`, `semantic-verbatim-snippet`, `keyword-line-match`.
- Filters output to only return lines containing valid Gujarati or Devanagari script.
- Optionally calls Gemini to produce a polished Gujarati final answer grounded in the retrieved evidence.
- Supports `--query`, `--interactive`, `--use-llm`, `--llm-model`, `--gemini-api-key`, `--ocr-backend` flags.

---

### `scripts/demo_pipeline.py`

A self-contained end-to-end demo using hardcoded sample text (no PDF or OCR required). Demonstrates all three modules in sequence: document understanding → chunking → keyword search → semantic answer.

---

## Key Design Decisions

**1. Transcript-first input**
The provided PDF produces mojibake when its embedded text layer is read directly. The pipeline prefers a clean UTF-8 transcript from `data/` when one is present, making the demo reliable without depending on OCR quality.

**2. Garbled text detection**
A heuristic ratio check (`looks_garbled`) measures the proportion of Latin-range glyph artifacts and replacement characters. If the ratio exceeds 3%, the embedded text layer is discarded and OCR is attempted instead.

**3. Gujlish query routing**
Common transliterated Gujarati patterns are mapped to the correct retrieval mode:
- `aama 1 shlok kayo che ?` → `exact-shlok-block`
- `pehlo adyay su kehva mange che ?` → `exact-chapter-block`
- Free-form questions → `semantic-verbatim-snippet` → `keyword-line-match`

**4. Script-quality filtering**
All retrieved lines are filtered through `is_preferred_script_text`, which checks for valid Gujarati (`\u0A80–\u0AFF`) or Devanagari (`\u0900–\u097F`) Unicode ranges. Broken OCR output is never returned as if it were valid text.

**5. Verbatim grounding**
The default output returns exact lines or blocks from the source file. Gemini synthesis is an optional final layer that stays grounded in the retrieved evidence via a strict prompt.

**6. Optional dependencies**
Every heavy dependency (PyMuPDF, Tesseract, EasyOCR, PaddleOCR, FAISS, sentence-transformers, Gemini) is wrapped in a `try/except` import. The pipeline degrades gracefully at each layer rather than crashing.

---

## Tech Stack

| Component | Library |
|---|---|
| PDF text extraction | `pymupdf` (fitz) |
| OCR (primary) | `pytesseract` + Tesseract OCR |
| OCR (optional) | `easyocr`, `paddleocr` |
| Image preprocessing | `Pillow`, `numpy` |
| Multilingual embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector search | `faiss-cpu` (in-memory cosine fallback) |
| LLM answer synthesis | `google-generativeai` (Gemini `gemini-1.5-flash`) |
| Environment config | `python-dotenv` |
| Language packs | Tesseract `.tessdata` for Gujarati (`guj`), Sanskrit (`san`), English (`eng`) |

---

## Setup & Installation

### 1. Clone and install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (Windows)

Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

The script auto-detects the executable at common Windows paths. No PATH configuration needed.

### 3. Set your Gemini API key (optional — only needed for `--use-llm`)

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Or set it as an environment variable:

```powershell
# PowerShell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

---

## Usage

### Run the pipeline demo (no PDF required)

```bash
python scripts/demo_pipeline.py
```

### Ask a single Gujlish question

```bash
python scripts/query_data_gujlish.py --query "aama 1 shlok kayo che ?"
```

### Ask a chapter meaning question

```bash
python scripts/query_data_gujlish.py --query "pehlo adyay su kehva mange che ?"
```

### Enable Gemini LLM synthesis

```bash
python scripts/query_data_gujlish.py --query "aama 1 shlok kayo che ?" --use-llm
```

### Specify a Gemini model

```bash
python scripts/query_data_gujlish.py --query "bijo adyay su kehva mange che ?" --use-llm --llm-model "gemini-1.5-flash"
```

### Pass the API key directly (without .env)

```bash
python scripts/query_data_gujlish.py --query "aama 1 shlok kayo che ?" --use-llm --gemini-api-key "your_key"
```

### Use a different OCR backend

```bash
python scripts/query_data_gujlish.py --query "aama 1 shlok kayo che ?" --ocr-backend easyocr
```

### Interactive mode

```bash
python scripts/query_data_gujlish.py --interactive
```

---

## Example Output

```
Using transcript: transcript_template_gu.txt

Mode: exact-shlok-block
Output from file:
----------------------------------------
શ્લોક ૧
ધૃતરાષ્ટ્ર ઉવાચ: ધર્મક્ષેત્રે કુરુક્ષેત્રે સમવેતા યુયુત્સવઃ ।
મામકાઃ પાંડવાશ્ચૈવ કિમકુર્વત સંજય ॥૧॥
```

With `--use-llm`:

```
LLM Answer:
----------------------------------------
ધૃતરાષ્ટ્ર સંજયને પૂછે છે: ધર્મક્ષેત્ર કુરુક્ષેત્રમાં એકઠા થયેલા મારા અને પાંડુના પુત્રોએ શું કર્યું?

Mode: exact-shlok-block+gemini
Output from file:
----------------------------------------
શ્લોક ૧
...
```

---

## Limitations

- The provided PDF text layer is font-encoded and produces mojibake; OCR is required for reliable extraction from the raw PDF.
- EasyOCR and Tesseract both struggle on this particular scan; the transcript-first path is the most reliable for the demo.
- EasyOCR does not support Gujarati natively — it falls back to Hindi as a practical approximation.
- PaddleOCR support for Gujarati/Sanskrit is limited; it defaults to English mode for these languages.
- Shlok and chapter routing depends on the source file containing recognizable Unicode Gujarati/Devanagari markers. Heavily garbled scans will fall through to the `not-found-or-low-quality` fallback.
