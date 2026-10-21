# PSAI

PSAI is an AI-assisted project management application. The preprocessing
package turns manually approved project PDFs into traceable text records and
document-level chunks for later annotation and retrieval workflows.

## Preprocessing overview

The default pipeline:

1. Discovers PDF files in an input directory (or accepts one PDF).
2. Extracts native text page by page and marks sparse-text pages as potential
   OCR candidates. It does not apply a mandatory automated PDF validation gate.
3. Optionally runs Tesseract OCR on candidate pages.
4. Conservatively cleans text and chunks each page independently.
5. Writes document records, chunks, processing errors, and a quality report.

The implementation uses the repository's `preprocessing` package directly;
run it from the repository root as `python -m preprocessing.pipeline`.

## Directory structure

```text
psai/
├── data/
│   ├── raw/                 # Manually approved input PDFs
│   └── processed/           # Generated JSONL and JSON reports
├── preprocessing/
│   ├── annotation.py
│   ├── chunking.py
│   ├── cleaning.py
│   ├── config.py
│   ├── data_quality.py
│   ├── deduplication.py
│   ├── extract_text.py
│   ├── ocr.py
│   ├── pipeline.py
│   └── split_dataset.py
├── tests/
└── requirements.txt
```

## Installation

Python 3.10 or later is required. From PowerShell at the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, allow activation for the current
PowerShell process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## OCR setup

OCR is optional and can be disabled. Installing the Python package
`pytesseract` **does not install the Tesseract OCR engine**. Install the
Tesseract executable separately, make it available on `PATH` (or configure
`pytesseract.pytesseract.tesseract_cmd` in the calling environment), and install
the language data used by the command. For example, `eng` requires English
trained data, while `vie+eng` requires both Vietnamese and English data.

If the executable or requested language data is missing, the pipeline records
an actionable OCR error and continues processing other PDFs with available
native text.

## Run the pipeline

Process PDFs recursively from the default raw directory:

```powershell
python -m preprocessing.pipeline --input-dir data/raw --output-dir data/processed
```

Process one PDF:

```powershell
python -m preprocessing.pipeline --input-dir data/raw\project-plan.pdf
```

Disable OCR, or configure Vietnamese and English OCR with a chosen resolution:

```powershell
python -m preprocessing.pipeline --input-dir data/raw --output-dir data/processed --no-ocr
python -m preprocessing.pipeline --input-dir data/raw --ocr-language vie+eng --ocr-dpi 300
```

Add `--non-recursive` to search only the specified input directory. Pages with
fewer than 30 non-whitespace native characters are OCR candidates by default;
change this with `--min-native-text-chars`.

## Configuration

The `PreprocessingConfig` dataclass centralizes paths, OCR language and
resolution, native-text threshold, chunk size/overlap, tokenizer selection,
split ratios, grouping key, and output names. For example:

```python
from pathlib import Path

from preprocessing.config import PreprocessingConfig

config = PreprocessingConfig(
    raw_pdf_dir=Path("data/raw"),
    processed_dir=Path("data/processed"),
    ocr_language="vie+eng",
    ocr_dpi=300,
    max_chunk_size=500,
    chunk_overlap=50,
    min_chunk_size=20,
    tokenizer="regex",
    recursive=True,
)
```

The default `regex` tokenizer is dependency-free and provides an approximate
token count. A `tiktoken` encoding name may also be selected; when `tiktoken`
or the requested encoding is unavailable, chunking falls back to the regex
tokenizer and records the tokenizer actually used in its statistics. Install
`tiktoken` separately if that tokenizer is needed.

## Outputs and traceability

The CLI writes UTF-8 files to the output directory:

- `documents.jsonl`: one JSON object per PDF, including its SHA-256
  `document_id`, filename, source path, page count, PDF metadata, and a `pages`
  array. Each page retains `native_text`; when OCR runs, `ocr_text`,
  `ocr_attempted`, `ocr_quality`, and `ocr_output_selected` are also recorded.
  `text` is the selected source text, and `cleaned_text` contains its cleaned
  version.
- `chunks.jsonl`: one object per chunk with deterministic `chunk_id`,
  `document_id`, `file_name`, `source_path`, one-based `page_number`, text,
  token count, tokenizer, extraction method, OCR selection status, and optional
  section heading. Pages are chunked independently so the page attribution is
  exact.
- `errors.jsonl`: per-file extraction errors and per-document OCR errors.
- `quality_report.json`: non-destructive document and chunk quality findings.

Encrypted, malformed, or unreadable PDFs are represented in the document/error
outputs; they are not silently dropped. Source PDFs are never modified. OCR
text is not selected over non-empty native text automatically, because OCR
length alone is not a reliable accuracy measure.

To manually inspect OCR, compare each page's `native_text`, `ocr_text`,
`ocr_quality`, and `text` fields in `documents.jsonl`. OCR confidence is an
indicator, not a guarantee of correctness.

## Generate local embeddings

Install the dependencies (the first `sentence-transformers` run downloads the
selected model if it is not already cached):

```powershell
python -m pip install -r requirements.txt
python -m preprocessing.pipeline --input-dir data/raw --output-dir data/processed --no-ocr
python -m embedding.pipeline --input data/processed/chunks.jsonl --output-dir data/processed/embeddings
```

The default `sentence-transformers/all-MiniLM-L6-v2` is a compact local model
for English text. It does not provide reliable Vietnamese or general
multilingual support. For Vietnamese and multilingual text, select
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; model quality
can vary by language and domain. Model weights are downloaded from the model
hub on first use unless a local model directory is supplied. No API key is
required. CPU is the default, so GPU availability is not assumed.

Example options:

```powershell
python -m embedding.pipeline --input data/processed/chunks.jsonl --output-dir data/processed/embeddings --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --batch-size 16 --device cpu
```

The input is one chunk JSON object per line; current preprocessing records
provide `chunk_id`, `document_id`, `file_name`, `source_path`, `page_number`,
`text`, `extraction_method`, `ocr_used`, `section_heading`, `token_count`, and
`tokenizer`. The embedding pipeline preserves each input field in the matching
metadata row, including `text`, and adds `embedding_index`, `embedding_model`,
`embedding_dimension`, and a text SHA-256 digest. Invalid JSON or records
without string text are reported as failures; empty text is reported as
skipped. The manifest includes counts and per-record failure/skip reasons.

Output files under the selected directory:

- `embeddings.npy`: float32 matrix, shape `(processed_count, dimension)`.
- `embedding_metadata.jsonl`: one metadata record per vector row, in exactly
  the same order.
- `embedding_manifest.json`: model, device, dimension, normalization,
  input/output counts, input checksum, output names, and parse/inference
  issues. The manifest is published last and acts as the completion marker;
  if a run is interrupted during replacement, no current manifest is left
  suggesting the new outputs are a complete set.

Vectors are L2-normalized by default so cosine similarity can be computed via
dot product. Use `--no-normalize` to retain raw model vectors. The model is
inference-only; this module does not train or fine-tune it, and does not
implement a vector database or retrieval.

## Annotations, duplicates, quality, and dataset splits

`load_annotations` accepts JSON arrays/objects and JSONL records. Required
fields and expected field types are optional arguments; unknown fields are
preserved and malformed records are returned with line numbers.
`find_file_duplicates` groups exact file matches by SHA-256.
`find_text_duplicates` identifies normalized exact-text groups and can return
optional `SequenceMatcher` similarity candidates for manual review. No source
or record is deleted automatically.

`split_records` assigns whole groups (by `document_id` by default) to train,
validation, and test sets using a fixed seed. Split before fitting any
data-dependent transformation or evaluating a model. If the records are
already chunked, group by `document_id` or a suitable project key so related
chunks cannot cross subsets. The splitter reports record and group counts;
small datasets may not match the requested ratios exactly.

## Testing

Run the test suite from the repository root:

```powershell
python -m unittest discover -s tests
```

The tests generate small PDF fixtures locally. End-to-end OCR quality depends
on an external Tesseract executable and installed language data; OCR logic is
also unit-tested with a stubbed recognizer so those tests do not require the
external engine.

Embedding unit tests use a small mock model and do not download pretrained
weights.

## Limitations and troubleshooting

- PDF text extraction depends on how the source PDF encodes text. Scanned
  pages need OCR, and complex layouts, tables, diagrams, handwriting, or
  unusual fonts may be extracted incorrectly.
- OCR can misread project IDs, task dependencies, dates, durations, and table
  values. Review these fields before downstream use.
- Cleaning preserves paragraph boundaries and project details and only joins
  a conservative lowercase hyphenated line-break pattern. It does not repair
  arbitrary layout or table structure.
- The default regex tokenizer is approximate; configure and install a
  `tiktoken` encoding for more model-specific token counts.
- If OCR reports a missing executable, verify `tesseract --version` in the
  same terminal used to run Python. If it reports missing language data,
  install that language's trained data and check Tesseract's tessdata path.
- If activation fails because script execution is restricted, use the
  process-scoped PowerShell command shown in Installation; it does not change
  the machine-wide execution policy.
