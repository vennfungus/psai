# /psai/preprocessing/extract_text.py
"""Extract native text from PDF files for PSAI."""

import hashlib
from pathlib import Path

import fitz  # PyMuPDF
from .config import PreprocessingConfig