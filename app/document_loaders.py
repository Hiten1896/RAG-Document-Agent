import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from typing import Dict, Any, Optional

# Cross-platform Tesseract executable configuration
if os.name == 'nt':
    default_win_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(default_win_path):
        pytesseract.pytesseract.tesseract_cmd = default_win_path

SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.txt', '.md'}


def load_pdf_fast(file_path: str, enable_ocr_fallback: bool = False) -> str:
    """Fast PDF extraction. Extracts digital text directly; falls back to OCR only if empty."""
    text_chunks = []
    doc = fitz.open(file_path)

    for page in doc:
        page_text = page.get_text("text")

        # Fast path: digital text exists
        if page_text and len(page_text.strip()) > 10:
            text_chunks.append(page_text)
        elif enable_ocr_fallback:
            # Slow path: scanned image page fallback (DPI reduced to 150 for speed)
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text_chunks.append(pytesseract.image_to_string(img))

    doc.close()
    return "\n\n".join(text_chunks)


def load_image(file_path: str) -> str:
    """Extract text from images using PyTesseract."""
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)


def load_text(file_path: str) -> str:
    """Read plain text and markdown files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(file_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
    """Main document loader dispatcher."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename_for_ext = original_filename if original_filename else file_path
    ext = os.path.splitext(filename_for_ext)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        content = load_pdf_fast(file_path, enable_ocr_fallback=False)
    elif ext in {".png", ".jpg", ".jpeg"}:
        content = load_image(file_path)
    elif ext in {".txt", ".md"}:
        content = load_text(file_path)
    else:
        raise ValueError(f"No parser available for extension: {ext}")

    return {
        "file_path": file_path,
        "file_name": original_filename or os.path.basename(file_path),
        "extension": ext,
        "content": content,
    }