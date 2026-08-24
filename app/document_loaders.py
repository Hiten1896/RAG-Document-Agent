import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from typing import List, Dict, Any

# Cross-platform Tesseract executable configuration
# Uses Windows path if running locally on Windows, defaults to system path on Linux/Docker
if os.name == 'nt':
    # Default Windows installation path (adjust if your local install path differs)
    default_win_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(default_win_path):
        pytesseract.pytesseract.tesseract_cmd = default_win_path

SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.txt', '.md'}


def load_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF.

    If text content is empty or sparse, fall back to OCR on rendered pages.
    """
    text = ""
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()

        # Simple OCR fallback if page has minimal native text (e.g. scanned PDF)
        if not page_text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)

        text += page_text + "\n"

    doc.close()
    return text


def load_image(file_path: str) -> str:
    """Extract text from images using PyTesseract."""
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)


def load_text(file_path: str) -> str:
    """Read plain text and markdown files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(file_path: str) -> Dict[str, Any]:
    """Main document loader dispatcher based on file extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        content = load_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        content = load_image(file_path)
    elif ext in {".txt", ".md"}:
        content = load_text(file_path)
    else:
        raise ValueError(f"No parser available for extension: {ext}")

    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "extension": ext,
        "content": content,
    }