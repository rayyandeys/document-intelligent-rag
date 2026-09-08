from pathlib import Path
from typing import List, Dict

import pymupdf  # PyMuPDF


def load_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text page-by-page from a PDF.

    Returns:
        [
            {
                "page": 1,
                "text": "...",
                "source": "filename.pdf"
            }
        ]
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    document = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                    "source": path.name,
                }
            )

    document.close()

    if not pages:
        raise ValueError(
            "No extractable text was found in this PDF."
        )

    return pages