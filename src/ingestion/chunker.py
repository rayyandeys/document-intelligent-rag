from typing import List, Dict


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 800,
    overlap: int = 120
) -> List[Dict]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": page["page"],
                    "source": page["source"]
                })

                chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks