from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_pages


pages = load_pdf("data/raw/sample.pdf")

chunks = chunk_pages(
    pages,
    chunk_size=800,
    overlap=120
)

print(f"Pages extracted: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

for chunk in chunks[:5]:
    print("\n" + "=" * 60)
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page']}")
    print(f"Source: {chunk['source']}")
    print("=" * 60)
    print(chunk["text"][:500])