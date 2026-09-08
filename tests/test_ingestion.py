from src.ingestion.pdf_loader import load_pdf


pages = load_pdf("data/raw/sample.pdf")

print(f"Pages extracted: {len(pages)}")

for page in pages[:3]:
    print("\n" + "=" * 60)
    print(f"Source: {page['source']}")
    print(f"Page: {page['page']}")
    print("=" * 60)
    print(page["text"][:1000])