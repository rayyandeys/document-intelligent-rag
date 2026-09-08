from typing import List, Dict


def smart_chunk_pages(
    pages: List[Dict],
    chunk_size: int = 800,
    overlap_sentences: int = 1
) -> List[Dict]:

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]

        sentences = [
            sentence.strip()
            for sentence in text.replace("\n", " ").split(". ")
            if sentence.strip()
        ]

        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence.endswith("."):
                sentence += "."

            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": page["page"],
                    "source": page["source"]
                })

                chunk_id += 1

                if overlap_sentences > 0:
                    current_sentences = current_sentences[-overlap_sentences:]
                    current_length = sum(
                        len(s) for s in current_sentences
                    )
                else:
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_length

        if current_sentences:
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(current_sentences),
                "page": page["page"],
                "source": page["source"]
            })

            chunk_id += 1

    return chunks