import os
from typing import List, Dict

from dotenv import load_dotenv
from google import genai


load_dotenv()


class Generator:
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. Check your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:

        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"[Source: {chunk['source']}, Page: {chunk['page']}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the evidence provided below.

Rules:
1. Do not use outside knowledge.
2. If the evidence is insufficient, respond with exactly:
   The provided documents do not contain enough information to answer this question.
3. Do not invent facts.
4. Cite supporting evidence using the format [Source: filename, Page: number].
5. Keep the answer clear and concise.
6. Return plain text only.
7. Do not use Markdown formatting such as asterisks, bold text, headings, or numbered Markdown lists.
8. When listing multiple points, use simple lines beginning with a hyphen (-).
9. Every factual claim in the answer must be supported by the provided evidence.

EVIDENCE:
{context}

QUESTION:
{query}

ANSWER:
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text