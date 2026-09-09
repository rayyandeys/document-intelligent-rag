from pathlib import Path
import shutil
import uuid
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.index_store import (
    save_retrieval_index,
    load_retrieval_index,
)
from src.generation.generator import Generator


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Document Intelligence RAG API",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

frontend_url = os.getenv("FRONTEND_URL")

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if frontend_url:
    allowed_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

UPLOAD_DIR = Path("data/uploads")
INDEX_DIR = Path("data/indexes")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# MODELS
# --------------------------------------------------

embedder = Embedder()
generator = Generator()


# --------------------------------------------------
# REQUEST SCHEMAS
# --------------------------------------------------

class QueryRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = 5


# --------------------------------------------------
# BASIC ENDPOINTS
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Document Intelligence RAG API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# DOCUMENT UPLOAD + INDEXING
# --------------------------------------------------

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    document_id = str(uuid.uuid4())

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    document_index_dir = INDEX_DIR / document_id

    try:
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        pages = load_pdf(
            str(pdf_path)
        )

        # Keep the UUID for internal storage while preserving
        # the user's original filename for source attribution.
        for page in pages:
            page["source"] = file.filename

        chunks = smart_chunk_pages(
            pages,
            chunk_size=400,
            overlap_sentences=1,
        )

        embeddings = embedder.embed_chunks(
            chunks
        )

        retriever = FAISSRetriever(
            chunks=chunks,
            embeddings=embeddings,
            embedder=embedder,
        )

        save_retrieval_index(
            retriever=retriever,
            chunks=chunks,
            directory=str(document_index_dir),
        )

    except Exception as error:
        pdf_path.unlink(
            missing_ok=True
        )

        shutil.rmtree(
            document_index_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {error}",
        )

    finally:
        await file.close()

    return {
        "document_id": document_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "status": "indexed",
    }


# --------------------------------------------------
# RAG QUERY
# --------------------------------------------------

@app.post("/query")
def query_document(
    request: QueryRequest,
):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0.",
        )

    document_index_dir = (
        INDEX_DIR / request.document_id
    )

    if not document_index_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Document index not found.",
        )

    try:
        retriever = load_retrieval_index(
            directory=str(document_index_dir),
            embedder=embedder,
        )

        results = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        answer = generator.generate(
            query=request.question,
            retrieved_chunks=results,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {error}",
        )

    sources = []

    for result in results:
        sources.append(
            {
                "chunk_id": result["chunk_id"],
                "page": result["page"],
                "source": result["source"],
                "score": result["score"],
                "text": result["text"],
            }
        )

    return {
        "document_id": request.document_id,
        "question": request.question,
        "answer": answer,
        "sources": sources,
    }