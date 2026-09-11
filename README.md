Document Intelligence RAG

A research-oriented Retrieval-Augmented Generation (RAG) system for scientific documents.

The project combines a working PDF question-answering application with controlled retrieval experiments. It studies how sentence-aware chunk size affects retrieval effectiveness and separately evaluates evidence-constrained generation behavior.

Highlights

PDF ingestion with page-level source metadata

Sentence-aware character-budget chunking with overlap

sentence-transformers/all-MiniLM-L6-v2 embeddings

384-dimensional normalized dense vectors

Exact FAISS IndexFlatIP retrieval

Persistent indexes and chunk metadata

Gemini-based evidence-constrained generation

Source/page citations and explicit insufficient-evidence refusal behavior

FastAPI backend

React + TypeScript frontend

SciFact retrieval benchmark

Controlled chunk-size ablation

Separate generation grounding/refusal evaluation

Research paper-style technical report

Research Question

How does document chunk size affect retrieval effectiveness in a retrieval-augmented generation pipeline for scientific documents?

A secondary question examines whether evidence-constrained generation can provide cited answers for supported questions while refusing questions unsupported by the retrieved evidence.

Architecture

PDF
|
v
PyMuPDF Text Extraction
|
v
Sentence-Aware Character-Budget Chunking
|
v
all-MiniLM-L6-v2 Embeddings
|
v
Normalized 384-D Vectors
|
v
FAISS IndexFlatIP
|
+----------------------+
|
User Question           |
|                  |
v                  |
MiniLM Query Embedding  |
|                  |
+----------------->|
v
Top-k Retrieval
|
v
Retrieved Evidence
|
v
Evidence-Constrained Gemini
|
v
Answer + Source/Page Citations

The application uses normalized embeddings. Therefore, inner-product ranking with FAISS IndexFlatIP is cosine-equivalent for these vectors. IndexFlatIP performs exact search; this project does not claim approximate-nearest-neighbor retrieval.

Retrieval Results

Retrieval was evaluated on 300 SciFact test queries over a corpus of 5,183 scientific documents.

Target chunk size

Indexed units

Hit@1

Hit@3

Hit@5

Hit@10

MRR@10

Whole document

5,183

0.5033

0.6767

0.7567

0.8000

0.6068

400 chars

36,263

0.5767

0.7133

0.7567

0.8333

0.6570

600 chars

21,565

0.5300

0.7200

0.7833

0.8333

0.6396

800 chars

14,567

0.5267

0.7033

0.7733

0.8167

0.6243

1000 chars

11,442

0.4967

0.6800

0.7700

0.8333

0.6096

1200 chars

9,304

0.5133

0.7033

0.7500

0.8200

0.6187

Main Findings

The 400-character configuration produced the strongest tested Hit@1 and MRR@10:

Hit@1: 50.33% -> 57.67%

Absolute Hit@1 improvement: +7.34 percentage points

MRR@10: 0.6068 -> 0.6570

Relative MRR@10 improvement: approximately 8.3%

However, 600-character chunks achieved the highest Hit@3 and Hit@5. The experiment therefore does not claim that 400 characters is universally optimal. The preferred granularity depends on retrieval depth and system objective.

A six-question controlled development set initially favored 600-character chunks, while the larger 300-query SciFact evaluation favored 400 characters by MRR@10. This difference is one reason the project treats the larger benchmark as the stronger basis for conclusions.

Generation Evaluation

A separate 12-question controlled evaluation is designed to test:

six questions supported by the controlled document;

six questions deliberately unsupported by that document.

Supported behavior currently requires a non-refusal answer containing source/page citation syntax. Unsupported behavior requires the system's exact insufficient-evidence response.

All 12 controlled questions completed: supported citation behavior was 5/6 (83.33%), unsupported refusal behavior was 6/6 (100.00%), and overall expected behavior was 11/12 (91.67%). These are citation/refusal behavior scores on a small controlled set, not general factual accuracy.

This evaluation is intentionally modest. Citation syntax alone does not prove that every generated claim is entailed by the cited evidence, so these results should not be interpreted as a complete factuality or hallucination benchmark.

Tech Stack

Backend and Research

Python 3.12

FastAPI

Uvicorn

PyMuPDF

Sentence Transformers

all-MiniLM-L6-v2

FAISS

NumPy

Google GenAI SDK / Gemini

Hugging Face Datasets

pandas

Frontend

React

TypeScript

Vite

Project Structure

document-intelligent-rag/
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── uploads/
│   └── indexes/
├── experiments/
│   ├── compare_retrieval_configs.py
│   ├── compare_scifact_chunk_sizes.py
│   ├── run_scifact_baseline.py
│   ├── run_generation_evaluation.py
│   ├── scifact_baseline_results.csv
│   ├── scifact_chunk_size_results.csv
│   └── generation_evaluation_results.csv
├── frontend/
├── research/
│   ├── paper.md
│   └── generation_evaluation_questions.json
├── src/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   └── retrieval/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md

Runtime documents, generated indexes, environment files, and other local artifacts are excluded from Git where appropriate.

Installation

Clone the repository

git clone https://github.com/rayyandeys/document-intelligent-rag.git
cd document-intelligent-rag

Create a Python virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

Install Python dependencies

python -m pip install -r requirements.txt

Configure Gemini

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

Do not commit .env or API keys.

Running the Application

Backend

From the project root:

python -m uvicorn app.main --reload

Default development API:

http://127.0.0.1:8000

Interactive FastAPI documentation:

http://127.0.0.1:8000/docs

Frontend

Open a second terminal:

cd frontend
npm install
npm run dev

On Windows PowerShell systems where script execution blocks npm.ps1, use:

npm.cmd install
npm.cmd run dev

Default Vite development URL:

http://localhost:5173

API Overview

GET /

Basic application information.

GET /health

Health check.

POST /documents/upload

Uploads and indexes a PDF.

The pipeline:

stores the uploaded document under an internal identifier;

extracts page text;

preserves the original filename as citation metadata;

creates sentence-aware chunks;

embeds the chunks;

builds a FAISS index;

persists index and chunk metadata.

POST /query

Loads a persisted document index, retrieves relevant chunks, sends evidence to the generator, and returns:

generated answer;

source filename;

page number;

chunk identifier;

similarity score;

retrieved text.

Running the Retrieval Experiments

SciFact whole-document baseline

python -m experiments.run_scifact_baseline

SciFact chunk-size comparison

python -m experiments.compare_scifact_chunk_sizes

The chunk-size experiment evaluates target budgets of:

400, 600, 800, 1000, 1200

with one-sentence overlap.

For chunked SciFact retrieval, the experiment retrieves the top 50 chunks, deduplicates them by document ID, and uses the first ten unique documents for document-level Hit@K and MRR@10 evaluation.

This is a chunk-first ranking approximation rather than exhaustive document-level max-score aggregation.

Running the Generation Evaluation

Place the original six-page controlled PDF at data/raw/sample.pdf. The runner reads the tracked question set from research/generation_evaluation_questions.json.

python -m experiments.run_generation_evaluation

The runner saves completed results incrementally and resumes from existing results, so a quota or server interruption does not require rerunning completed questions.

The reproducible question set is available at:

research/generation_evaluation_questions.json

Research Paper

The paper-style technical report is stored at:

research/paper.md

It documents:

research questions;

related work;

architecture;

methodology;

SciFact setup;

retrieval metrics;

chunk-size results;

controlled generation evaluation;

discussion;

limitations;

reproducibility notes.

Important Methodological Notes

The chunker is sentence-aware character-budget chunking, not semantic chunking.

MiniLM embeddings are normalized.

FAISS IndexFlatIP is used for exact inner-product retrieval.

With normalized vectors, the implemented inner-product ranking is cosine-equivalent.

No explicit retrieval-score threshold currently determines refusal.

Gemini is instructed to refuse unsupported questions through the generation prompt.

The SciFact chunk experiment retrieves top-50 chunks and deduplicates them into document rankings.

The project does not present FAISS latency or scalability benchmarks.

The generation evaluation is separate from retrieval evaluation.

API quota failures are not treated as model-answer failures.

Limitations

The current study uses one embedding model and one sentence-aware chunking family. It does not compare semantic chunkers, rerankers, multiple embedding models, or approximate vector indexes.

SciFact provides document-level relevance judgments, whereas the product workflow operates on PDF chunks with page metadata. The two evaluations therefore measure related but not identical retrieval tasks.

The generation evaluation is small and complete for the planned 12 questions. Its automated citation criterion verifies citation presence, not full claim-level entailment.

Future Work

Potential extensions include:

additional embedding models;

semantic and structure-aware chunking;

reranking;

document-level max-score aggregation;

nDCG and broader BEIR-style evaluation;

larger supported/unsupported generation sets;

entailment-based citation verification;

human evaluation;

deployment-oriented performance benchmarking.

Author

Syed Mohammed Rayyan

B.Tech Software Engineering student.

Status

Active research/engineering project. Core retrieval experiments and the end-to-end RAG application are implemented. The planned 12-question controlled generation evaluation is complete (11/12 expected behaviors).