\# Document Intelligence RAG



A PDF question-answering application and reproducible study of document chunking for scientific-document retrieval.



Built with FastAPI, React/TypeScript, MiniLM embeddings, FAISS and Gemini. The application retrieves page-linked evidence and generates answers with source citations and insufficient-evidence refusal instructions.



\## Research highlights



\- Evaluated four retrieval conditions on \*\*623 queries across SciFact and NFCorpus\*\*.

\- Compared BM25, whole-document MiniLM, and 400- and 600-character chunk configurations.

\- Used exact document-level maximum-score aggregation for the journal experiments.

\- Reported paired bootstrap uncertainty and encoder input-length diagnostics.

\- Completed a separate 12-question citation/refusal behavior check.

\- Manuscript submitted to \*\*SN Computer Science\*\*, September 2026. \*\*Not yet accepted or published.\*\*



\## Application architecture



```mermaid

flowchart TD

&#x20;   A\["PDF upload"] --> B\["Page-preserving text extraction"]

&#x20;   B --> C\["Sentence-aware character-budget chunks"]

&#x20;   C --> D\["MiniLM embeddings"]

&#x20;   D --> E\["Persistent FAISS index and metadata"]

&#x20;   Q\["User question"] --> F\["MiniLM query embedding"]

&#x20;   F --> G\["Top-k evidence retrieval"]

&#x20;   E --> G

&#x20;   G --> H\["Evidence-constrained Gemini generation"]

&#x20;   H --> I\["Answer with source and page citations"]

```



MiniLM produces normalized 384-dimensional vectors. FAISS `IndexFlatIP` performs exact inner-product search, equivalent to cosine ranking for these normalized vectors.



The application retrieves PDF chunks. The journal evaluation separately ranks benchmark documents using the maximum similarity across \*\*all\*\* chunks belonging to each document.



\## Retrieval results



The journal suite uses SciFact's 300 test queries and 5,183 documents, and NFCorpus's 323 test queries and 3,633 documents.



| Dataset | Configuration | Hit@1 | MRR@10 | nDCG@10 | Recall@10 |

|---|---|---:|---:|---:|---:|

| SciFact | BM25 | 0.5333 | 0.6276 | 0.6617 | 0.7909 |

| SciFact | MiniLM whole document | 0.5033 | 0.6068 | 0.6484 | 0.7883 |

| SciFact | MiniLM 400 characters | 0.5767 | 0.6570 | 0.6961 | 0.8250 |

| SciFact | MiniLM 600 characters | 0.5300 | 0.6396 | 0.6830 | 0.8250 |

| NFCorpus | BM25 | 0.4334 | 0.5151 | 0.3069 | 0.1491 |

| NFCorpus | MiniLM whole document | 0.4272 | 0.5104 | 0.3186 | 0.1589 |

| NFCorpus | MiniLM 400 characters | 0.4303 | 0.5279 | 0.3374 | 0.1677 |

| NFCorpus | MiniLM 600 characters | 0.4365 | 0.5221 | 0.3354 | 0.1630 |



The 400-character configuration achieved the highest observed MRR@10 and nDCG@10 among these conditions on both datasets. It was not best on every metric.



Paired, exploratory 95% bootstrap intervals for its nDCG@10 improvement:



| Dataset | Comparator | Difference | 95% interval |

|---|---|---:|---|

| SciFact | Whole-document MiniLM | +0.04769 | \[0.01943, 0.07624] |

| SciFact | BM25 | +0.03442 | \[-0.00058, 0.07049] |

| NFCorpus | Whole-document MiniLM | +0.01873 | \[0.00720, 0.03104] |

| NFCorpus | BM25 | +0.03042 | \[0.01110, 0.04998] |



Intervals use 10,000 paired query-bootstrap replicates with seed 42 and are not adjusted for multiple comparisons. The SciFact interval against BM25 includes zero.



\### Truncation and interpretation



The recorded MiniLM input limit is 256 tokens. Approximately 71.58% of SciFact whole-document inputs and 79.05% of NFCorpus whole-document inputs exceeded that limit.



Small chunks change representation granularity and preserve access to later document text. These experiments do not isolate those effects. They do not establish a universally optimal chunk size or state-of-the-art retrieval performance.



The chunker uses punctuation-based sentence splitting, character targets and one-sentence overlap. It is not semantic chunking, and its character targets are not strict token limits.



\## Separate generation evaluation



A historical evaluation used 12 questions about a controlled six-page PDF.



| Behavior | Result |

|---|---:|

| Supported answer with citation syntax | 5/6 |

| Expected refusal for unsupported question | 6/6 |

| Overall expected behavior | 11/12 |



These measure citation presence and refusal behavior, not general factual accuracy or claim-level entailment. One supported question received a false refusal.



Generation was not evaluated on the SciFact or NFCorpus benchmark queries. Historical retrieved contexts and per-response model versions were not recorded, limiting exact replay.



\## Manuscript and research artifacts



\*\*Title:\*\* Evaluating Chunking Strategies for Retrieval Augmented Generation over Scientific Documents  

\*\*Author:\*\* Syed Mohammed Rayyan  

\*\*Status:\*\* Submitted to SN Computer Science, September 2026.



\- \[Submission supplementary archive](https://github.com/rayyandeys/document-intelligent-rag/blob/2e9f270987561cbc90cc439864704d9996733571/ESM\_1.zip)

\- \[Saved journal experiment outputs](experiments/journal/)

\- \[Earlier project technical report](research/paper.md)

\- \[Controlled generation questions](research/generation\_evaluation\_questions.json)



The earlier technical report describes a previous stage of the project and is not the submitted manuscript.



The supplementary archive contains the source snapshot, saved rankings, manifests, summaries and truncation analysis. The source snapshot was collected after execution; it is not a contemporaneous cryptographic record of the executed code.



ChatGPT/Codex assisted with implementation, analysis and manuscript preparation, as disclosed in the manuscript. The human author remains responsible for the work.



\## Technology



| Component | Technology |

|---|---|

| API | Python, FastAPI, Uvicorn |

| PDF extraction | PyMuPDF |

| Embeddings | Sentence Transformers, all-MiniLM-L6-v2 |

| Application vector search | FAISS IndexFlatIP |

| Generation | Google GenAI SDK / Gemini |

| Evaluation | NumPy, pandas, benchmark qrels |

| Frontend | React, TypeScript, Vite |



\## Local setup



```bash

git clone https://github.com/rayyandeys/document-intelligent-rag.git

cd document-intelligent-rag

```



Create and activate a virtual environment.



Windows CMD:



```cmd

python -m venv .venv

.venv\\Scripts\\activate

```



macOS/Linux:



```bash

python3 -m venv .venv

source .venv/bin/activate

```



Install dependencies:



```bash

python -m pip install -r requirements.txt

```



Create a `.env` file in the project root:



```dotenv

GEMINI\_API\_KEY=your\_api\_key\_here

```



Keep API keys private. Gemini generation requires a valid key and available quota.



\### Start the API



From the project root:



```bash

python -m uvicorn app.main --reload

```



\- Local API: http://127.0.0.1:8000

\- Interactive API documentation: http://127.0.0.1:8000/docs



\### Start the frontend



In a separate terminal:



```bash

cd frontend

npm install

npm run dev

```



Open the URL printed by Vite, usually http://localhost:5173.



If PowerShell blocks `npm.ps1`, use `npm.cmd install` and `npm.cmd run dev`.



\## API overview



| Endpoint | Purpose |

|---|---|

| `GET /` | Application information |

| `GET /health` | Health check |

| `POST /documents/upload` | Upload, extract, chunk and index a PDF |

| `POST /query` | Retrieve evidence and generate an answer |



Use the interactive API documentation for request schemas.



The upload pipeline preserves the original filename and page metadata for citations. Persisted indexes and chunk metadata support later queries.



Refusal is prompted through Gemini; there is no explicit retrieval-score threshold guaranteeing refusal.



\## Reproducing the experiments



\### Journal retrieval suite



From the project root with the Python environment active:



```bash

python -m experiments.run\_journal\_suite

```



This evaluates BM25 and three MiniLM representations on SciFact and NFCorpus. Dataset/model downloads require network access, and dense encoding can take substantial time on a CPU.



The saved runs record model revision:



```text

1110a243fdf4706b3f48f1d95db1a4f5529b4d41

```



Check the manifests for package versions, input hashes and configuration. Numerical reproduction depends on the recorded environment and model.



\### Truncation analysis



After the retrieval suite has produced its required outputs:



```bash

python -m experiments.analyze\_truncation

```



Saved rankings, manifests, summaries and length diagnostics are available under `experiments/journal/`.



\### Earlier SciFact experiments



```bash

python -m experiments.run\_scifact\_baseline

python -m experiments.compare\_scifact\_chunk\_sizes

```



The earlier chunk-size sweep used 400, 600, 800, 1000 and 1200 characters. It retrieved the top 50 chunks and deduplicated them into document rankings.



The journal suite instead uses exhaustive maximum-score aggregation across all chunks per document. Results from these protocols should not be combined into one controlled comparison.



The 400- and 600-character budgets were informed by earlier SciFact test-set exploration; this is not a preregistered confirmatory study.



\### Generation behavior check



Place the original controlled six-page PDF at `data/raw/sample.pdf`. An arbitrary replacement PDF will not reproduce the evaluation.



```bash

python -m experiments.run\_generation\_evaluation

```



The runner uses `research/generation\_evaluation\_questions.json`, saves completed results incrementally and resumes from existing results. Reruns can incur API usage and need not reproduce historical model responses.



\## Repository guide



| Path | Contents |

|---|---|

| `app/` | FastAPI application |

| `frontend/` | React/TypeScript interface |

| `src/ingestion/` | Document extraction |

| `src/retrieval/` | Chunking, embeddings and retrieval |

| `src/generation/` | Evidence-constrained generation |

| `src/evaluation/` | Evaluation utilities |

| `experiments/` | Experiment runners |

| `experiments/journal/` | Saved research evidence |

| `research/` | Reports and evaluation questions |

| `tests/` | Application/project tests |



Runtime uploads, downloaded corpora, generated indexes, credentials and dependency folders are excluded from version control.



\## Limitations and next research directions



The study covers one encoder, one simple chunking family and two related-domain datasets. BM25 is untuned. It does not benchmark semantic chunkers, rerankers, latency, storage cost or approximate vector search.



Useful extensions include coverage-matched token windows, semantic segmentation, additional encoders, independent development-set tuning, and larger generation evaluations with claim-level citation verification.



\## Author



\*\*Syed Mohammed Rayyan\*\*  

B.Tech Software Engineering student, SRM Institute of Science and Technology.

