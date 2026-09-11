import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const INSUFFICIENT_EVIDENCE_MESSAGE =
  "The provided documents do not contain enough information to answer this question.";


interface UploadResponse {
  document_id: string;
  filename: string;
  pages: number;
  chunks: number;
  status: string;
}

interface Source {
  chunk_id: number;
  page: number;
  source: string;
  score: number;
  text: string;
}

interface QueryResponse {
  document_id: string;
  question: string;
  answer: string;
  sources: Source[];
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [document, setDocument] =
    useState<UploadResponse | null>(null);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);

  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const insufficientEvidence =
  answer.trim() === INSUFFICIENT_EVIDENCE_MESSAGE;

  async function uploadDocument() {
    if (!file) {
      setError("Choose a PDF before uploading.");
      return;
    }

    setUploading(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `${API_URL}/documents/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Document upload failed."
        );
      }

      setDocument(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Document upload failed."
      );
    } finally {
      setUploading(false);
    }
  }

  async function askQuestion() {
    if (!document) {
      setError("Upload and index a document first.");
      return;
    }

    if (!question.trim()) {
      setError("Enter a question.");
      return;
    }

    setAsking(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(
        `${API_URL}/query`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            document_id: document.document_id,
            question: question.trim(),
            top_k: 5,
          }),
        }
      );

      const data: QueryResponse & {
        detail?: string;
      } = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Question processing failed."
        );
      }

      setAnswer(data.answer);
      setSources(data.sources);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Question processing failed."
      );
    } finally {
      setAsking(false);
    }
  }

  function resetDocument() {
    setFile(null);
    setDocument(null);
    setQuestion("");
    setAnswer("");
    setSources([]);
    setError("");
  }

  return (
    <div className="app-shell">
      {/* Decorative liquid background */}
      <div className="ambient-background">
        <div className="orb orb-one" />
        <div className="orb orb-two" />
        <div className="orb orb-three" />
        <div className="grid-overlay" />
      </div>

      {/* Navigation */}
      <nav className="navbar glass">
        <div className="brand">
          <div className="brand-symbol">
            <span>DI</span>
          </div>

          <div className="brand-copy">
            <strong>Document Intelligence</strong>
            <span>Research RAG System</span>
          </div>
        </div>

        <div className="nav-status">
          <span className="status-pulse" />
          System operational
        </div>
      </nav>

      <main>
        {/* HERO */}
        <section className="hero-section">
          <div className="hero-content">
            <div className="hero-badge glass">
              <span className="mini-pulse" />
              EVIDENCE-AWARE RAG
            </div>

            <h1>
              Intelligence grounded
              <br />
              <span className="gradient-text">
                in your documents.
              </span>
            </h1>

            <p className="hero-description">
              A research-driven retrieval augmented generation
              system that transforms documents into searchable
              evidence using sentence-aware chunking, semantic
              embeddings, persistent vector retrieval and
              grounded generation.
            </p>

            <div className="hero-actions">
              <a
                href="#workspace"
                className="primary-button hero-button"
              >
                Query a document
                <span>↓</span>
              </a>

              <a
                href="#architecture"
                className="secondary-button"
              >
                Explore architecture
              </a>
            </div>

            <div className="technology-strip">
              <span>Python</span>
              <i />
              <span>FastAPI</span>
              <i />
              <span>MiniLM</span>
              <i />
              <span>FAISS</span>
              <i />
              <span>Gemini</span>
            </div>
          </div>

          {/* Floating architecture preview */}
          <div className="hero-visual glass">
            <div className="visual-header">
              <div>
                <span className="visual-label">
                  LIVE PIPELINE
                </span>
                <h3>Retrieval Architecture</h3>
              </div>

              <span className="live-indicator">
                ACTIVE
              </span>
            </div>

            <div className="mini-architecture">
              <div className="architecture-node">
                <span className="node-number">01</span>
                <div>
                  <strong>PDF</strong>
                  <small>Page-aware extraction</small>
                </div>
              </div>

              <div className="flow-line">
                <span />
              </div>

              <div className="architecture-node">
                <span className="node-number">02</span>
                <div>
                  <strong>Chunks</strong>
                  <small>Sentence-aware · 400 chars</small>
                </div>
              </div>

              <div className="flow-line">
                <span />
              </div>

              <div className="architecture-node">
                <span className="node-number">03</span>
                <div>
                  <strong>MiniLM</strong>
                  <small>384-d normalized vectors</small>
                </div>
              </div>

              <div className="flow-line">
                <span />
              </div>

              <div className="architecture-node active-node">
                <span className="node-number">04</span>
                <div>
                  <strong>FAISS</strong>
                  <small>Persistent vector index</small>
                </div>
              </div>

              <div className="flow-line">
                <span />
              </div>

              <div className="architecture-node">
                <span className="node-number">05</span>
                <div>
                  <strong>Gemini</strong>
                  <small>Evidence-grounded answer</small>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* RESEARCH METRICS */}
        <section className="metrics-section">
          <div className="metric glass">
            <span className="metric-label">
              EVALUATION
            </span>
            <strong>300</strong>
            <p>SciFact queries</p>
          </div>

          <div className="metric glass">
            <span className="metric-label">
              HIT@1
            </span>
            <strong>57.67%</strong>
            <p>
              <span className="metric-improvement">
                +7.34 pp
              </span>{" "}
              vs document baseline
            </p>
          </div>

          <div className="metric glass">
            <span className="metric-label">
              MRR@10
            </span>
            <strong>0.6570</strong>
            <p>
              <span className="metric-improvement">
                +0.0502
              </span>{" "}
              vs baseline
            </p>
          </div>

          <div className="metric glass">
            <span className="metric-label">
              EMBEDDINGS
            </span>
            <strong>384D</strong>
            <p>Normalized MiniLM vectors</p>
          </div>
        </section>

        {/* WORKSPACE */}
        <section
          className="workspace-section"
          id="workspace"
        >
          <div className="section-intro">
            <span className="section-tag">
              INTERACTIVE PIPELINE
            </span>

            <h2>Query your evidence.</h2>

            <p>
              Upload a text-based PDF. The backend extracts,
              chunks, embeds and persists its vector index
              before accepting questions.
            </p>
          </div>

          <div className="workspace-grid">
            {/* Upload */}
            <article className="workspace-card glass">
              <div className="card-header">
                <div>
                  <span className="card-step">01</span>
                  <h3>Document ingestion</h3>
                </div>

                {document && (
                  <span className="indexed-badge">
                    <span />
                    INDEXED
                  </span>
                )}
              </div>

              {!document ? (
                <>
                  <label className="upload-zone">
                    <input
                      type="file"
                      accept=".pdf,application/pdf"
                      onChange={(event) => {
                        const selected =
                          event.target.files?.[0] ||
                          null;

                        setFile(selected);
                        setError("");
                      }}
                    />

                    <div className="upload-bubble">
                      <span>↑</span>
                    </div>

                    <strong>
                      {file
                        ? file.name
                        : "Drop in your research document"}
                    </strong>

                    <p>
                      Select a text-based PDF to build a
                      persistent semantic index.
                    </p>

                    <span className="file-type">
                      PDF
                    </span>
                  </label>

                  <button
                    className="primary-button full-button"
                    onClick={uploadDocument}
                    disabled={uploading || !file}
                  >
                    {uploading ? (
                      <>
                        <span className="button-loader" />
                        Building vector index...
                      </>
                    ) : (
                      <>
                        Build document index
                        <span>→</span>
                      </>
                    )}
                  </button>
                </>
              ) : (
                <div className="indexed-document">
                  <div className="pdf-bubble">
                    PDF
                  </div>

                  <div className="document-details">
                    <span className="document-label">
                      ACTIVE DOCUMENT
                    </span>

                    <strong>
                      {document.filename}
                    </strong>

                    <div className="document-stats">
                      <span>
                        {document.pages} pages
                      </span>
                      <span>
                        {document.chunks} chunks
                      </span>
                      <span>FAISS indexed</span>
                    </div>
                  </div>

                  <button
                    className="replace-button"
                    onClick={resetDocument}
                  >
                    Replace
                  </button>
                </div>
              )}
            </article>

            {/* Question */}
            <article
              className={`workspace-card glass ${
                !document ? "disabled-card" : ""
              }`}
            >
              <div className="card-header">
                <div>
                  <span className="card-step">02</span>
                  <h3>Semantic query</h3>
                </div>

                <span className="top-k-label">
                  TOP-K · 5
                </span>
              </div>

              <div className="query-area">
                <textarea
                  value={question}
                  disabled={!document}
                  placeholder={
                    document
                      ? "Ask a question grounded in the uploaded document..."
                      : "Index a document to activate semantic search..."
                  }
                  onChange={(event) => {
                    setQuestion(event.target.value);
                    setError("");
                  }}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey
                    ) {
                      event.preventDefault();

                      if (
                        document &&
                        question.trim() &&
                        !asking
                      ) {
                        askQuestion();
                      }
                    }
                  }}
                />

                <div className="query-meta">
                  <span>
                    Enter to query · Shift + Enter
                    for newline
                  </span>

                  <span>
                    FAISS · cosine-equivalent IP
                  </span>
                </div>
              </div>

              <button
                className="primary-button full-button"
                onClick={askQuestion}
                disabled={
                  !document ||
                  !question.trim() ||
                  asking
                }
              >
                {asking ? (
                  <>
                    <span className="button-loader" />
                    Retrieving evidence...
                  </>
                ) : (
                  <>
                    Run grounded query
                    <span>→</span>
                  </>
                )}
              </button>
            </article>
          </div>

          {error && (
            <div className="error-panel glass">
              <div className="error-icon">!</div>

              <div>
                <strong>Pipeline error</strong>
                <p>{error}</p>
              </div>
            </div>
          )}
        </section>

        {/* ANSWER */}
        {(answer || asking) && (
          <section className="results-section">
            <div className="section-intro">
              <span className="section-tag">
                RETRIEVAL OUTPUT
              </span>

              <h2>Document response.</h2>
            </div>

            <div className="answer-card glass">
              <div className="answer-card-header">
                <div>
                  <span className="card-step">03</span>
                  <h3>Generated answer</h3>
                </div>

                <div
  className={`evidence-badge ${
    insufficientEvidence ? "insufficient-badge" : ""
  }`}
>
  <span className="status-pulse" />

  {asking
    ? "GENERATING ANSWER"
    : insufficientEvidence
      ? "INSUFFICIENT EVIDENCE"
      : "BASED ON RETRIEVED TEXT"}
</div>
              </div>

              {asking ? (
                <div className="answer-loading">
                  <div className="liquid-loader">
                    <span />
                    <span />
                    <span />
                  </div>

                  <div>
                    <strong>
                      Searching semantic space
                    </strong>
                    <p>
                      Retrieving evidence and generating
                      a response from the retrieved text...
                    </p>
                  </div>
                </div>
              ) : (
                <p className="answer-text">
                  {answer}
                </p>
              )}
            </div>

            {!asking && sources.length > 0 && (
              <div className="evidence-area">
                <div className="evidence-heading">
                  <div>
                    <span className="section-tag">
  {insufficientEvidence
    ? "RETRIEVAL DIAGNOSTICS"
    : "PROVENANCE"}
</span>

<h3>
  {insufficientEvidence
    ? "Retrieved candidates"
    : "Retrieved evidence"}
</h3>
                  </div>

                  <span>
  {insufficientEvidence
    ? `${sources.length} candidates · insufficient support`
    : `${sources.length} semantic matches`}
</span>
                </div>

                <p>Similarity measures text relevance, not answer confidence. Check the cited passages to verify the answer.</p>

                <div className="evidence-grid">
                  {sources.map((source, index) => (
                    <article
                      className="evidence-card glass"
                      key={`${source.chunk_id}-${index}`}
                    >
                      <div className="evidence-top">
                        <span className="evidence-rank">
                          0{index + 1}
                        </span>

                        <div className="similarity">
                          <span>
                            Similarity
                          </span>
                          <strong>
                            {(
                              source.score * 100
                            ).toFixed(1)}
                            %
                          </strong>
                        </div>
                      </div>

                      <div className="similarity-track">
                        <div
                          className="similarity-fill"
                          style={{
                            width: `${Math.max(
                              0,
                              Math.min(
                                100,
                                source.score * 100
                              )
                            )}%`,
                          }}
                        />
                      </div>

                      <div className="source-location">
                        <span>
                          PAGE {source.page}
                        </span>
                        <span>
                          CHUNK {source.chunk_id}
                        </span>
                      </div>

                      <p>{source.text}</p>
                    </article>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {/* ARCHITECTURE */}
        <section
          className="architecture-section"
          id="architecture"
        >
          <div className="section-intro centered">
            <span className="section-tag">
              SYSTEM DESIGN
            </span>

            <h2>
              Designed as a retrieval pipeline,
              <br />
              not a chatbot wrapper.
            </h2>

            <p>
              Retrieval and generation remain separate,
              observable stages so each component can be
              evaluated and improved independently.
            </p>
          </div>

          <div className="system-flow glass">
            <div className="system-node">
              <span>01</span>
              <strong>Client</strong>
              <small>React + TypeScript</small>
            </div>

            <div className="system-arrow">
              <span>HTTP</span>
              →
            </div>

            <div className="system-node">
              <span>02</span>
              <strong>API Layer</strong>
              <small>FastAPI</small>
            </div>

            <div className="system-arrow">
              <span>INGEST</span>
              →
            </div>

            <div className="system-node">
              <span>03</span>
              <strong>Embedding</strong>
              <small>all-MiniLM-L6-v2</small>
            </div>

            <div className="system-arrow">
              <span>SEARCH</span>
              →
            </div>

            <div className="system-node highlight-node">
              <span>04</span>
              <strong>Vector Store</strong>
              <small>FAISS IndexFlatIP</small>
            </div>

            <div className="system-arrow">
              <span>CONTEXT</span>
              →
            </div>

            <div className="system-node">
              <span>05</span>
              <strong>Generation</strong>
              <small>Gemini + citations</small>
            </div>
          </div>

          <div className="design-principles">
            <div className="principle glass">
              <span>Persistence</span>
              <p>
                Vector indexes and chunk metadata are
                persisted so documents do not require
                re-embedding for every query.
              </p>
            </div>

            <div className="principle glass">
              <span>Grounding</span>
              <p>
                Generation is constrained to retrieved
                evidence with page-level provenance.
              </p>
            </div>

            <div className="principle glass">
              <span>Evaluation</span>
              <p>
                Retrieval configuration is selected through
                controlled benchmark experiments rather than
                intuition alone.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <div>
          <strong>Document Intelligence</strong>
          <span>Research-oriented RAG system</span>
        </div>

        <div className="footer-stack">
          React · FastAPI · MiniLM · FAISS · Gemini
        </div>
      </footer>
    </div>
  );
}

export default App;