Evaluating Chunking Strategies for Retrieval-Augmented Generation over Scientific Documents

Syed Mohammed Rayyan

Research paper-style technical report. This manuscript is a working draft; the controlled generation evaluation remains partially incomplete because of external API quota limits.

Abstract

Retrieval-Augmented Generation (RAG) systems depend on the quality of retrieved evidence to produce grounded responses, making document segmentation an important design decision. This work investigates how sentence-aware chunk size affects retrieval effectiveness in a RAG pipeline for scientific documents. An end-to-end prototype was implemented using PDF text extraction, sentence-aware character-budget chunking with overlap, 384-dimensional MiniLM embeddings, exact vector retrieval, persistent FAISS indexing, and evidence-constrained language-model generation with source and page citations.

Retrieval performance was evaluated on the SciFact benchmark using 5,183 scientific documents and 300 evaluated test queries. Five chunk-size configurations with target budgets of 400, 600, 800, 1000, and 1200 characters were compared while holding the embedding model and one-sentence overlap strategy constant. A whole-document MiniLM retrieval configuration was used as a baseline. The 400-character configuration achieved the highest Hit@1 (57.67%) and MRR@10 (0.6570), compared with 50.33% and 0.6068 for the whole-document baseline. This is a 7.34 percentage-point improvement in Hit@1 and approximately 8.3% relative improvement in MRR@10. The 600-character configuration, however, achieved the highest Hit@3 (72.00%) and Hit@5 (78.33%), showing that the preferred chunk size depends on retrieval depth and objective rather than a single universally optimal setting.

The system also includes a controlled generation evaluation that separately examines citation behavior for supported questions and refusal behavior for unsupported questions. At the present checkpoint, five supported-query trials have been recorded: four produced non-refusal answers containing source/page citations, while one supported query was incorrectly refused despite a high retrieval similarity score. The remaining planned trials were not completed because of external API quota exhaustion and are therefore not treated as generation failures. These observations reinforce the need to evaluate retrieval and generation as distinct stages of a RAG pipeline.

1. Introduction

Retrieval-Augmented Generation combines information retrieval with language-model generation by retrieving external evidence before producing an answer. This architecture is useful for scientific-document question answering, where responses should be grounded in identifiable source material rather than generated solely from a model's internal knowledge. A RAG system can nevertheless fail even when its language model is capable: the retrieval stage may omit the needed evidence, retrieve evidence at an unhelpful granularity, or provide distracting context.

A central design choice is how source text is divided before embedding and indexing. Small chunks can represent focused concepts but may separate a statement from context needed to interpret it. Large chunks retain more context but may mix multiple ideas into one embedding, potentially reducing retrieval specificity. Overlap can preserve information near boundaries, at the cost of additional indexed chunks. These trade-offs motivate controlled evaluation rather than selecting chunk parameters by intuition alone.

This study asks:

Primary research question: How does document chunk size affect retrieval effectiveness in a retrieval-augmented generation pipeline for scientific documents?

Secondary research question: Can evidence-constrained generation provide cited answers for supported questions while refusing questions unsupported by supplied evidence?

The work contributes:

An end-to-end scientific-document RAG prototype.

A controlled chunk-size ablation across five character-budget configurations on 300 SciFact test queries.

A comparison against a whole-document dense-retrieval baseline.

A small, separate framework for examining citation and refusal behavior at the generation stage.

2. Background and Related Work

RAG pipelines generally separate ingestion, representation, retrieval, and generation. During ingestion, documents are parsed and segmented; text units are embedded into dense vectors; a query is embedded using the same representation model; and similar units are retrieved as evidence for a generator. This decomposition makes retrieval quality an upstream constraint on generation quality.

Chunking has already been studied in RAG, so this work does not claim to introduce chunk-size evaluation. Recent studies have compared fixed-size, recursive, sentence-based, semantic, and structure-aware chunking under different domains and evaluation settings. Kreileder, Reisinger, and Fischer (2026), for example, evaluated chunking strategies on academic texts and found that a cluster-based semantic approach did not outperform simpler strategies under their tested configuration. Bennani and Moslonka (2026) systematically varied chunking choices for question answering and reported that the best configuration depends on the evaluation goal and context regime. These findings support treating chunking as an empirical design variable rather than assuming one method or size is universally best.

The present work is narrower: it isolates target chunk size within a sentence-aware character-budget strategy, keeps the embedding model and overlap fixed, and evaluates retrieval on the scientific SciFact benchmark. It then connects this retrieval study to a working document-question-answering prototype and a separate controlled generation behavior evaluation. The contribution is therefore a reproducible implementation and empirical configuration study, not a claim of being the first RAG chunking investigation.

3. System Architecture

The prototype follows this pipeline:

PDF -> Text Extraction -> Sentence-Aware Chunking -> MiniLM Embeddings -> Persistent FAISS Index -> Query Embedding -> Top-k Retrieval -> Evidence-Constrained Gemini Generation -> Answer with Source/Page Citations

PDF text is extracted with PyMuPDF. Each page retains page-level metadata and the original uploaded filename for user-facing citations, while uploaded files are stored internally under generated identifiers. Chunks are embedded using sentence-transformers/all-MiniLM-L6-v2. The implementation normalizes embeddings, producing 384-dimensional vectors.

For the product retrieval path, FAISS IndexFlatIP performs exact inner-product search. With normalized vectors, the ranking is cosine-equivalent. IndexFlatIP is exact rather than approximate nearest-neighbor search.

Indexes and chunk metadata are persisted so queries can reload an existing document index without re-embedding the document. A FastAPI backend exposes document-upload and query endpoints. A React/TypeScript client provides upload, question-answering, evidence inspection, retrieval scores, architecture information, and benchmark metrics.

The generation stage uses Gemini under a prompt that restricts answers to retrieved evidence, requires source/page citations, and prescribes an exact refusal message when evidence is insufficient.

4. Methodology

4.1 Document Processing

The product workflow accepts PDF documents, extracts text page by page, and attaches page number and source filename metadata. Page metadata is carried into chunks and returned with retrieved evidence, enabling answers to cite the source document and page. The internal storage identifier is deliberately separated from the displayed source filename.

4.2 Sentence-Aware Chunking

The implemented chunker is best described as sentence-aware character-budget chunking, not semantic chunking. Text is split around sentence boundaries and accumulated until a target character budget is reached. Neighboring chunks retain one sentence of overlap in the benchmark experiments.

The main SciFact ablation uses target budgets of:

400 characters

600 characters

800 characters

1000 characters

1200 characters

This design varies retrieval granularity while avoiding arbitrary mid-sentence boundaries where possible.

4.3 Embedding and Retrieval

Text chunks and queries are encoded with all-MiniLM-L6-v2. The model maps text to a 384-dimensional dense vector space. Embeddings are normalized before similarity search.

The application uses FAISS IndexFlatIP for exact search over stored vectors. A separate NumPy exact dot-product retriever was used during development and evaluation. An equivalence test verified identical top-five rankings and scores between the NumPy and FAISS implementations for the controlled document. This validates retrieval behavior but is not a latency or scalability benchmark.

4.4 Evidence-Constrained Generation

The top retrieved chunks are assembled into an evidence context containing source and page metadata. The generator is instructed to:

Use only supplied evidence.

Avoid outside knowledge.

Cite factual claims using source/page metadata.

Return a fixed insufficiency message when the evidence does not support an answer.

No explicit similarity-score threshold currently triggers refusal. Refusal is a generation-policy behavior induced by the prompt. This distinction is important when interpreting unsupported-query tests.

5. Experimental Setup

5.1 Controlled Document Evaluation

An initial six-page scientific-document-style PDF was used as a small controlled development set. Six questions were associated with expected evidence pages. This evaluation was useful for validating ingestion, chunking, retrieval, citation metadata, and the end-to-end workflow, but it is too small to support general conclusions about optimal chunk size.

The small evaluation favored a 600-character configuration, whereas the larger SciFact experiment favored 400 characters by MRR@10. This illustrates the risk of drawing configuration conclusions from a tiny question set.

5.2 SciFact Benchmark

The primary retrieval experiment uses SciFact through the BEIR-formatted dataset. The loaded corpus contains 5,183 scientific documents. The test relevance judgments correspond to 300 unique evaluated queries, represented by 339 relevance rows in the loaded qrels. All benchmark metrics reported here use those 300 evaluated test queries.

A whole-document baseline represents each SciFact document with one MiniLM embedding. The chunked configurations divide each document using the sentence-aware character-budget chunker with one-sentence overlap.

Configuration

Indexed units

Whole document

5,183

400 characters

36,263 chunks

600 characters

21,565 chunks

800 characters

14,567 chunks

1000 characters

11,442 chunks

1200 characters

9,304 chunks

For the chunked benchmark, the retrieval procedure obtains the top 50 chunks and deduplicates results by document identifier, preserving the first occurrence of each document until ten unique documents are obtained. This is a chunk-first document-ranking approximation and should not be interpreted as exhaustive document-level max-score aggregation over every chunk.

5.3 Retrieval Metrics

Hit@K records whether at least one relevance-judged document appears within the first K unique retrieved documents. Hit@1, Hit@3, Hit@5, and Hit@10 are reported.

MRR@10 measures the reciprocal rank of the first relevant document when it appears within the top ten, averaged over evaluated queries.

Together, these metrics distinguish immediate first-result quality from the ability to recover evidence deeper in the retrieval list.

5.4 Generation and Grounding Evaluation

A planned 12-question controlled generation set contains six questions supported by the controlled PDF and six deliberately unsupported general-knowledge questions.

Supported behavior is currently scored as a non-refusal answer that contains the required source/page citation syntax. Unsupported behavior is scored as returning the exact insufficiency message.

This automated criterion is intentionally limited: citation syntax does not establish that every generated claim is entailed by the cited passage. The experiment should therefore not be described as a complete factuality or hallucination benchmark.

The evaluation retrieves five chunks per question. Results are written incrementally so external API failures do not erase completed trials. At the current checkpoint, five supported questions have completed. The remaining trials were interrupted by Gemini free-tier quota exhaustion and are left incomplete rather than counted as failures.

6. Results

6.1 Whole-Document Baseline

The whole-document MiniLM baseline achieved:

Hit@1 = 0.5033

Hit@3 = 0.6767

Hit@5 = 0.7567

Hit@10 = 0.8000

MRR@10 = 0.6068

across the 300 evaluated SciFact queries.

6.2 Chunk-Size Ablation

Target chunk size

Chunks

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

400

36,263

0.5767

0.7133

0.7567

0.8333

0.6570

600

21,565

0.5300

0.7200

0.7833

0.8333

0.6396

800

14,567

0.5267

0.7033

0.7733

0.8167

0.6243

1000

11,442

0.4967

0.6800

0.7700

0.8333

0.6096

1200

9,304

0.5133

0.7033

0.7500

0.8200

0.6187

The 400-character configuration achieved the highest Hit@1 (0.5767) and MRR@10 (approximately 0.6570). Relative to the whole-document baseline, Hit@1 increased from 50.33% to 57.67%, a gain of 7.34 percentage points. MRR@10 increased from 0.6068 to approximately 0.6570, an absolute gain of about 0.0502 and a relative improvement of roughly 8.3%.

The 400-character configuration was not strongest at every retrieval depth. The 600-character configuration achieved the highest Hit@3 (0.7200) and Hit@5 (0.7833). The results therefore do not support a universal claim that 400 characters is optimal. Instead, they show a retrieval-depth trade-off: 400 characters produced the strongest early-ranking performance by Hit@1 and MRR@10, while 600 characters recovered relevant documents more frequently within the top three and five.

6.3 Grounded Generation Evaluation

Five supported-query trials are currently recorded. Four produced non-refusal answers containing source/page citations, yielding 4/5 (80.0%) supported citation behavior under the automated criterion.

One supported question, asking why chunking is necessary in a RAG system, was incorrectly refused even though its highest retrieved chunk had a similarity score of approximately 0.765. This provides an example in which high retrieval similarity did not guarantee successful answer generation.

No final unsupported-question refusal rate is reported because the unsupported portion of the recorded run has not yet completed. API quota errors are operational interruptions, not model-answer failures, and are excluded from behavioral accuracy calculations until the corresponding questions produce responses.

7. Discussion

The SciFact results show that retrieval granularity materially changes ranking behavior even when the embedding model is unchanged. Moving from whole-document representations to smaller sentence-aware chunks improved first-result retrieval and reciprocal rank under the best tested configuration.

One plausible explanation is that smaller chunks reduce representational dilution: an embedding can focus on a narrower scientific claim rather than averaging information across a larger document representation. This explanation is consistent with the observed results but is not independently proven by the present experiment.

The fact that 600-character chunks outperform 400-character chunks at Hit@3 and Hit@5 is equally important. A downstream generator often receives multiple retrieved chunks, so the configuration with the best first result is not necessarily the configuration that provides the best evidence set for every application. Selection should therefore depend on retrieval depth, generation context budget, latency and storage constraints, and intended question type.

The small controlled-document evaluation and the SciFact experiment also produced different preferred configurations. The six-question development set favored 600 characters, while the 300-query SciFact benchmark favored 400 by MRR@10. This difference demonstrates why tiny hand-authored evaluations are useful for debugging but weak evidence for general configuration decisions.

The recorded generation failure provides a second lesson. The system retrieved semantically strong evidence for a supported question yet the generator returned the insufficiency message. This separates retrieval success from generation success and argues against evaluating an end-to-end RAG system with a single aggregate score. Retrieval metrics, evidence inspection, and generation behavior should be analyzed separately.

8. Limitations

This study has several limitations.

First, only one embedding model, all-MiniLM-L6-v2, is used in the main experiments. Another embedding model could change the ranking of chunk configurations.

Second, the chunking strategy is sentence-aware but simple and does not perform semantic topic-boundary detection.

Third, SciFact is evaluated at document relevance level while the product workflow retrieves PDF chunks with page metadata. The benchmark and application are related but not identical tasks.

Fourth, chunked SciFact ranking retrieves the top 50 chunks and deduplicates them into ten unique documents rather than performing exhaustive max-score aggregation across all chunks for every document.

Fifth, this report uses Hit@K and MRR@10 rather than the complete set of metrics often reported in BEIR studies, such as nDCG@10.

Sixth, the FAISS IndexFlatIP implementation is exact; no approximate-nearest-neighbor scalability claim or latency benchmark is made.

Seventh, the generation evaluation is small and incomplete at the current checkpoint. Its supported-answer criterion verifies non-refusal plus citation syntax but does not automatically prove factual entailment of every claim.

Finally, the generator is an external, nondeterministic service subject to model updates and API quotas. No explicit similarity threshold currently controls refusal; the generator decides insufficiency from the evidence and prompt.

9. Conclusion

This work implemented and evaluated an end-to-end RAG prototype for scientific documents with a focus on retrieval granularity. On 300 evaluated SciFact queries, sentence-aware chunking materially changed retrieval effectiveness.

Among the tested settings, a 400-character target achieved the strongest Hit@1 and MRR@10, improving Hit@1 by 7.34 percentage points over the whole-document MiniLM baseline and increasing MRR@10 by approximately 8.3% relative. A 600-character target achieved the strongest Hit@3 and Hit@5, demonstrating that the preferred chunk size depends on retrieval depth and system objective.

The study also illustrates why retrieval and generation should be evaluated separately. A controlled supported query was refused despite high retrieval similarity, showing that evidence retrieval does not guarantee correct generation behavior.

Future work can extend the experiment to additional embedding models, semantic and structure-aware chunking, document-level score aggregation, reranking, broader BEIR metrics, larger grounding evaluations, and human or entailment-based assessment of citation support.

References

Bennani, S., & Moslonka, C. (2026). A Systematic Analysis of Chunking Strategies for Reliable Question Answering. arXiv:2601.14123.

Kreileder, V. J. J., Reisinger, J., & Fischer, A. (2026). Evaluating Chunking Strategies for Retrieval-Augmented Generation on Academic Texts. arXiv:2607.01852.

Thakur, N., Reimers, N., Rucklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks.

Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or Fiction: Verifying Scientific Claims. EMNLP.

Sentence Transformers. all-MiniLM-L6-v2 model documentation. Hugging Face.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data.

Appendix A. Reproducibility Notes

Core experimental artifacts in the project include the SciFact baseline results CSV, chunk-size results CSV, controlled generation evaluation runner and incremental results CSV, retrieval evaluation utilities, sentence-aware chunker, MiniLM embedder, NumPy and FAISS retrievers, persistent index storage, FastAPI backend, and React frontend.

The generation-question JSON is currently stored under a data path ignored by Git and should be moved or explicitly tracked before the final reproducibility release.

The current generation evaluation can be resumed without rerunning completed questions. The five recorded supported trials should be preserved; incomplete questions should only be added when the external API returns an actual model response.