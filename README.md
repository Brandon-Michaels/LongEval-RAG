# LongEval-RAG CLEF 2026

Repository for our CLEF LongEval 2026 submission for **Task 4: LongEval-RAG (Retrieval Augmented Generation)**.

This project studies how well Retrieval-Augmented Generation (RAG) systems cope with the **evolution of scientific knowledge over time**. Given a natural-language query and a provided set of candidate document IDs from the LongEval 2026 scientific corpus, the goal is to:

1. generate a grounded textual answer, and  
2. identify the most relevant document extracts supporting that answer.

Our system is built around a staged pipeline using:
- sparse retrieval and similarity baselines,
- dense reranking with E5,
- cross-encoder / NLI-based evidence filtering,
- extractive evidence selection,
- grounded answer generation from selected extracts only.

## Task Overview

For each query, the task provides:
- a **query ID** and **textual query**
- a set of **candidate document IDs** from the corpus

The system must output:

### Part A: Answer Generation
A textual answer for each query:
(query_id, answer)

### Part B: Relevant Document Extracts
A set of up to 5 supporting extracts:
(query_id, {document extract})

The generated answer must be based solely on the relevant extracts selected from the provided candidate documents.

## Planned Runs

We plan to prepare up to 3-5 runs corresponding to progressively stronger pipelines:

1. **Baseline Similarity QA**
   - BM25 / lexical retrieval
   - BGE dense reranking
   - simple extract selection
   - grounded answer generation

2. **ColBERT + CiteFix**
   - candidate retrieval
   - ColBERT dense embeddings
   - CiteFix grounded citations for claims

3. **CRAG / CiteFix**
   - BM25 / lexical retrieval
   - BGE dense reranking
   - CRAG grounded answer generation
   - CiteFix grounded citations for claims
  
4. **Manual ChatGPT 5.5 Thinking**
   - Answers generated manually with ChatGPT 5.5 Thinking
