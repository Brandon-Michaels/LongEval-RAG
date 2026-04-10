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

We plan to prepare up to 3 runs corresponding to progressively stronger pipelines:

1. **Baseline Similarity QA**
   - BM25 / lexical retrieval
   - E5 dense reranking
   - simple extract selection
   - grounded answer generation

2. **Cross-Encoder / NLI Evidence Selection**
   - candidate retrieval
   - NLI-based reranking of evidence
   - answer generation from factually supported passages

3. **Baseline → NLI → NLI with Inference**
   - baseline retrieval
   - stronger extract filtering
   - multiple extraction strategies (LLM / spaCy / NLI-supported)
   - final grounded answer synthesis

## Repository Structure

```text
longeval-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_baseline_similarity_qa.ipynb
│   ├── 02_cross_encoder_nli.ipynb
│   └── 03_nli_with_inference.ipynb
├── src/
│   ├── nli_reranker.py
│   ├── extract_llm.py
│   ├── extract_spacy.py
│   ├── qa_pipeline.py
│   └── eval.py
├── submissions/
└── data/
