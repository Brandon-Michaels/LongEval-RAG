# LongEval RAG - Task 4 - CLEF 2026

Repository for **DS@GT ARC’s CLEF 2026 LongEval Task 4** submission:  
**“DS@GT ARC at LongEval: Citation Integrity and Factual Grounding in Scientific QA.”**

This project studies citation grounded Retrieval-Augmented Generation (RAG) for scientific question answering across temporally evolving document corpora. The LongEval-RAG task provides each query with a fixed set of 10 candidate documents from the CORE scientific literature corpus. The system must generate a concise natural-language answer and cite the candidate documents that support the generated response.

Our work focuses on the gap between standard answer quality metrics such as ROUGE/BERTScore and stricter grounding metrics that evaluate whether generated claims are actually supported by the cited documents.

## Task Overview

For each LongEval-RAG query, the input contains:

- a natural-language scientific query,
- a list of 10 pre-retrieved candidate document IDs,
- document fields including title, abstract, and full text.

The system outputs a JSONL entry containing:

- metadata for the query and run,
- the original `references` array of candidate document IDs,
- one generated answer,
- citation indices pointing into the local `references` array.

Citations are local positional indices into the provided `references` array, not global document IDs.

## Implemented Pipelines

This repository contains the automatic DS@GT ARC pipelines used for the paper.

### 1. Hybrid Baseline: BM25 + BGE + Gemma

The baseline is a standard post-retrieval RAG pipeline over the organizer-provided 10-document candidate set.

Pipeline:

1. Split each candidate document into overlapping chunks.
2. Rank chunks with BM25 lexical scoring and BGE dense embeddings.
3. Select the top-ranked evidence chunks.
4. Generate a concise answer using Gemma-4-31B.
5. Assign citation indices based on the evidence supporting the generated answer.

This run corresponds to the paper’s **DS@GT Hybrid Baseline**.

### 2. CRAG + CiteFix

The main research pipeline extends the hybrid baseline with pre-generation evidence filtering and post-generation citation repair.

Pipeline:

1. Split and rank candidate document chunks using BM25 + BGE.
2. Apply a CRAG-style filtering step to remove low-support chunks before generation.
3. Generate an answer from the filtered evidence using Gemma-4-31B.
4. Apply a CiteFix-style post-generation step:
   - split the answer into claims,
   - score claim support against retrieved chunks,
   - prune or repair weakly supported claims/citations,
   - output final citation indices.

This run corresponds to the paper’s **CRAG+CiteFix** system.

## Frontier Model Benchmarks

The paper also reports benchmark runs using **GPT-5.5 Thinking** and **Claude Opus 4.7 Thinking**. These runs were generated separately by prompting the frontier models over the raw text of the 10 provided candidate documents.

They are included in the paper as upper-bound reasoning-model comparisons, but the repository’s automatic notebooks focus on the reproducible baseline and CRAG+CiteFix pipelines.

## Main Results

Both automatic DS@GT ARC systems improved over the LongEval organizer naive baseline on the main official answer similarity and citation precision metrics.

| System | ROUGE-F1 | BERT-F1 | Citation Precision |
|---|---:|---:|---:|
| Organizer Naive Baseline | 0.082 | -0.118 | 0.200 |
| DS@GT Hybrid Baseline | 0.132 | 0.118 | 0.766 |
| CRAG+CiteFix | 0.126 | 0.076 | 0.692 |
| GPT-5.5 Thinking | 0.188 | 0.227 | 0.904 |
| Claude Opus 4.7 Thinking | 0.169 | 0.157 | 1.000 |

The official LongEval metrics showed that frontier-model runs achieved the strongest natural-language similarity and answer relevancy scores. However, our reference-free RAGAs diagnostics showed that these models often produced relevant answers without strictly grounding all claims in the cited documents.

The CRAG+CiteFix pipeline achieved the strongest grounding-oriented diagnostic scores among the evaluated systems:

| System | Global Faithfulness | Citation Faithfulness | Answer Relevancy |
|---|---:|---:|---:|
| Hybrid Baseline | 0.741 | 0.741 | 0.531 |
| CRAG+CiteFix | 0.784 | 0.758 | 0.476 |
| GPT-5.5 Thinking | 0.559 | 0.417 | 0.756 |
| Claude Opus 4.7 Thinking | 0.553 | 0.528 | 0.646 |

Our results highlight that trustworthy scientific RAG systems should be evaluated not only by answer similarity or fluency, but also by whether their generated claims are actually supported by the cited evidence. For continued reading, reference the CLEF 2026 Working Notes.
