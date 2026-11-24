# SciLitRAG — Evaluation and Benchmark Results

## 1. Introduction

This document presents the empirical evaluation of **SciLitRAG's** retrieval, context-selection, and answer-generation pipeline. The assessment focuses on three core dimensions:

1. **Retrieval Performance:** Whether the system identifies the correct source papers and ranks relevant passages effectively.
2. **Context Efficiency:** Whether adaptive context selection reduces token consumption compared with a fixed top-$k$ policy.
3. **Answer Quality:** Whether generated responses are coherent, useful, and grounded in retrieved evidence.

> *Note: These experiments serve as an initial engineering benchmark of the pipeline rather than a large-scale statistical validation across scientific domains.*

---

## 2. Evaluation Overview

The evaluation is conducted on a manually verified benchmark of **15 research questions** designed to isolate retrieval accuracy, passage ranking, context efficiency, and response quality.

### Benchmark Composition

| Characteristic | Value |
| :--- | :---: |
| **Total Research Questions** | `15` |
| **Answerable Questions** | `13` |
| **Unanswerable Questions** | `2` |

### Evaluation Matrix

| Pipeline Stage | Comparison / Assessment Method |
| :--- | :--- |
| **Paper Retrieval** | Hybrid retrieval (`BM25` + `BAAI/bge-m3` via RRF) |
| **Passage Retrieval** | Hybrid retrieval vs. Cross-encoder reranking (`BAAI/bge-reranker-v2-m3`) |
| **Context Selection** | Adaptive context selection vs. Fixed `top-10` selection |
| **Answer Generation** | Manual qualitative audit of generated responses |

---

## 3. Retrieval Evaluation

### 3.1 Objective & Configurations

The retrieval evaluation compares two pipeline configurations across both **paper-level** and **chunk-level** granularities:

* **Hybrid Retrieval:** Candidate passages produced by combining sparse (`BM25`) and dense (`BAAI/bge-m3`) retrieval via Reciprocal Rank Fusion (RRF).
* **After Reranking:** Hybrid candidates re-scored using the `BAAI/bge-reranker-v2-m3` cross-encoder.

### 3.2 Metrics

| Metric | Definition |
| :--- | :--- |
| **Recall@$k$** | Proportion of queries where at least one relevant item appears within the top-$k$ results. |
| **MRR@$k$** | Mean Reciprocal Rank of the first relevant result within the top-$k$ results. |
| **nDCG@$k$** | Normalized Discounted Cumulative Gain, accounting for both relevance and rank position. |

* **Paper-level metrics** verify whether the correct source document is surfaced.
* **Chunk-level metrics** verify whether the most pertinent passages within that document are prioritized.

### 3.3 Quantitative Results

| Metric | Hybrid Retrieval (RRF) | After Cross-Encoder Reranking | Change |
| :--- | :---: | :---: | :---: |
| **Paper Recall@5** | `1.0000` | `1.0000` | — |
| **Paper Recall@10** | `1.0000` | `1.0000` | — |
| **Paper MRR@10** | `1.0000` | `1.0000` | — |
| **Paper nDCG@5** | `1.0000` | `1.0000` | — |
| **Chunk Recall@10** | `0.7086` | `0.7560` | `+0.0474` |
| **Chunk MRR@10** | `0.6564` | `0.7564` | `+0.1000` |
| **Chunk nDCG@5** | `0.5510` | `0.5771` | `+0.0261` |

### 3.4 Analysis & Key Findings

* **Document-Level Consistency:** Paper-level metrics reached `1.0000` across all evaluated cutoffs for both configurations, indicating that hybrid retrieval reliably surfaces the target paper.
* **Passage-Level Gains:** Cross-encoder reranking improved every chunk-level metric:
  * **Chunk Recall@10:** `+0.0474` (`0.7086` $\rightarrow$ `0.7560`)
  * **Chunk MRR@10:** `+0.1000` (`0.6564` $\rightarrow$ `0.7564`)
  * **Chunk nDCG@5:** `+0.0261` (`0.5510` $\rightarrow$ `0.5771`)
* **Significance of MRR Improvement:** The `+0.1000` gain in `Chunk MRR@10` demonstrates that cross-encoder reranking substantially elevates the highest-value passage toward the top of the ranking.

> **Key Takeaway:** Identifying the right research paper does not guarantee optimal passage ordering. Two-stage reranking bridges this gap by improving passage-level precision without altering document-level recall.

---

## 4. Context Selection Evaluation

### 4.1 Objective & Policies

This experiment evaluates whether dynamic passage selection reduces prompt token overhead compared with a static passage budget:

* **Adaptive Selection:** Dynamically selects a variable number of passages based on relevance thresholds/selection criteria.
* **Fixed Top-10:** Always supplies the top `10` ranked passages regardless of query complexity.

### 4.2 Quantitative Results

| Metric | Adaptive Selection | Fixed Top-10 | Relative Change |
| :--- | :---: | :---: | :---: |
| **Average Context Tokens** | `3,502` | `4,711` | **-25.7%** |
| **Average Selected Passages** | `7.54` | `10.00` | **-24.6%** |

### 4.3 Token Reduction Calculation

Adaptive selection reduced average context size from $4,711$ tokens to $3,502$ tokens:

$$
\text{Reduction} = \frac{4711 - 3502}{4711} \times 100 \approx 25.7\%
$$

### 4.4 Analysis & Caveats

* **Efficiency Benefits:** Reducing context size by **25.7%** (averaging `7.54` passages instead of `10`) trims redundant evidence, lowers inference latency/compute overhead, and preserves context budget for multi-turn conversation history.
* **Scope of Claim:** Token reduction demonstrates **context efficiency**, not universal superiority. Compact contexts are beneficial only insofar as essential supporting evidence is retained.

---

## 5. Answer Quality Evaluation

### 5.1 Methodology

Because retrieval metrics cannot measure synthesis faithfulness or clarity, a manual qualitative review was conducted across **8 generated responses**:

* **4** benchmark research questions
* **2** context-selection policies (`Adaptive Selection` vs. `Fixed Top-10`)

### 5.2 Observations & Limitations

* **Qualitative Findings:** Reviewed responses were coherent, well-structured, and aligned with the retrieved source passages, providing initial confirmation that the end-to-end RAG workflow functions as intended.
* **Sample Size Constraints:** An 8-response audit is too small to establish statistically robust claims regarding factual accuracy, citation precision, or hallucination resistance across diverse scientific literature.

---

## 6. Summary of Findings & Limitations

### Core Takeaways

1. **Paper Retrieval:** Achieved `1.0000` across evaluated top-$k$ cutoffs for both hybrid and reranked pipelines.
2. **Passage Ranking:** Cross-encoder reranking improved chunk-level `Recall@10`, `MRR@10`, and `nDCG@5`.
3. **Context Efficiency:** Adaptive selection reduced context token usage by **~25.7%** while maintaining qualitatively sound responses in manual checks.

### Current Limitations

| Dimension | Limitation |
| :--- | :--- |
| **Benchmark Scale** | Limited to `15` manually verified questions (`13` answerable, `2` unanswerable). |
| **Paper-Level Ceiling** | Perfect paper-level scores on a small test set do not guarantee generalization to larger corpora. |
| **Corpus Specificity** | Chunk-level metrics reflect the current indexed paper collection and question distribution. |
| **Context vs. Quality** | Lower token counts prove efficiency, not necessarily higher answer accuracy across all query types. |
| **Qualitative Audit Scope** | Manual review covers `8` outputs and lacks formal inter-annotator scoring. |
| **Domain Generalization** | Cross-disciplinary performance across varied document layouts and scientific fields remains untested. |

---

## 7. Future Evaluation Roadmap

To build on this baseline, future evaluation work should address five priorities:

1. **Expand Benchmark Coverage:** Scale the dataset with diverse question types (e.g., multi-hop synthesis, methodology comparison, quantitative lookup) across multiple scientific disciplines.
2. **Ablate Retrieval Components:** Benchmark `BM25`-only and dense-only baselines alongside hybrid (`RRF`) and reranked configurations to quantify each stage's individual contribution.
3. **Formalize Answer-Quality Rubrics:** Introduce structured evaluation criteria—and where feasible, domain-expert review—scoring **factual correctness**, **evidence faithfulness**, **citation precision**, and **unsupported claims**.
4. **Joint Efficiency–Quality Tradeoff Analysis:** Evaluate adaptive vs. fixed top-$k$ policies simultaneously on token cost and answer accuracy to verify that pruning passages does not drop critical edge-case evidence.
5. **Stress-Test Unanswerable Queries:** Expand the unanswerable subset to rigorously measure abstention behavior and ensure the model refuses to speculate when corpus evidence is absent.

---

## 8. Conclusion

The initial evaluation of **SciLitRAG** demonstrates strong retrieval and context-efficiency characteristics. Hybrid retrieval consistently locates relevant source papers, cross-encoder reranking meaningfully improves passage-level ranking (`+0.1000` `Chunk MRR@10`), and adaptive context selection reduces prompt token consumption by **25.7%**. While manual response checks are encouraging, expanding benchmark scale and formalizing faithfulness metrics remain the primary next steps for rigorous validation.