
"""
Integration test for the SciLitRAG context assembly stage.

Runs the existing hybrid retrieval and reranking components, then
verifies that ContextBuilder produces correctly labelled evidence
with the corresponding paper and page metadata.
"""

import logging

from scilitrag.pipeline.context_builder import ContextBuilder
from scilitrag.retrieval.hybrid_retriever import HybridRetriever
from scilitrag.retrieval.reranker import CrossEncoderReranker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    query = "How do machine learning methods quantify pore-space evolution over time?"

    hybrid = None

    try:
        # Stage 1: Retrieve a broad set of candidates using BM25 + dense search.
        LOGGER.info("Initializing hybrid retriever...")
        hybrid = HybridRetriever()

        candidates = hybrid.search(
            query=query,
            top_k=20,
            retrieval_k=20,
        )

        LOGGER.info("Hybrid retrieval returned %d candidates.", len(candidates))

        if not candidates:
            LOGGER.warning("No candidates retrieved. Cannot test context assembly.")
            return

        # Stage 2: Rerank candidates using the cross-encoder.
        LOGGER.info("Initializing cross-encoder reranker...")
        reranker = CrossEncoderReranker()

        ranked_results = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=5,
        )

        LOGGER.info("Reranking returned %d results.", len(ranked_results))

        # Stage 3: Assemble the final evidence context.
        # The tokenizer is omitted here, so ContextBuilder uses its
        # approximate token counter. We will integrate the Qwen tokenizer later.
        builder = ContextBuilder(
            max_tokens=6000,
            max_chunks_per_paper=3,
        )

        assembled = builder.build(ranked_results)

        # Stage 4: Inspect the assembled evidence and its metadata.
        print("\n" + "=" * 75)
        print("CONTEXT ASSEMBLY SUMMARY")
        print("=" * 75)
        print(f"Retrieved candidates : {len(candidates)}")
        print(f"Reranked candidates  : {len(ranked_results)}")
        print(f"Selected evidence    : {len(assembled.evidence)}")
        print(f"Estimated tokens     : {assembled.token_count}")

        print("\n" + "=" * 75)
        print("EVIDENCE METADATA")
        print("=" * 75)

        for item in assembled.evidence:
            print(
                f"\n[{item.evidence_id}] {item.title}\n"
                f"    arXiv ID       : {item.arxiv_id}\n"
                f"    Page           : {item.page_number}\n"
                f"    Chunk ID       : {item.chunk_id}\n"
                f"    Reranker score : {item.reranker_score:.4f}"
            )

        print("\n" + "=" * 75)
        print("ASSEMBLED CONTEXT (first 3000 characters)")
        print("=" * 75)
        print(assembled.context[:3000])

        print("\n" + "=" * 75)
        print("CONTEXT BUILDER TEST COMPLETED")
        print("=" * 75)

    finally:
        # Release the dense retrieval model and its associated resources.
        if hybrid is not None:
            hybrid.close()


if __name__ == "__main__":
    main()
