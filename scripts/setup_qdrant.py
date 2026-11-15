
"""
Initialize and verify the SciLitRAG Qdrant collection.

Responsibilities:
    - Connect to the local Qdrant server.
    - Create the scilit_rag collection if it does not exist.
    - Configure a named dense vector using BGE-M3 embeddings.
    - Verify the existing collection configuration without
      deleting or recreating existing data.

Assumptions:
    - Qdrant is accessible at http://localhost:6333.
    - BGE-M3 embeddings have 1,024 dimensions.
    - Cosine similarity is used for dense retrieval.
"""

import logging
import sys

from qdrant_client import QdrantClient, models


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "scilit_rag"

VECTOR_NAME = "dense"
VECTOR_SIZE = 1024
DISTANCE_METRIC = models.Distance.COSINE


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Collection initialization
# ---------------------------------------------------------------------

def main() -> int:
    """Create or verify the SciLitRAG Qdrant collection."""

    client = None

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=10)

        logger.info("Connected to Qdrant at %s", QDRANT_URL)

        if not client.collection_exists(COLLECTION_NAME):
            logger.info("Creating collection '%s'...", COLLECTION_NAME)

            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    VECTOR_NAME: models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=DISTANCE_METRIC,
                    )
                },
            )

            logger.info("Collection created successfully.")
        else:
            logger.info("Collection already exists. Verifying configuration.")

        collection = client.get_collection(COLLECTION_NAME)
        vector_config = collection.config.params.vectors

        if not isinstance(vector_config, dict):
            raise ValueError("Expected named vectors, but found unnamed vectors.")

        if VECTOR_NAME not in vector_config:
            raise ValueError(f"Missing named vector: {VECTOR_NAME}")

        actual_config = vector_config[VECTOR_NAME]

        if actual_config.size != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE} dimensions; found {actual_config.size}."
            )

        if actual_config.distance != DISTANCE_METRIC:
            raise ValueError(
                f"Expected {DISTANCE_METRIC}; found {actual_config.distance}."
            )

        logger.info("=" * 60)
        logger.info("QDRANT COLLECTION VERIFIED")
        logger.info("Collection: %s", COLLECTION_NAME)
        logger.info("Vector: %s", VECTOR_NAME)
        logger.info("Dimensions: %d", actual_config.size)
        logger.info("Distance: %s", actual_config.distance)
        logger.info("Existing points: %d", collection.points_count or 0)
        logger.info("=" * 60)

        return 0

    except Exception:
        logger.exception("Qdrant collection initialization failed.")
        return 1

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
