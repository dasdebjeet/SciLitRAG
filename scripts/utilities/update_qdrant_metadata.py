
from qdrant_client import QdrantClient, models

TARGET_ID = "2611.16752v1"
EXPECTED_OLD_TITLE = (
    "AgriSense: Vision-Language Decision Support and Robotic "
    "Intervention in a Virtual Greenhouse Copy title"
)
CORRECT_TITLE = (
    "AgriSense: Vision-Language Decision Support and Robotic "
    "Intervention in a Virtual Greenhouse"
)
EXPECTED_POINTS = 189

client = QdrantClient(url="http://localhost:16333")

try:
    # Select only points belonging to the affected paper.
    paper_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="arxiv_id",
                match=models.MatchValue(value=TARGET_ID),
            )
        ]
    )

    # Retrieve matching point IDs without downloading vectors.
    points, next_offset = client.scroll(
        collection_name="scilit_rag",
        scroll_filter=paper_filter,
        limit=256,
        with_payload=True,
        with_vectors=False,
    )

    if next_offset is not None:
        raise RuntimeError(
            "Unexpected pagination: more than one scroll page is required. "
            "Stop and inspect the results before updating."
        )

    if len(points) != EXPECTED_POINTS:
        raise RuntimeError(
            f"Expected {EXPECTED_POINTS} points, found {len(points)}. "
            "No changes were made."
        )

    # Confirm every point has the exact contaminated title.
    unexpected = [
        point.id
        for point in points
        if point.payload.get("title") != EXPECTED_OLD_TITLE
    ]

    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} points have unexpected titles. "
            "No changes were made."
        )

    point_ids = [point.id for point in points]

    print(f"Validated {len(point_ids)} points.")
    print("Updating title payload only...")

    # set_payload modifies only the specified payload field.
    # Existing vectors and all other payload fields remain untouched.
    client.set_payload(
        collection_name="scilit_rag",
        payload={"title": CORRECT_TITLE},
        points=point_ids,
        wait=True,
    )

    # Verify all matching points after the update.
    updated_points, _ = client.scroll(
        collection_name="scilit_rag",
        scroll_filter=paper_filter,
        limit=256,
        with_payload=True,
        with_vectors=False,
    )

    incorrect = [
        point.id
        for point in updated_points
        if point.payload.get("title") != CORRECT_TITLE
    ]

    if incorrect or len(updated_points) != EXPECTED_POINTS:
        raise RuntimeError(
            f"Verification failed: {len(incorrect)} incorrect titles "
            f"among {len(updated_points)} points."
        )

    print("\nSUCCESS: Qdrant metadata updated and verified.")
    print(f"Paper: {TARGET_ID}")
    print(f"Updated points: {len(updated_points)}")
    print(f"Correct title: {CORRECT_TITLE}")

finally:
    client.close()
