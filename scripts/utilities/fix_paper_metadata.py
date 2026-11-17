
"""
Correct known metadata contamination for a paper.

Creates backups before modifying JSON/JSONL files.
Only the exact old title is replaced; all other fields and records
are preserved.
"""

import json
import shutil
from pathlib import Path

TARGET_ID = "2611.16752v1"

OLD_TITLE = (
    "AgriSense: Vision-Language Decision Support and Robotic "
    "Intervention in a Virtual Greenhouse Copy title"
)

NEW_TITLE = (
    "AgriSense: Vision-Language Decision Support and Robotic "
    "Intervention in a Virtual Greenhouse"
)

FILES = [
    Path("data/metadata/papers.json"),
    Path("data/processed/pages.jsonl"),
    Path("data/processed/chunks.jsonl"),
]


def correct_json_file(path: Path) -> int:
    """Correct the target paper's title in a JSON document."""
    data = json.loads(path.read_text(encoding="utf-8"))
    replacements = 0

    def update(obj):
        nonlocal replacements

        if isinstance(obj, dict):
            if (
                obj.get("arxiv_id") == TARGET_ID
                and obj.get("title") == OLD_TITLE
            ):
                obj["title"] = NEW_TITLE
                replacements += 1

            for value in obj.values():
                update(value)

        elif isinstance(obj, list):
            for value in obj:
                update(value)

    update(data)

    if replacements:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return replacements


def correct_jsonl_file(path: Path) -> int:
    """Correct both record titles and nested PDF metadata titles."""
    replacements = 0
    corrected_lines = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if record.get("arxiv_id") == TARGET_ID:

                # Correct the main record title if it is still contaminated.
                if record.get("title") == OLD_TITLE:
                    record["title"] = NEW_TITLE
                    replacements += 1

                # PDF metadata is nested separately from the main title.
                pdf_metadata = record.get("pdf_metadata")

                if (
                    isinstance(pdf_metadata, dict)
                    and pdf_metadata.get("title") == OLD_TITLE
                ):
                    pdf_metadata["title"] = NEW_TITLE
                    replacements += 1

            corrected_lines.append(
                json.dumps(record, ensure_ascii=False)
            )

    if replacements:
        path.write_text(
            "\n".join(corrected_lines) + "\n",
            encoding="utf-8",
        )

    return replacements


def main():
    """Back up and correct all known affected files."""
    total_replacements = 0

    for path in FILES:
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

        # Preserve the first backup instead of overwriting it on reruns.
        backup_path = path.with_suffix(path.suffix + ".bak")

        if not backup_path.exists():
            shutil.copy2(path, backup_path)

        if path.suffix == ".jsonl":
            replacements = correct_jsonl_file(path)
        else:
            replacements = correct_json_file(path)

        total_replacements += replacements

        print(
            f"{path}: corrected {replacements} record(s); "
            f"backup: {backup_path}"
        )

    print(f"\nTotal corrected records: {total_replacements}")


if __name__ == "__main__":
    main()
