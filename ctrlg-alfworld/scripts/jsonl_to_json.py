#!/usr/bin/env python3
"""Convert JSON Lines to a JSON array, omitting token-ID fields."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def is_token_id_field(key: str) -> bool:
    """Return whether a field name represents token IDs."""
    normalized = key.casefold()
    return normalized == "token_ids" or normalized.endswith("_token_ids")


def remove_token_id_fields(value: Any) -> tuple[Any, int]:
    """Recursively remove token-ID fields and return their count."""
    if isinstance(value, dict):
        cleaned = {}
        removed = 0
        for key, item in value.items():
            if is_token_id_field(key):
                removed += 1
                continue
            cleaned_item, nested_removed = remove_token_id_fields(item)
            cleaned[key] = cleaned_item
            removed += nested_removed
        return cleaned, removed

    if isinstance(value, list):
        cleaned_items = []
        removed = 0
        for item in value:
            cleaned_item, nested_removed = remove_token_id_fields(item)
            cleaned_items.append(cleaned_item)
            removed += nested_removed
        return cleaned_items, removed

    return value, 0


def convert_jsonl(source: Path, destination: Path, *, indent: int | None = None) -> tuple[int, int]:
    """Stream JSONL records into a JSON array and return records/fields removed."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("input and output paths must be different")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    records_written = 0
    fields_removed = 0

    try:
        with source.open("r", encoding="utf-8-sig") as input_file, tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            output_file.write("[\n")

            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{source}:{line_number}: invalid JSON: {error.msg}"
                    ) from error

                cleaned, removed = remove_token_id_fields(record)
                if records_written:
                    output_file.write(",\n")
                json.dump(
                    cleaned,
                    output_file,
                    ensure_ascii=False,
                    indent=indent,
                    separators=(",", ":") if indent is None else None,
                )
                records_written += 1
                fields_removed += removed

            output_file.write("\n]\n")

        os.replace(temporary_path, destination)
        return records_written, fields_removed
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file to a JSON array and remove every token-ID field."
    )
    parser.add_argument("input", type=Path, help="input .jsonl file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: input path with a .json extension)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="pretty-print with this indentation level (compact by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.output or args.input.with_suffix(".json")
    records, removed = convert_jsonl(args.input, destination, indent=args.indent)
    print(
        f"Wrote {records} records to {destination} "
        f"and removed {removed} token-ID fields."
    )


if __name__ == "__main__":
    main()
