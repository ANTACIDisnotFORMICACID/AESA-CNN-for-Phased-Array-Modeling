"""Offline parser for the subset of AEDT text needed by the V2.8 build.

The parser does not launch AEDT and never modifies the input project. It is a
preflight aid only; final geometry and boundary truth must still come from AEDT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


BEGIN_RE = re.compile(r"^\s*\$begin '([^']+)'\s*$")
END_RE = re.compile(r"^\s*\$end '([^']+)'\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else None


def find_named_block(lines: list[str], name: str, start: int = 0) -> tuple[int, int]:
    begin_index = None
    depth = 0
    for index in range(start, len(lines)):
        begin = BEGIN_RE.match(lines[index])
        end = END_RE.match(lines[index])
        if begin:
            if begin_index is None and begin.group(1) == name:
                begin_index = index
                depth = 1
                continue
            if begin_index is not None:
                depth += 1
        elif end and begin_index is not None:
            depth -= 1
            if depth == 0:
                return begin_index, index
    raise ValueError("AEDT block {!r} was not closed".format(name))


def direct_child_blocks(lines: list[str], parent_name: str, child_name: str) -> Iterable[list[str]]:
    parent_start, parent_end = find_named_block(lines, parent_name)
    index = parent_start + 1
    while index < parent_end:
        begin = BEGIN_RE.match(lines[index])
        if begin and begin.group(1) == child_name:
            child_start, child_end = find_named_block(lines, child_name, index)
            yield lines[child_start : child_end + 1]
            index = child_end + 1
        else:
            index += 1


def parse_geometry(lines: list[str]) -> list[dict[str, object]]:
    records = []
    for block in direct_child_blocks(lines, "ToplevelParts", "GeometryPart"):
        text = "\n".join(block)
        body_ids = [int(item) for item in re.findall(r"\bBodyID=(\d+)", text)]
        operation_types = re.findall(r"OperationType='([^']+)'", text)
        record = {
            "name": first_match(text, r"^\s*Name='([^']+)'"),
            "flags": first_match(text, r"^\s*Flags='([^']*)'"),
            "group_id": first_match(text, r"^\s*GroupId=(-?\d+)"),
            "material": first_match(text, r"^\s*MaterialValue='([^']*)'"),
            "body_ids": body_ids,
            "final_body_id": body_ids[-1] if body_ids else None,
            "operation_types": operation_types,
            "x_position": first_match(text, r"^\s*XPosition='([^']+)'"),
            "y_position": first_match(text, r"^\s*YPosition='([^']+)'"),
            "z_position": first_match(text, r"^\s*ZPosition='([^']+)'"),
            "x_size": first_match(text, r"^\s*XSize='([^']+)'"),
            "y_size": first_match(text, r"^\s*YSize='([^']+)'"),
            "z_size": first_match(text, r"^\s*ZSize='([^']+)'"),
        }
        records.append(record)
    return records


def parse_boundaries(lines: list[str]) -> list[dict[str, object]]:
    records = []
    boundaries_start, boundaries_end = find_named_block(lines, "Boundaries")
    index = boundaries_start + 1
    while index < boundaries_end:
        begin = BEGIN_RE.match(lines[index])
        if begin:
            name = begin.group(1)
            block_start, block_end = find_named_block(lines, name, index)
            text = "\n".join(lines[block_start : block_end + 1])
            boundary_type = first_match(text, r"^\s*BoundType='([^']+)'" )
            if boundary_type:
                objects_match = re.search(r"^\s*Objects\(([^)]*)\)", text, flags=re.MULTILINE)
                object_ids = []
                if objects_match and objects_match.group(1).strip():
                    object_ids = [int(item.strip()) for item in objects_match.group(1).split(",")]
                records.append(
                    {
                        "name": name,
                        "type": boundary_type,
                        "object_ids": object_ids,
                    }
                )
            index = block_end + 1
        else:
            index += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    lines = project.read_text(encoding="utf-8", errors="replace").splitlines()
    geometry = parse_geometry(lines)
    body_to_name = {
        int(record["final_body_id"]): str(record["name"])
        for record in geometry
        if record["final_body_id"] is not None
    }
    boundaries = parse_boundaries(lines)
    for boundary in boundaries:
        boundary["object_names"] = [
            body_to_name.get(int(object_id), "<unknown:{}>".format(object_id))
            for object_id in boundary["object_ids"]
        ]
    report = {
        "schema_version": 1,
        "source": "offline_aedt_text_parser",
        "project": str(project),
        "project_sha256": sha256_file(project),
        "geometry_count": len(geometry),
        "geometry": geometry,
        "boundaries": boundaries,
        "warning": "Preflight only; AEDT GUI/API remains authoritative.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
