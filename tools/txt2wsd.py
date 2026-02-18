from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class Instance:
    """
    One instance derived from one line in the data file.

    Fields
    - id: a stable identifier for this instance, derived from line index
    - template_id: fixed label describing how this instance should be interpreted downstream
    - nodes: the raw node sequence as written in the data line
    - edges: PlantUML inheritance edges derived from nodes

    Edge convention
    - In the data line, the token on the right is the child of the token on the left.
    - For PlantUML generalization, we emit: child <|-- parent
    """
    id: str
    template_id: str
    nodes: List[str]
    edges: List[str]

def read_data_lines(data_path: Path) -> List[str]:
    """
    Read the raw data file and return a list of non-empty lines.

    What it does
    - Reads the file as UTF eight text
    - Strips whitespace
    - Drops empty lines
    """
    lines: List[str] = []
    for raw in data_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
    return lines

def tokenize_line(line: str) -> List[str]:
    """
    Split one data line into tokens.

    Token rule
    - Split by any whitespace via Python's built in split
    """
    return line.split()

def build_edges_from_nodes(nodes: List[str], template_id: str) -> List[str]:
    """
    Convert a node chain into inheritance edges.

    Input semantics
    - nodes are ordered from left to right
    - the node on the right is the parent of the node on the left
      example: Resume Document File means
      Resume extends Document, Document extends File

    Output semantics
    - each adjacent pair yields one PlantUML inheritance edge:
    """
    edges: List[str] = []
    if len(nodes) < 2:
        return edges

    if template_id == "inheritance":
        for child, parent in zip(nodes[:-1], nodes[1:]):
            edges.append(f"{child} <|-- {parent}")
        return edges
    elif template_id == "composition":
        for child, parent in zip(nodes[:-1], nodes[1:]):
            edges.append(f"{parent} *-- {child}")
        return edges
    elif template_id == "aggregation":
        for child, parent in zip(nodes[:-1], nodes[1:]):
            edges.append(f"{parent} o-- {child}")
        return edges
    elif template_id == "dependency":
        for child, parent in zip(nodes[:-1], nodes[1:]):
            edges.append(f"{parent} ..> {child}")
        return edges
    # elif template_id == "association":
    #     for child, parent in zip(nodes[:-1], nodes[1:]):
    #         edges.append(f"{child} --> {parent}")
        # return edges

def parse_instances(lines: List[str], template_id: str) -> List[Instance]:
    """
    Turn raw lines into structured instances.

    What it does
    - Each line becomes one Instance
    - id is generated as line index based name
    - nodes are tokens split from the line
    - edges are derived using build_edges_from_nodes
    """
    instances: List[Instance] = []
    for idx, line in enumerate(lines, start=1):
        nodes = tokenize_line(line)
        edges = build_edges_from_nodes(nodes, template_id)
        instances.append(
            Instance(
                id=f"{idx}",
                template_id=template_id,
                nodes=nodes,
                edges=edges,
            )
        )
    return instances

def reverse_instances(instances: List[Instance]) -> List[Instance]:
    """
    Build reversed instances from forward instances.

    Rule
    - Keep id and template_id unchanged.
    - Reverse node order, then rebuild edges with the same template logic.
    """
    reversed_list: List[Instance] = []
    for inst in instances:
        rev_nodes = list(reversed(inst.nodes))
        rev_edges = build_edges_from_nodes(rev_nodes, inst.template_id)
        reversed_list.append(
            Instance(
                id=inst.id,
                template_id=inst.template_id,
                nodes=rev_nodes,
                edges=rev_edges,
            )
        )
    return reversed_list


def write_jsonl(instances: List[Instance], out_path: Path) -> None:
    """
    Write instances into a jsonl file.

    Format
    - One JSON object per line
    - UTF eight encoding
    - ensure_ascii is false to preserve non ASCII characters if any appear
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for inst in instances:
            obj: Dict[str, Any] = {
                "id": inst.id,
                "template_id": inst.template_id,
                "nodes": inst.nodes,
                "edges": inst.edges,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_filename(name: str) -> str:
    """
    Make a filesystem-friendly filename while keeping it readable.

    What it does
    - Strips surrounding spaces
    - Replaces path separators and spaces with underscore

    It does not use regex.
    """
    s = name.strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = s.replace(" ", "_").replace("\t", "_")
    return s or "unnamed"


def pick_name_node(nodes: List[str], template_id: str, direction: str) -> str:
    """
    Pick which node text should be used in output filename.

    Direction rules
    - forward:
      aggregation/composition -> first node
      inheritance/dependency -> last node
    - reverse:
      aggregation/composition -> last node
      inheritance/dependency -> first node
    """
    if not nodes:
        return "unnamed"

    if direction not in {"forward", "reverse"}:
        raise ValueError(f"Unsupported direction: {direction}")

    if template_id in {"aggregation", "composition"}:
        index = 0 if direction == "forward" else -1
    elif template_id in {"inheritance", "dependency"}:
        index = -1 if direction == "forward" else 0
    else:
        index = 0

    return safe_filename(nodes[index])


def safe_unlink(path: Path) -> bool:
    """
    Best-effort delete for files that may be read-only on Windows.
    """
    try:
        path.unlink()
    except PermissionError:
        try:
            os.chmod(path, 0o666)
            path.unlink()
        except OSError as exc:
            return False
    except FileNotFoundError:
        return True
    return True


def clear_wsd_files(out_dir: Path) -> None:
    """
    Remove existing .wsd files under out_dir before regeneration.
    """
    if not out_dir.exists():
        return
    failed = 0
    for stale in out_dir.glob("*.wsd"):
        if not safe_unlink(stale):
            failed += 1
    if failed:
        print(f"[WARN] failed to remove {failed} stale .wsd files under {out_dir}")


def write_wsd_files(
    instances: List[Instance],
    out_dir: Path,
    overwrite: bool,
    direction: str = "forward",
) -> None:
    """
    Generate one .wsd file per instance.

    Naming rule
    - Use pick_name_node based on template_id and direction.

    Content rule
    - File content is still:
      @startuml
      ...
      @enduml
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        clear_wsd_files(out_dir)

    for i, inst in enumerate(instances):
        if not inst.nodes:
            continue

        root_name = pick_name_node(inst.nodes, inst.template_id, direction=direction)
        out_path = out_dir / f"{i+1}_{root_name}.wsd"

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {out_path}")

        body = "\n".join(inst.edges)
        content = "@startuml\n" + body + "\n@enduml\n"
        out_path.write_text(content, encoding="utf-8")


def write_reverse_wsd_files(
    forward_instances: List[Instance],
    out_dir: Path,
    overwrite: bool,
    start_index: int = 1,
) -> None:
    """
    Generate reversed-direction .wsd files while keeping the exact same filenames
    as forward outputs.

    Reverse rule
    - Reverse node order first, then reuse the same edge-building logic.
    - Filename node follows reverse naming rules.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, inst in enumerate(forward_instances):
        if not inst.nodes:
            continue

        reverse_nodes = list(reversed(inst.nodes))
        reverse_edges = build_edges_from_nodes(reverse_nodes, inst.template_id)

        root_name = pick_name_node(reverse_nodes, inst.template_id, direction="reverse")
        out_path = out_dir / f"{start_index + i}_{root_name}.wsd"

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {out_path}")

        body = "\n".join(reverse_edges)
        content = "@startuml\n" + body + "\n@enduml\n"
        out_path.write_text(content, encoding="utf-8")


def run_pipeline(
    data_path: Path,
    jsonl_path: Path,
    template_id: str,
    wsd_dir: Optional[Path],
    overwrite_wsd: bool,
) -> List[Instance]:
    """
    Pipeline entry for one data.txt.

    Stages
    - Read plain text lines from data.txt
    - Parse each line into an Instance
    - Write instances.jsonl
    - Optionally write .wsd files
    """
    lines = read_data_lines(data_path)
    instances = parse_instances(lines, template_id=template_id)
    write_jsonl(instances, jsonl_path)

    if wsd_dir is not None:
        write_wsd_files(instances, wsd_dir, overwrite=overwrite_wsd, direction="forward")
    return instances


def write_jsonl_rows(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_all_data_txt(root_dir: Path) -> List[Path]:
    """
    Traverse root_dir recursively and return all data.txt paths.

    Rule
    - Any file named exactly 'data.txt' is considered an input table.
    """
    results: List[Path] = []
    for p in root_dir.rglob("data.txt"):
        if p.is_file():
            results.append(p)
    results.sort()
    return results


def remove_empty_parents(start_dir: Path, stop_dir: Path) -> None:
    """
    Remove empty folders upwards until stop_dir (exclusive).
    """
    cur = start_dir
    while cur != stop_dir and cur.exists():
        try:
            next(cur.iterdir())
            break
        except StopIteration:
            cur.rmdir()
            cur = cur.parent


def prune_mirror_generated_layout(
    mirror_root: Path,
    expected_rel_dirs: set[Path],
    wsd_subdir_name: str,
    jsonl_name: str,
) -> None:
    """
    Keep mirror generated layout aligned with source data.txt folders.

    What gets pruned
    - all mirror data.txt files (mirror side should only keep generated outputs)
    - mirror/<subset>/instances.jsonl when <subset> is no longer present in source
    - mirror/<subset>/out_wsd when <subset> is no longer present in source
    """
    for data_file in sorted(p for p in mirror_root.rglob("data.txt") if p.is_file()):
        safe_unlink(data_file)
        remove_empty_parents(data_file.parent, mirror_root)

    for jsonl_file in sorted(p for p in mirror_root.rglob(jsonl_name) if p.is_file()):
        if jsonl_file == mirror_root / jsonl_name:
            continue
        rel_parent = jsonl_file.parent.relative_to(mirror_root)
        if rel_parent not in expected_rel_dirs:
            safe_unlink(jsonl_file)
            remove_empty_parents(jsonl_file.parent, mirror_root)

    for wsd_dir in sorted(p for p in mirror_root.rglob(wsd_subdir_name) if p.is_dir()):
        rel_parent = wsd_dir.parent.relative_to(mirror_root)
        if rel_parent not in expected_rel_dirs:
            shutil.rmtree(wsd_dir, ignore_errors=True)
            remove_empty_parents(wsd_dir.parent, mirror_root)


def run_for_root(
    root_dir: Path,
    template_id: str,
    overwrite_wsd: bool,
    wsd_subdir_name: str,
    jsonl_name: str,
    reverse_root: Optional[Path] = None,
    reverse_layout: str = "mirror",
) -> None:
    """
    Run the pipeline for every data.txt found under root_dir.

    Output placement rule
    - For each folder that contains data.txt:
      instances.jsonl is written into the same folder
      .wsd files are written into a subfolder under the same folder
    """
    data_files = find_all_data_txt(root_dir)
    merged_rows: List[Dict[str, Any]] = []
    global_id = 1
    reverse_merged_rows: List[Dict[str, Any]] = []
    reverse_global_id = 1

    reverse_dataset_root: Optional[Path] = None
    reverse_flat_wsd_dir: Optional[Path] = None
    if reverse_root is not None:
        reverse_dataset_root = reverse_root / root_dir.name
        reverse_dataset_root.mkdir(parents=True, exist_ok=True)

        if reverse_layout == "flat":
            reverse_flat_wsd_dir = reverse_dataset_root / wsd_subdir_name
            reverse_flat_wsd_dir.mkdir(parents=True, exist_ok=True)
        elif reverse_layout != "mirror":
            raise ValueError(f"Unsupported reverse_layout: {reverse_layout}")

    expected_rel_dirs = set(p.parent.relative_to(root_dir) for p in data_files)
    if reverse_dataset_root is not None and reverse_layout == "mirror":
        prune_mirror_generated_layout(
            mirror_root=reverse_dataset_root,
            expected_rel_dirs=expected_rel_dirs,
            wsd_subdir_name=wsd_subdir_name,
            jsonl_name=jsonl_name,
        )

    for data_path in data_files:
        folder = data_path.parent
        jsonl_path = folder / jsonl_name
        wsd_dir = folder / wsd_subdir_name
        instances = run_pipeline(
            data_path=data_path,
            jsonl_path=jsonl_path,
            template_id=template_id,
            wsd_dir=wsd_dir,
            overwrite_wsd=overwrite_wsd,
        )

        source_subset = folder.relative_to(root_dir).as_posix()
        if source_subset == ".":
            source_subset = ""

        reverse_local_instances: Optional[List[Instance]] = None
        if reverse_dataset_root is not None:
            reverse_local_instances = reverse_instances(instances)
            if reverse_layout == "mirror":
                reverse_folder = reverse_dataset_root / source_subset
                reverse_jsonl_path = reverse_folder / jsonl_name
                reverse_wsd_dir = reverse_folder / wsd_subdir_name
                if overwrite_wsd and reverse_wsd_dir.exists():
                    shutil.rmtree(reverse_wsd_dir, ignore_errors=True)
                write_jsonl(reverse_local_instances, reverse_jsonl_path)
                write_wsd_files(
                    reverse_local_instances,
                    reverse_wsd_dir,
                    overwrite=overwrite_wsd,
                    direction="reverse",
                )
            else:
                # Backward-compatible layout:
                # reverse/<Dataset>/out_wsd/<global_id>_*.wsd and only forward-style merged jsonl.
                write_reverse_wsd_files(
                    instances,
                    reverse_flat_wsd_dir,
                    overwrite=overwrite_wsd,
                    start_index=reverse_global_id,
                )

        for inst in instances:
            merged_rows.append(
                {
                    "id": str(global_id),
                    "template_id": inst.template_id,
                    "nodes": inst.nodes,
                    "edges": inst.edges,
                    "source_subset": source_subset,
                    "source_id": inst.id,
                }
            )
            global_id += 1

        if reverse_local_instances is not None:
            for inst in reverse_local_instances:
                reverse_merged_rows.append(
                    {
                        "id": str(reverse_global_id),
                        "template_id": inst.template_id,
                        "nodes": inst.nodes,
                        "edges": inst.edges,
                        "source_subset": source_subset,
                        "source_id": inst.id,
                    }
                )
                reverse_global_id += 1

    merged_jsonl_path = root_dir / jsonl_name
    write_jsonl_rows(merged_rows, merged_jsonl_path)

    if reverse_dataset_root is not None and reverse_layout == "mirror":
        reverse_merged_jsonl_path = reverse_dataset_root / jsonl_name
        write_jsonl_rows(reverse_merged_rows, reverse_merged_jsonl_path)


if __name__ == "__main__":
    """
    Default usage
    - for each subfolder containing data.txt:
      write instances.jsonl in that folder
      write .wsd files into that folder's out_wsd subfolder
    """

    overwrite_wsd = True

    wsd_subdir_name = "out_wsd"
    jsonl_name = "instances.jsonl"
    dataset_specs = [
        ("2Class_Inheritance", "inheritance"),
        ("3Class_Inheritance", "inheritance"),
        ("2Class_Aggregation", "aggregation"),
        ("3Class_Aggregation", "aggregation"),
        ("2Class_Composition", "composition"),
        ("3Class_Composition", "composition"),
        ("2Class_Dependency", "dependency"),
        ("3Class_Dependency", "dependency"),
    ]

    project_root = Path(__file__).resolve().parent.parent

    # Preferred new layout.
    # Always prefer data_reverse as source when it has data.txt.
    # The opposite side receives mirror-structured outputs.
    base_forward = project_root / "data_forward"
    base_reverse = project_root / "data_reverse"

    reverse_has_data = base_reverse.exists() and any(base_reverse.rglob("data.txt"))
    forward_has_data = base_forward.exists() and any(base_forward.rglob("data.txt"))

    if reverse_has_data:
        source_base = base_reverse
        reverse_base = base_forward
    else:
        source_base = base_forward
        reverse_base = base_reverse

    if source_base.exists():
        reverse_base.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] source_base={source_base} reverse_base={reverse_base}")
        for dataset_name, template_id in dataset_specs:
            root_dir = source_base / dataset_name
            if not root_dir.exists():
                print(f"[WARN] skip missing dataset dir: {root_dir}")
                continue
            run_for_root(
                root_dir=root_dir,
                template_id=template_id,
                overwrite_wsd=overwrite_wsd,
                wsd_subdir_name=wsd_subdir_name,
                jsonl_name=jsonl_name,
                reverse_root=reverse_base,
                reverse_layout="mirror",
            )
    else:
        # Backward-compatible legacy layout.
        reverse_root = project_root / "reverse"
        print(f"[INFO] using legacy layout under parent folder with reverse root {reverse_root}")
        for dataset_name, template_id in dataset_specs:
            root_dir = project_root / dataset_name
            if not root_dir.exists():
                print(f"[WARN] skip missing dataset dir: {root_dir}")
                continue
            run_for_root(
                root_dir=root_dir,
                template_id=template_id,
                overwrite_wsd=overwrite_wsd,
                wsd_subdir_name=wsd_subdir_name,
                jsonl_name=jsonl_name,
                reverse_root=reverse_root,
                reverse_layout="flat",
            )
