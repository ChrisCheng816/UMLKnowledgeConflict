from __future__ import annotations

import json
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
    elif template_id == "association":
        for child, parent in zip(nodes[:-1], nodes[1:]):
            edges.append(f"{child} --> {parent}")
        return edges

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


def write_wsd_files(instances: List[Instance], out_dir: Path, overwrite: bool) -> None:
    """
    Generate one .wsd file per instance.

    Naming rule change
    - Output filename is the "largest parent class" of the instance,
      which is the first token in nodes, for example:
      Coupe Car -> Coupe.wsd

    Content rule
    - File content is still:
      @startuml
      ...
      @enduml
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, inst in enumerate(instances):
        if not inst.nodes:
            continue

        root_name = safe_filename(inst.nodes[0])
        out_path = out_dir / f"{i}_{root_name}.wsd"

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {out_path}")

        body = "\n".join(inst.edges)
        content = "@startuml\n" + body + "\n@enduml\n"
        out_path.write_text(content, encoding="utf-8")


def run_pipeline(
    data_path: Path,
    jsonl_path: Path,
    template_id: str,
    wsd_dir: Optional[Path],
    overwrite_wsd: bool,
) -> None:
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
        write_wsd_files(instances, wsd_dir, overwrite=overwrite_wsd)


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


def run_for_root(
    root_dir: Path,
    template_id: str,
    overwrite_wsd: bool,
    wsd_subdir_name: str,
    jsonl_name: str,
) -> None:
    """
    Run the pipeline for every data.txt found under root_dir.

    Output placement rule
    - For each folder that contains data.txt:
      instances.jsonl is written into the same folder
      .wsd files are written into a subfolder under the same folder
    """
    data_files = find_all_data_txt(root_dir)
    for data_path in data_files:
        folder = data_path.parent
        jsonl_path = folder / jsonl_name
        wsd_dir = folder / wsd_subdir_name
        run_pipeline(
            data_path=data_path,
            jsonl_path=jsonl_path,
            template_id=template_id,
            wsd_dir=wsd_dir,
            overwrite_wsd=overwrite_wsd,
        )


if __name__ == "__main__":
    """
    Default usage
    - root directory is ./Nagetive
    - for each subfolder containing data.txt:
      write instances.jsonl in that folder
      write .wsd files into that folder's out_wsd subfolder
    """

    template_id = "inheritance"
    overwrite_wsd = True

    wsd_subdir_name = "out_wsd"
    jsonl_name = "instances.jsonl"

    run_for_root(
        root_dir=Path("../2Class_Inheritance"),
        template_id="inheritance",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    run_for_root(
        root_dir=Path("../3Class_Inheritance"),
        template_id="inheritance",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    
    run_for_root(
        root_dir=Path("../2Class_Aggregation"),
        template_id="aggregation",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    run_for_root(
        root_dir=Path("../3Class_Aggregation"),
        template_id="aggregation",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    run_for_root(
        root_dir=Path("../2Class_Composition"),
        template_id="composition",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    run_for_root(
        root_dir=Path("../3Class_Composition"),
        template_id="composition",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    run_for_root(
        root_dir=Path("../2Class_Association"),
        template_id="association",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    run_for_root(
        root_dir=Path("../3Class_Association"),
        template_id="association",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    run_for_root(
        root_dir=Path("../2Class_Dependency"),
        template_id="dependency",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
    run_for_root(
        root_dir=Path("../3Class_Dependency"),
        template_id="dependency",
        overwrite_wsd=overwrite_wsd,
        wsd_subdir_name=wsd_subdir_name,
        jsonl_name=jsonl_name,
    )
