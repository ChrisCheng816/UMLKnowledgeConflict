import argparse
import io
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch convert SVG to PNG with consistent clarity.\n"
            "Recommended: keep a fixed pixels-per-unit (PPU), then optionally pad to a common canvas."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "reverse", "both"],
        default="both",
        help="Which split to process.",
    )
    parser.add_argument(
        "--forward-input",
        default="images/data_forward",
        help="Forward SVG root directory.",
    )
    parser.add_argument(
        "--reverse-input",
        default="images/data_reverse",
        help="Reverse SVG root directory.",
    )
    parser.add_argument(
        "--forward-output",
        default="image_forward",
        help="Forward PNG output root directory.",
    )
    parser.add_argument(
        "--reverse-output",
        default="image_reverse",
        help="Reverse PNG output root directory.",
    )
    parser.add_argument(
        "--px-per-unit",
        type=float,
        default=4.0,
        help="Pixels per SVG viewBox unit. Same value means same rendering sharpness across files.",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="PNG background color (e.g. white, transparent, #FFFFFF).",
    )
    parser.add_argument(
        "--pad-to-max",
        action="store_true",
        help="Pad all rendered PNGs to the max width/height in this run (centered, no stretch).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cairosvg", "inkscape"],
        default="auto",
        help="Rendering backend. auto = cairosvg first, then inkscape.",
    )
    return parser.parse_args()


_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z%]*)\s*$")


def _size_to_px(value: Optional[str], dpi: float = 96.0) -> Optional[float]:
    if not value:
        return None
    m = _SIZE_RE.match(value)
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "px").lower()
    if unit in ("", "px"):
        return num
    if unit == "pt":
        return num * dpi / 72.0
    if unit == "pc":
        return num * dpi / 6.0
    if unit == "in":
        return num * dpi
    if unit == "cm":
        return num * dpi / 2.54
    if unit == "mm":
        return num * dpi / 25.4
    if unit == "%":
        return None
    return None


def _parse_svg_geometry(svg_path: Path) -> Tuple[float, float]:
    """
    Returns (base_width, base_height) in 'SVG units'.
    Priority:
    1) viewBox width/height (best for cross-file consistent PPU)
    2) width/height converted to px
    """
    try:
        root = ET.parse(str(svg_path)).getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid SVG XML: {svg_path}") from e

    view_box = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if view_box:
        parts = re.split(r"[,\s]+", view_box.strip())
        if len(parts) == 4:
            try:
                vb_w = float(parts[2])
                vb_h = float(parts[3])
                if vb_w > 0 and vb_h > 0:
                    return vb_w, vb_h
            except ValueError:
                pass

    width_px = _size_to_px(root.attrib.get("width"))
    height_px = _size_to_px(root.attrib.get("height"))
    if width_px and height_px and width_px > 0 and height_px > 0:
        return width_px, height_px

    raise ValueError(
        f"Cannot infer geometry from SVG: {svg_path}. Please ensure viewBox or width/height exists."
    )


def _collect_svgs(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*.svg") if p.is_file())


def _render_size(base_w: float, base_h: float, ppu: float) -> Tuple[int, int]:
    out_w = max(1, int(math.ceil(base_w * ppu)))
    out_h = max(1, int(math.ceil(base_h * ppu)))
    return out_w, out_h


def _render_svg_to_png_bytes(svg_path: Path, out_w: int, out_h: int, background: str) -> bytes:
    import cairosvg

    return cairosvg.svg2png(
        url=str(svg_path),
        output_width=out_w,
        output_height=out_h,
        background_color=background,
    )


def _render_svg_to_png_bytes_inkscape(svg_path: Path, out_w: int, out_h: int, background: str) -> bytes:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        raise RuntimeError("inkscape command not found in PATH")

    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "tmp.png"
        cmd = [
            inkscape,
            str(svg_path),
            "--export-type=png",
            f"--export-filename={out_file}",
            f"--export-width={out_w}",
            f"--export-height={out_h}",
            f"--export-background={background}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Inkscape conversion failed for {svg_path}:\n{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return out_file.read_bytes()


def _pick_backend(requested: str) -> str:
    if requested == "cairosvg":
        return "cairosvg"
    if requested == "inkscape":
        return "inkscape"

    try:
        import cairosvg  # noqa: F401
        return "cairosvg"
    except Exception:
        if shutil.which("inkscape"):
            return "inkscape"
    raise RuntimeError(
        "No available SVG renderer found. Install one of:\n"
        "1) pip install cairosvg\n"
        "2) Inkscape and ensure 'inkscape' is in PATH"
    )


def _save_png_with_optional_padding(
    png_bytes: bytes,
    out_path: Path,
    canvas_size: Optional[Tuple[int, int]],
    background: str,
):
    if canvas_size is None:
        out_path.write_bytes(png_bytes)
        return

    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow is required when --pad-to-max is enabled. Install via: pip install pillow") from e

    canvas_w, canvas_h = canvas_size
    with Image.open(io.BytesIO(png_bytes)) as img:
        mode = "RGBA" if background == "transparent" else "RGB"
        bg = (0, 0, 0, 0) if background == "transparent" else background
        canvas = Image.new(mode, (canvas_w, canvas_h), bg)
        x = (canvas_w - img.width) // 2
        y = (canvas_h - img.height) // 2
        if img.mode != mode:
            img = img.convert(mode)
        canvas.paste(img, (x, y), img if mode == "RGBA" else None)
        canvas.save(out_path, format="PNG")


def _convert_one_tree(
    input_dir: Path,
    output_dir: Path,
    ppu: float,
    background: str,
    pad_to_max: bool,
    overwrite: bool,
    backend: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if ppu <= 0:
        raise ValueError("--px-per-unit must be > 0")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    svg_files = _collect_svgs(input_dir)
    if not svg_files:
        raise FileNotFoundError(f"No .svg files found in: {input_dir}")

    plans = []
    max_w, max_h = 0, 0
    for svg_path in svg_files:
        base_w, base_h = _parse_svg_geometry(svg_path)
        out_w, out_h = _render_size(base_w, base_h, ppu)
        rel = svg_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".png")
        plans.append((svg_path, out_path, out_w, out_h))
        max_w = max(max_w, out_w)
        max_h = max(max_h, out_h)

    canvas_size = (max_w, max_h) if pad_to_max else None

    converted = 0
    skipped = 0
    for svg_path, out_path, out_w, out_h in plans:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        if backend == "cairosvg":
            png_bytes = _render_svg_to_png_bytes(svg_path, out_w, out_h, background)
        else:
            png_bytes = _render_svg_to_png_bytes_inkscape(svg_path, out_w, out_h, background)

        _save_png_with_optional_padding(
            png_bytes=png_bytes,
            out_path=out_path,
            canvas_size=canvas_size,
            background=background,
        )
        converted += 1

    print(f"[DONE] input: {input_dir}")
    print(f"[DONE] output: {output_dir}")
    print(f"[DONE] svg files: {len(svg_files)}")
    print(f"[DONE] converted: {converted}")
    print(f"[DONE] skipped: {skipped}")
    print(f"[DONE] px_per_unit: {ppu}")
    print(f"[DONE] backend: {backend}")
    if pad_to_max:
        print(f"[DONE] padded canvas: {canvas_size[0]}x{canvas_size[1]}")
    return len(svg_files), converted, skipped


def _resolve_data_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / p).resolve()


def main():
    args = parse_args()
    backend = _pick_backend(args.backend)

    tasks = []
    if args.mode in ("forward", "both"):
        tasks.append((_resolve_data_path(args.forward_input), _resolve_data_path(args.forward_output), "forward"))
    if args.mode in ("reverse", "both"):
        tasks.append((_resolve_data_path(args.reverse_input), _resolve_data_path(args.reverse_output), "reverse"))

    total_svg = 0
    total_converted = 0
    total_skipped = 0
    for input_dir, output_dir, tag in tasks:
        print(f"[INFO] processing {tag}")
        svg_count, converted, skipped = _convert_one_tree(
            input_dir=input_dir,
            output_dir=output_dir,
            ppu=args.px_per_unit,
            background=args.background,
            pad_to_max=args.pad_to_max,
            overwrite=args.overwrite,
            backend=backend,
        )
        total_svg += svg_count
        total_converted += converted
        total_skipped += skipped

    print("[SUMMARY]")
    print(f"[SUMMARY] total svg files: {total_svg}")
    print(f"[SUMMARY] total converted: {total_converted}")
    print(f"[SUMMARY] total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
