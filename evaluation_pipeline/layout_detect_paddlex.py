#!/usr/bin/env python3
"""Multi-GPU PaddleX PP-DocLayoutV3 inference with v0-compatible JSONL output."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_MODEL_DIR = None
DEFAULT_THRESHOLD = 0.25
DEFAULT_BATCH_SIZE = 8
REGION_FIELDS = (
    "index",
    "label",
    "score",
    "bbox_2d",
    "polygon",
    "task_type",
)
DEFAULT_POLYGON_SCHEMA = "normalized four-point bbox rectangle"
PRESERVED_POLYGON_SCHEMA = (
    "clipped detector polygon_points normalized as floats in [0, 1000] "
    "(falls back to 4-point bbox rectangle)"
)

LABEL_TASK_MAPPING = {
    "text": [
        "abstract", "algorithm", "content", "doc_title", "figure_title",
        "paragraph_title", "reference_content", "text", "vertical_text",
        "vision_footnote", "seal", "formula_number", "header", "footer",
        "number", "footnote", "aside_text", "reference",
    ],
    "table": ["table"],
    "formula": ["display_formula", "inline_formula"],
    "skip": ["chart", "image", "footer_image", "header_image"],
    "abandon": [],
}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            images = row.get("images")
            if not isinstance(images, list) or not images:
                raise ValueError(f"Line {line_number} must contain a nonempty images list")
            rows.append(row)
    return rows


def task_type_for_label(label: str) -> str | None:
    for task_type, labels in LABEL_TASK_MAPPING.items():
        if label in labels:
            return task_type
    return None


def normalize_device(device: str) -> str:
    if device.startswith("cuda"):
        return f"gpu{device[len('cuda'):]}"
    return device


def extract_boxes(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "get") and result.get("boxes") is not None:
        return result["boxes"]
    try:
        inner = result["res"]
    except (KeyError, TypeError, AttributeError):
        inner = None
    if hasattr(inner, "get") and inner.get("boxes") is not None:
        return inner["boxes"]
    json_value = getattr(result, "json", None)
    if callable(json_value):
        json_value = json_value()
    if isinstance(json_value, dict):
        if json_value.get("boxes") is not None:
            return json_value["boxes"]
        inner = json_value.get("res")
        if isinstance(inner, dict) and inner.get("boxes") is not None:
            return inner["boxes"]
    return []


def _polygon_shoelace_area(points: list[list[float]] | list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(points)
    for idx in range(n):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)


def _clip_polygon_to_page_rect(
    pts: list[tuple[float, float]],
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    """Clip an ordered 2D contour against `[0, width] x [0, height]` via Sutherland-Hodgman."""
    poly = list(pts)
    if len(poly) < 3 or width <= 0.0 or height <= 0.0:
        return []

    def _inside(p: tuple[float, float], kind: str, bound: float) -> bool:
        if kind == "x_min":
            return p[0] >= bound
        if kind == "x_max":
            return p[0] <= bound
        if kind == "y_min":
            return p[1] >= bound
        return p[1] <= bound

    def _intersect(
        p1: tuple[float, float],
        p2: tuple[float, float],
        kind: str,
        bound: float,
    ) -> tuple[float, float] | None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if kind in ("x_min", "x_max"):
            if dx == 0.0 or not math.isfinite(dx):
                return None
            t = (bound - p1[0]) / dx
            ix = bound
            iy = p1[1] + t * dy
        else:
            if dy == 0.0 or not math.isfinite(dy):
                return None
            t = (bound - p1[1]) / dy
            ix = p1[0] + t * dx
            iy = bound
        if not (math.isfinite(ix) and math.isfinite(iy)):
            return None
        return (ix, iy)

    for kind, bound in (
        ("x_min", 0.0),
        ("x_max", float(width)),
        ("y_min", 0.0),
        ("y_max", float(height)),
    ):
        if not poly:
            return []
        clipped: list[tuple[float, float]] = []
        prev = poly[-1]
        prev_in = _inside(prev, kind, bound)
        for curr in poly:
            curr_in = _inside(curr, kind, bound)
            if curr_in != prev_in:
                inter = _intersect(prev, curr, kind, bound)
                if inter is None:
                    return []
                clipped.append(inter)
            if curr_in:
                clipped.append(curr)
            prev = curr
            prev_in = curr_in
        poly = clipped
    return poly


def _normalize_polygon_points(
    raw_polygon: Any,
    image_width: int,
    image_height: int,
    fallback: list[list[int]],
) -> list[list[float]] | list[list[int]]:
    """Clip and normalize an ordered 2D contour to float `[0, 1000]`, else return `fallback`."""
    if raw_polygon is None or image_width <= 0 or image_height <= 0:
        return fallback
    raw_seq = raw_polygon
    tolist = getattr(raw_seq, "tolist", None)
    if callable(tolist):
        try:
            raw_seq = tolist()
        except Exception:
            return fallback
    if not isinstance(raw_seq, (list, tuple)) or len(raw_seq) < 3:
        return fallback

    pts: list[tuple[float, float]] = []
    for raw_pt in raw_seq:
        pt = raw_pt
        pt_tolist = getattr(pt, "tolist", None)
        if callable(pt_tolist):
            try:
                pt = pt_tolist()
            except Exception:
                return fallback
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return fallback
        if isinstance(pt[0], (bool, np.bool_)) or isinstance(pt[1], (bool, np.bool_)):
            return fallback
        try:
            px = float(pt[0])
            py = float(pt[1])
        except (TypeError, ValueError, OverflowError):
            return fallback
        if not (math.isfinite(px) and math.isfinite(py)):
            return fallback
        pts.append((px, py))

    clipped = _clip_polygon_to_page_rect(pts, float(image_width), float(image_height))
    if len(clipped) < 3:
        return fallback

    dedup: list[tuple[float, float]] = []
    for px, py in clipped:
        if not dedup or abs(px - dedup[-1][0]) > 1e-9 or abs(py - dedup[-1][1]) > 1e-9:
            dedup.append((px, py))
    if len(dedup) >= 2 and abs(dedup[0][0] - dedup[-1][0]) <= 1e-9 and abs(dedup[0][1] - dedup[-1][1]) <= 1e-9:
        dedup.pop()
    if len(dedup) < 3:
        return fallback
    area = _polygon_shoelace_area(dedup)
    if not (math.isfinite(area) and area > 1e-9):
        return fallback

    inv_w = 1000.0 / float(image_width)
    inv_h = 1000.0 / float(image_height)
    norm_pts: list[list[float]] = []
    for px, py in dedup:
        nx = px * inv_w
        ny = py * inv_h
        if not (math.isfinite(nx) and math.isfinite(ny)):
            return fallback
        pt_norm = [max(0.0, min(1000.0, nx)), max(0.0, min(1000.0, ny))]
        if not norm_pts or pt_norm != norm_pts[-1]:
            norm_pts.append(pt_norm)
    if len(norm_pts) >= 2 and norm_pts[0] == norm_pts[-1]:
        norm_pts.pop()
    if len(norm_pts) < 3:
        return fallback
    norm_area = _polygon_shoelace_area(norm_pts)
    if not (math.isfinite(norm_area) and norm_area > 0.0):
        return fallback
    return norm_pts


def serialize_regions(
    result: Any,
    image_width: int,
    image_height: int,
    preserve_polygon: bool = False,
) -> list[dict[str, Any]]:
    regions = []
    for box in extract_boxes(result):
        label = str(box.get("label", ""))
        coordinate = box.get("coordinate", [])
        if len(coordinate) != 4:
            continue
        task_type = task_type_for_label(label)
        if task_type is None or task_type == "abandon":
            continue
        x1, y1, x2, y2 = [float(value) for value in coordinate]
        x1 = max(0.0, min(x1, float(image_width)))
        y1 = max(0.0, min(y1, float(image_height)))
        x2 = max(0.0, min(x2, float(image_width)))
        y2 = max(0.0, min(y2, float(image_height)))
        if x2 <= x1 or y2 <= y1:
            continue
        bbox_2d = [
            int(x1 / image_width * 1000),
            int(y1 / image_height * 1000),
            int(x2 / image_width * 1000),
            int(y2 / image_height * 1000),
        ]
        x1_normalized, y1_normalized, x2_normalized, y2_normalized = bbox_2d
        fallback_polygon = [
            [x1_normalized, y1_normalized],
            [x2_normalized, y1_normalized],
            [x2_normalized, y2_normalized],
            [x1_normalized, y2_normalized],
        ]
        polygon = (
            _normalize_polygon_points(
                box.get("polygon_points"),
                image_width,
                image_height,
                fallback_polygon,
            )
            if preserve_polygon
            else fallback_polygon
        )
        regions.append({
            "index": len(regions),
            "label": label,
            "score": float(box.get("score", 0.0)),
            "bbox_2d": bbox_2d,
            "polygon": polygon,
            "task_type": task_type,
        })
    return regions


def load_batch(
    items: list[tuple[int, str]],
) -> tuple[list[int], list[Image.Image], list[tuple[int, str, str]]]:
    indices = []
    images = []
    failures = []
    for index, image_path in items:
        path = image_path[7:] if image_path.startswith("file://") else image_path
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
            indices.append(index)
        except Exception as error:
            print(f"  [WARN] load failed: {image_path}: {error}", flush=True)
            failures.append((index, image_path, str(error)))
    return indices, images, failures


def worker_main(
    worker_id: int,
    gpu_id: int,
    model_dir: str | None,
    threshold: float,
    batch_size: int,
    items: list[tuple[int, str]],
    result_queue: mp.Queue,
    preserve_polygon: bool = False,
) -> None:
    try:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddlex import create_model

        device = normalize_device(f"cuda:{gpu_id}")
        print(
            f"  [GPU-{gpu_id}] Loading PaddleX PP-DocLayoutV3 "
            f"(threshold={threshold})...",
            flush=True,
        )
        options = {"device": device, "threshold": threshold}
        if model_dir is not None:
            options["model_dir"] = model_dir
        model = create_model("PP-DocLayoutV3", **options)
        print(f"  [GPU-{gpu_id}] PaddleX layout ready", flush=True)

        for offset in range(0, len(items), batch_size):
            batch_items = items[offset:offset + batch_size]
            indices, images, failures = load_batch(batch_items)
            for index, image_path, error in failures:
                result_queue.put(("error", index, image_path, error))
            if not images:
                continue

            arrays = [np.asarray(image) for image in images]
            results = list(model.predict(arrays))
            if len(results) != len(images):
                raise RuntimeError(
                    f"PaddleX result count {len(results)} != image count {len(images)}"
                )
            for index, image, result in zip(indices, images, results):
                regions = serialize_regions(
                    result,
                    *image.size,
                    preserve_polygon=preserve_polygon,
                )
                result_queue.put(("result", index, list(image.size), regions))
            for image in images:
                image.close()

        del model
        result_queue.put(("done", worker_id))
    except BaseException as error:
        result_queue.put((
            "worker_error",
            worker_id,
            gpu_id,
            f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
        ))


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise argparse.ArgumentTypeError("--gpu_ids must not be empty")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("--gpu_ids must not contain duplicate GPU IDs")
    return gpu_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PaddleX PP-DocLayoutV3 multi-GPU layout inference",
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu_ids", default="0", help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--flush_every", type=int, default=100)
    parser.add_argument(
        "--preserve_polygon",
        action="store_true",
        help=(
            "Export clipped detector polygon_points normalized to [0, 1000] "
            "(falls back to 4-point bbox rectangle)"
        ),
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.flush_every < 1:
        parser.error("--flush_every must be >= 1")
    try:
        args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return args


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    polygon_schema = (
        PRESERVED_POLYGON_SCHEMA
        if getattr(args, "preserve_polygon", False)
        else DEFAULT_POLYGON_SCHEMA
    )
    return {
        "layout_backend": "paddlex",
        "layout_model": args.model_dir,
        "layout_threshold": args.threshold,
        "layout_postprocess": "paddlex_native",
        "gpu_ids": args.gpu_ids,
        "batch_size": args.batch_size,
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "output_schema": {
            "top_level": "input fields + layout_result",
            "region_fields": list(REGION_FIELDS),
            "bbox_coordinates": "normalized integers in [0, 1000]",
            "polygon": polygon_schema,
        },
        "label_task_mapping": LABEL_TASK_MAPPING,
    }


def validate_or_write_manifest(
    output_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        if not manifest_path.is_file():
            raise SystemExit(
                f"[FAIL] Existing output has no manifest: {output_path}; "
                "refuse to resume an unknown run"
            )
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatibility_keys = (
            "layout_backend",
            "layout_model",
            "layout_threshold",
            "layout_postprocess",
            "input_jsonl",
            "output_jsonl",
            "output_schema",
        )
        mismatches = [
            key for key in compatibility_keys
            if existing_manifest.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise SystemExit(
                f"[FAIL] Existing Layout manifest does not match current run: "
                f"{mismatches}"
            )
    else:
        temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)


def main() -> None:
    args = parse_args()
    if args.model_dir is not None and not Path(args.model_dir).is_dir():
        raise SystemExit(f"[FAIL] PaddleX model directory not found: {args.model_dir}")

    dataset = load_jsonl(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest = build_manifest(args)
    validate_or_write_manifest(output_path, manifest_path, manifest)

    completed_paths = set()
    if output_path.exists():
        for row in load_jsonl(str(output_path)):
            if isinstance(row.get("layout_result"), list):
                completed_paths.add(row["images"][0])

    todo = [
        (index, row["images"][0])
        for index, row in enumerate(dataset)
        if row["images"][0] not in completed_paths
    ]
    print(f"Input pages: {len(dataset)}")
    print(f"Already completed: {len(completed_paths)}")
    print(f"To process: {len(todo)}")
    print(f"PaddleX GPUs: {args.gpu_ids}")
    print(f"Threshold: {args.threshold}")
    if args.preserve_polygon:
        print("Preserve polygon: True")
    if not todo:
        print("Nothing to process.")
        return

    shards = [[] for _ in args.gpu_ids]
    for position, item in enumerate(todo):
        shards[position % len(shards)].append(item)

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=max(64, len(args.gpu_ids) * args.batch_size * 4))
    processes = []
    for worker_id, (gpu_id, shard) in enumerate(zip(args.gpu_ids, shards)):
        process = context.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_id,
                args.model_dir,
                args.threshold,
                args.batch_size,
                shard,
                result_queue,
                args.preserve_polygon,
            ),
        )
        process.start()
        processes.append(process)
        print(f"  GPU {gpu_id}: {len(shard)} pages")

    interrupted = False

    def handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\nSignal {signum} received; stopping workers...", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    completed = 0
    errors = []
    done_workers = set()
    started_at = time.time()
    progress = tqdm(total=len(todo), desc="PaddleX Layout", unit="page")
    with output_path.open("a", encoding="utf-8") as output_file:
        while completed < len(todo) and not interrupted:
            try:
                message = result_queue.get(timeout=10)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    break
                continue

            kind = message[0]
            if kind == "result":
                _, index, image_size, regions = message
                row = dict(dataset[index])
                row["layout_result"] = regions
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed += 1
                progress.update(1)
                if completed % args.flush_every == 0:
                    output_file.flush()
            elif kind == "error":
                _, index, image_path, error = message
                errors.append(f"{image_path}: {error}")
                completed += 1
                progress.update(1)
            elif kind == "done":
                done_workers.add(message[1])
            elif kind == "worker_error":
                _, worker_id, gpu_id, error = message
                errors.append(f"worker={worker_id} gpu={gpu_id}: {error}")
                print(f"\n[ERROR] worker={worker_id} gpu={gpu_id}\n{error}", flush=True)

        output_file.flush()
    progress.close()

    for process in processes:
        if interrupted and process.is_alive():
            process.terminate()
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if process.exitcode not in (0, None):
            errors.append(f"worker pid={process.pid} exitcode={process.exitcode}")

    elapsed = time.time() - started_at
    print(f"Done: {completed}/{len(todo)} pages in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")
    if interrupted:
        raise SystemExit(130)
    if completed != len(todo) or errors:
        print(f"[FAIL] errors={len(errors)} completed={completed}/{len(todo)}", file=sys.stderr)
        for error in errors[:20]:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
