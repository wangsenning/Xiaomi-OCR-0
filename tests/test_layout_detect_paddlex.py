"""Unit tests for opt-in --preserve_polygon support in layout_detect_paddlex."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_pipeline.layout_detect_paddlex import (
    DEFAULT_POLYGON_SCHEMA,
    PRESERVED_POLYGON_SCHEMA,
    build_manifest,
    parse_args,
    serialize_regions,
    validate_or_write_manifest,
)


class _BrokenToList:
    def tolist(self):
        raise RuntimeError("boom")


class TestLayoutDetectPaddleX(unittest.TestCase):
    def _sample_result(self, polygon_points=None) -> dict:
        box = {
            "label": "text",
            "score": 0.95,
            "coordinate": [100.0, 200.0, 500.0, 400.0],
        }
        if polygon_points is not None:
            box["polygon_points"] = polygon_points
        return {"boxes": [box]}

    def test_default_serialization_preserves_exact_four_point_bbox(self) -> None:
        poly = [[100, 220], [300, 200], [500, 210], [490, 390], [110, 400]]
        result = self._sample_result(poly)

        # 3-arg call matches skills/xiaomi-ocr/mcp_ocr_server.py
        regions_default = serialize_regions(result, 1000, 1000)
        regions_explicit_false = serialize_regions(result, 1000, 1000, preserve_polygon=False)

        expected_region = {
            "index": 0,
            "label": "text",
            "score": 0.95,
            "bbox_2d": [100, 200, 500, 400],
            "polygon": [[100, 200], [500, 200], [500, 400], [100, 400]],
            "task_type": "text",
        }
        self.assertEqual(len(regions_default), 1)
        self.assertEqual(regions_default, regions_explicit_false)
        self.assertEqual(regions_default[0], expected_region)
        self.assertEqual(list(regions_default[0].keys()), list(expected_region.keys()))
        for pt in regions_default[0]["polygon"]:
            self.assertIs(type(pt[0]), int)
            self.assertIs(type(pt[1]), int)

    def test_opt_in_preserves_multi_vertex_rotated_quad_and_numpy_arrays(self) -> None:
        # 5-vertex contour as NumPy array
        poly5 = np.array([[100.0, 220.0], [300.0, 200.0], [500.0, 210.0], [490.0, 390.0], [110.0, 400.0]])
        regions = serialize_regions(self._sample_result(poly5), 1000, 500, preserve_polygon=True)
        exported = regions[0]["polygon"]
        self.assertEqual(
            exported,
            [[100.0, 440.0], [300.0, 400.0], [500.0, 420.0], [490.0, 780.0], [110.0, 800.0]],
        )
        for pt in exported:
            self.assertIs(type(pt[0]), float)
            self.assertIs(type(pt[1]), float)

        # 4-vertex rotated quad with repeated closing vertex (deduped and preserved as floats)
        quad = [[200, 100], [400, 150], [350, 350], [150, 300], [200, 100]]
        regions_quad = serialize_regions(self._sample_result(quad), 1000, 1000, preserve_polygon=True)
        quad_out = regions_quad[0]["polygon"]
        self.assertEqual(quad_out, [[200.0, 100.0], [400.0, 150.0], [350.0, 350.0], [150.0, 300.0]])
        for pt in quad_out:
            self.assertIs(type(pt[0]), float)
            self.assertIs(type(pt[1]), float)

    def test_opt_in_clips_out_of_bounds_polygon_to_page_rect(self) -> None:
        # Crosses left, right, top, and bottom page boundaries simultaneously
        poly = [[-100.0, -50.0], [1200.0, -50.0], [1200.0, 600.0], [-100.0, 600.0]]
        regions = serialize_regions(self._sample_result(poly), 1000, 500, preserve_polygon=True)
        exported = regions[0]["polygon"]
        self.assertEqual(len(exported), 4)
        self.assertEqual(
            sorted(exported),
            sorted([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]]),
        )

        # Rotated corner-crossing triangle clipped against top and right edges
        tri = [[800.0, 200.0], [1200.0, -200.0], [1200.0, 200.0]]
        regions_tri = serialize_regions(self._sample_result(tri), 1000, 1000, preserve_polygon=True)
        tri_out = regions_tri[0]["polygon"]
        self.assertEqual(
            sorted(tri_out),
            sorted([[800.0, 200.0], [1000.0, 0.0], [1000.0, 200.0]]),
        )
        for x, y in tri_out:
            self.assertIs(type(x), float)
            self.assertIs(type(y), float)

    def test_opt_in_falls_back_on_invalid_or_degenerate_polygons(self) -> None:
        expected_rect = [[100, 200], [500, 200], [500, 400], [100, 400]]
        bad_cases = [
            None,
            [],
            [[100, 200], [200, 300]],  # < 3 vertices
            [[100, 200], [200, 200], [300, 200]],  # collinear / zero area
            [[float("nan"), 200], [500, 200], [500, 400]],  # NaN
            [[float("inf"), 200], [500, 200], [500, 400]],  # Inf
            [[10**400, 200], [500, 200], [500, 400]],  # float() OverflowError
            [[-1e308, 100], [1e308, 100], [1e308, 300]],  # overflow in edge diff
            [[True, 200], [500, 200], [500, 400]],  # bool coord
            np.array([[True, False], [True, True], [False, True]]),  # np.bool_
            [[100, 200, 0], [500, 200, 0], [500, 400, 0]],  # 3D point
            [[-50, -50], [-10, -50], [-10, -10]],  # completely outside page
            _BrokenToList(),
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad):
                regions = serialize_regions(self._sample_result(bad), 1000, 1000, preserve_polygon=True)
                self.assertEqual(regions[0]["polygon"], expected_rect)
                for pt in regions[0]["polygon"]:
                    self.assertIs(type(pt[0]), int)
                    self.assertIs(type(pt[1]), int)

    def test_cli_flag_default_and_opt_in(self) -> None:
        default_args = parse_args(["--input_jsonl", "in.jsonl", "--output_jsonl", "out.jsonl"])
        self.assertFalse(default_args.preserve_polygon)

        opt_in_args = parse_args([
            "--input_jsonl", "in.jsonl",
            "--output_jsonl", "out.jsonl",
            "--preserve_polygon",
        ])
        self.assertTrue(opt_in_args.preserve_polygon)

    def test_manifest_default_schema_and_four_way_resume_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_path = tmp_dir / "layout.jsonl"
            manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")

            # Literally frozen upstream 4abc240 manifest structure
            frozen_upstream_manifest = {
                "layout_backend": "paddlex",
                "layout_model": None,
                "layout_threshold": 0.25,
                "layout_postprocess": "paddlex_native",
                "gpu_ids": [0],
                "batch_size": 8,
                "input_jsonl": "in.jsonl",
                "output_jsonl": str(out_path),
                "output_schema": {
                    "top_level": "input fields + layout_result",
                    "region_fields": ["index", "label", "score", "bbox_2d", "polygon", "task_type"],
                    "bbox_coordinates": "normalized integers in [0, 1000]",
                    "polygon": "normalized four-point bbox rectangle",
                },
                "label_task_mapping": {
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
                },
            }

            default_args = parse_args(["--input_jsonl", "in.jsonl", "--output_jsonl", str(out_path)])
            opt_in_args = parse_args([
                "--input_jsonl", "in.jsonl",
                "--output_jsonl", str(out_path),
                "--preserve_polygon",
            ])

            default_manifest = build_manifest(default_args)
            opt_in_manifest = build_manifest(opt_in_args)

            # Default manifest matches literally frozen upstream 4abc240 manifest
            self.assertEqual(default_manifest, frozen_upstream_manifest)
            self.assertEqual(default_manifest["output_schema"]["polygon"], DEFAULT_POLYGON_SCHEMA)
            self.assertEqual(opt_in_manifest["output_schema"]["polygon"], PRESERVED_POLYGON_SCHEMA)

            # 0. Fresh output writes manifest cleanly
            validate_or_write_manifest(out_path, manifest_path, default_manifest)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), frozen_upstream_manifest)

            # 1. Existing frozen upstream cache -> new default run resumes cleanly
            out_path.write_text('{"images": ["a.png"], "layout_result": []}\n', encoding="utf-8")
            manifest_path.write_text(json.dumps(frozen_upstream_manifest), encoding="utf-8")
            validate_or_write_manifest(out_path, manifest_path, default_manifest)

            # 2. Upstream default cache -> opt-in run must be rejected
            with self.assertRaises(SystemExit):
                validate_or_write_manifest(out_path, manifest_path, opt_in_manifest)

            # 3. Opt-in cache -> opt-in run resumes cleanly
            manifest_path.write_text(json.dumps(opt_in_manifest), encoding="utf-8")
            validate_or_write_manifest(out_path, manifest_path, opt_in_manifest)

            # 4. Opt-in cache -> default run must be rejected
            with self.assertRaises(SystemExit):
                validate_or_write_manifest(out_path, manifest_path, default_manifest)

            # 5. Non-empty output without manifest must be rejected
            manifest_path.unlink()
            with self.assertRaises(SystemExit):
                validate_or_write_manifest(out_path, manifest_path, default_manifest)


if __name__ == "__main__":
    unittest.main()
