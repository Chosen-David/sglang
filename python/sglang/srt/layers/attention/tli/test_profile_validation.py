"""CPU-only profile contracts, loading the actual module without SGLang/CUDA."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "tli_profile_under_test", Path(__file__).with_name("config.py")
)
_CONFIG = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONFIG)


class ProfileValidationTests(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_TLI_")}
        self.env_patch = patch.dict(os.environ, env, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_method_values(self):
        for field in ("FAR", "NEAR"):
            key = "SGLANG_TLI_" + field + "_METHOD"
            for value in ("avg", "minmax"):
                with patch.dict(os.environ, {key: value}):
                    self.assertEqual(
                        getattr(_CONFIG.TLIProfile(), field.lower() + "_method"), value
                    )
            for value in ("typo", "AVG", ""):
                with (
                    self.subTest(field=field, value=value),
                    patch.dict(os.environ, {key: value}),
                ):
                    with self.assertRaises(ValueError):
                        _CONFIG.TLIProfile()

    def test_subspace_dimensions_reject_invalid_geometry(self):
        p = _CONFIG.TLIProfile()
        for head, coarse in (
            (127, 32),
            (0, 32),
            (-2, 32),
            (128, 0),
            (128, -2),
            (128, 1),
            (128, 31),
            (128, 130),
        ):
            with self.subTest(head=head, coarse=coarse):
                p.coarse_dim = coarse
                with self.assertRaises(ValueError):
                    p.subspace_idx(head)

    def test_refine_dimensions_reject_invalid_geometry(self):
        p = _CONFIG.TLIProfile()
        for head, delta in (
            (127, 16),
            (0, 16),
            (-2, 16),
            (128, 0),
            (128, -1),
            (128, 65),
        ):
            with self.subTest(head=head, delta=delta):
                p.delta = delta
                with self.assertRaises(ValueError):
                    p.refine_idx(head)

    def test_valid_index_coordinates_and_boundaries(self):
        p = _CONFIG.TLIProfile()
        self.assertEqual(
            p.subspace_idx(128), list(range(48, 64)) + list(range(112, 128))
        )
        self.assertEqual(p.refine_idx(128), p.subspace_idx(128))
        for head in (2, 64, 128, 256):
            for width in (2, head):
                p.coarse_dim = width
                p.delta = width // 2
                expected = list(range(head // 2 - width // 2, head // 2)) + list(
                    range(head - width // 2, head)
                )
                self.assertEqual(p.subspace_idx(head), expected)
                self.assertEqual(p.refine_idx(head), expected)
                self.assertEqual(len(set(expected)), width)


if __name__ == "__main__":
    unittest.main()
