"""CPU profile/host-region checks, not kernel or GPU performance tests."""

import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("geometry_config", ROOT / "config.py")
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)


def regions(profile):
    tree = ast.parse((ROOT / "indexer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "TLIIndexer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "_taskmd_regions")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "indexer.py", "exec"), ns)
    return ns[fn.name](SimpleNamespace(profile=profile), 4096)


class GeometryPageBudgetTests(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_TLI_")}
        self.env = patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def profile(self, pages, tokens="0", k1=2):
        with patch.dict(os.environ, {
            "SGLANG_TLI_K1_BLOCKS": str(k1),
            "SGLANG_TLI_NEAR_PAGES": str(pages),
            "SGLANG_TLI_NEAR_TOKENS": tokens,
            "SGLANG_TLI_NEAR_LEN": "256",
        }):
            return CONFIG.TLIProfile()

    def test_near_pages_cannot_exceed_total_page_budget(self):
        for tokens in ("0", "off"):
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(ValueError, "K1"):
                    self.profile(3, tokens)

    def test_valid_boundary_quotas_keep_exact_total(self):
        for pages in (0, 1, 2):
            for tokens in ("0", "off"):
                with self.subTest(pages=pages, tokens=tokens):
                    r = regions(self.profile(pages, tokens))
                    self.assertEqual((r["nb_near"], r["nb_far"]), (pages, 2-pages))
                    self.assertEqual(r["gamma_off"], tokens == "off")
        r = regions(self.profile(0, k1=0))
        self.assertEqual((r["nb_near"], r["nb_far"]), (0, 0))

    def test_legacy_defaults_unchanged(self):
        p = CONFIG.TLIProfile()
        self.assertFalse(p.taskmd)
        self.assertEqual(p.k1_blocks, 128)
        self.assertIsNone(p.nt_near_pages)


if __name__ == "__main__":
    unittest.main()
