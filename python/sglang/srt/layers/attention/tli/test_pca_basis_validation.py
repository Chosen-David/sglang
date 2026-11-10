"""Actual PCA initialization on CPU; no full service or CUDA execution."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


def initialize(basis, tp=1, rank=0, local_heads=2, layers=2):
    path = Path(__file__).with_name("backend.py")
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TLISparseAttnBackend"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_init_from_runner"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    parallel = SimpleNamespace(attn_tp_size=tp, attn_tp_rank=rank, attn_dcp_size=1)
    dist = ModuleType("sglang.srt.distributed")
    dist.get_parallel = lambda: parallel
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=None,
        req_to_token_pool=SimpleNamespace(req_to_token=None),
        model_config=SimpleNamespace(
            head_dim=4,
            num_hidden_layers=layers,
            get_num_kv_heads=lambda *a: local_heads,
        ),
    )
    obj = SimpleNamespace(
        runner=runner,
        profile=SimpleNamespace(
            layer_skip_path=None,
            proj_basis_path="test-only",
            proj_rank=2,
            dense_threshold=16,
        ),
    )
    with (
        patch.dict(sys.modules, {"sglang.srt.distributed": dist}),
        patch.object(torch, "load", return_value=basis),
    ):
        ns["_init_from_runner"](obj, runner)
    return obj.proj_basis


class PCABasisValidationTests(unittest.TestCase):
    def test_invalid_layer_and_head_layouts_rejected(self):
        for shape, tp, heads in (
            ((1, 2, 4, 2), 1, 2),
            ((2, 1, 4, 2), 1, 2),
            ((2, 3, 4, 2), 2, 1),
            ((2, 4, 4, 2), 2, 1),
        ):
            with self.subTest(shape=shape, tp=tp, heads=heads):
                with self.assertRaises(ValueError):
                    initialize(torch.randn(shape), tp=tp, local_heads=heads)

    def test_scalar_basis_rejected(self):
        with self.assertRaises(ValueError):
            initialize(torch.tensor(1.0))

    def test_invalid_tensor_rank_and_head_dimension_rejected(self):
        for shape in ((2, 2, 2), (2, 2, 4, 1, 2), (2, 2, 5, 2), (2, 2, 1, 2)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    initialize(torch.zeros(shape))

    def test_valid_tp_slices_exactly_match(self):
        basis = torch.arange(2 * 8 * 4 * 2).reshape(2, 8, 4, 2).float()
        for tp in (1, 2, 4, 8):
            outputs = []
            for rank in range(tp):
                got = initialize(basis, tp=tp, rank=rank, local_heads=8 // tp)
                expected = basis[:, rank * (8 // tp) : (rank + 1) * (8 // tp)]
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                outputs.append(got)
            torch.testing.assert_close(torch.cat(outputs, dim=1), basis, rtol=0, atol=0)

    def test_rank_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            initialize(torch.zeros(2, 2, 4, 3))


if __name__ == "__main__":
    unittest.main()
