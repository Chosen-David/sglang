"""CPU numerical/index-volume checks; no CUDA or full service validation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def production(fused=None):
    path = Path(__file__).with_name('backend.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'TLISparseAttnBackend')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == '_sparse_attn_batched')
    ns = {'torch': torch, 'tli_sparse_gather_attn_dot': fused}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns[fn.name]


class IndexProbe(TorchDispatchMode):
    def __init__(self, buffers):
        self.storages = {x.untyped_storage().data_ptr() for x in buffers}
        self.index_shapes = []
        self.values = []

    def __torch_dispatch__(self, fn, types, args=(), kwargs=None):
        result = fn(*args, **(kwargs or {}))
        if (fn is torch.ops.aten.index.Tensor
                and args[0].untyped_storage().data_ptr() in self.storages):
            self.index_shapes.append([tuple(x.shape) for x in args[1] if x is not None])
            self.values.append(result.clone())
        return result


def inputs(n, hkv, groups, dim, dtype):
    gen = torch.Generator().manual_seed(42)
    k = torch.randn(24, hkv, dim, generator=gen).to(dtype)
    v = torch.randn(24, hkv, dim, generator=gen).to(dtype)
    q = torch.randn(n, hkv * groups, dim, generator=gen).to(dtype)
    rows = torch.arange(n - 1, -1, -1)
    reqs = torch.arange(n)
    table = torch.arange(n * 8).view(n, 8)
    lengths = torch.tensor([3 + i for i in range(n)])
    # Duplicates and both per-row and capacity sentinels; valid lane always exists.
    sel = torch.tensor([0, 1, 1, 7]).view(1, 1, 4).expand(n, hkv, 4).clone()
    sel[:, :, -1] = lengths[:, None]
    if n > 1:
        sel[-1, :, -1] = 8
    return q, rows, sel, lengths, SimpleNamespace(req_pool_indices=reqs), table, k, v


def reference(args, hkv, groups):
    q, rows, sel, lengths, fb, table, k, v = args
    n, _, dim = q.shape
    bound = lengths.view(-1, 1, 1)
    valid = sel < bound
    positions = torch.minimum(sel, bound - 1)
    ks, vs = [], []
    for i in range(n):
        ks.append(torch.stack([k[table[fb.req_pool_indices[rows[i]], positions[i, h]], h]
                               for h in range(hkv)]))
        vs.append(torch.stack([v[table[fb.req_pool_indices[rows[i]], positions[i, h]], h]
                               for h in range(hkv)]))
    ks, vs = torch.stack(ks).float(), torch.stack(vs).float()
    query = q[rows].float().view(n, hkv, groups, dim)
    scores = torch.einsum('nhgd,nhkd->nhgk', query, ks) * dim**-.5
    weights = torch.softmax(scores.masked_fill(~valid.unsqueeze(2), float('-inf')), -1)
    return torch.einsum('nhgk,nhkd->nhgd', weights, vs).view(n, hkv * groups, dim), ks, vs


class SparseGatherIndexingTests(unittest.TestCase):
    def invoke(self, args, hkv, groups, fused=False, callback=None):
        q, rows, sel, lengths, fb, table, k, v = args
        backend = SimpleNamespace(head_dim=q.shape[-1],
                                  profile=SimpleNamespace(use_sparse_attn_kernel=fused))
        pool = SimpleNamespace(get_kv_buffer=lambda _: (k, v))
        return production(callback)(backend, q, rows, sel, lengths, fb, table,
                                    pool, 0, hkv, groups)

    def test_exact_values_across_shapes_dtypes_and_sentinels(self):
        for n in (1, 3):
            for hkv in (1, 2):
                for groups in (1, 4):
                    for dim in (8, 32):
                        for dtype in (torch.float32, torch.float16, torch.bfloat16):
                            with self.subTest(n=n, hkv=hkv, groups=groups, dim=dim, dtype=dtype):
                                args = inputs(n, hkv, groups, dim, dtype)
                                expected, ks, vs = reference(args, hkv, groups)
                                with IndexProbe(args[-2:]) as probe:
                                    actual = self.invoke(args, hkv, groups)
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                                torch.testing.assert_close(probe.values[0].float().reshape_as(ks), ks, rtol=0, atol=0)
                                torch.testing.assert_close(probe.values[1].float().reshape_as(vs), vs, rtol=0, atol=0)

    def test_kv_indices_do_not_expand_across_head_dimension(self):
        args = inputs(3, 2, 4, 128, torch.float32)
        with IndexProbe(args[-2:]) as probe:
            self.invoke(args, 2, 4)
        self.assertEqual(probe.index_shapes, [[(3, 2, 4), (1, 2, 1)]] * 2)

    def test_fused_dispatch_still_bypasses_eager_gathers(self):
        args = inputs(3, 2, 4, 8, torch.float16)
        sentinel, calls = object(), []
        def fused(*a, **kw):
            calls.append((a, kw))
            return sentinel
        with IndexProbe(args[-2:]) as probe:
            actual = self.invoke(args, 2, 4, True, fused)
        self.assertIs(actual, sentinel)
        self.assertEqual(probe.index_shapes, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['S_loc'], 24)
        self.assertEqual(tuple(calls[0][1]['valid'].shape), (3, 2, 4))


if __name__ == '__main__':
    unittest.main()
