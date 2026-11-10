"""CPU tests of actual eager metadata/decode bodies, without CUDA imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import time

import torch


def backend_methods():
    path = Path(__file__).with_name('backend.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'TLISparseAttnBackend')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('init_forward_metadata', 'forward_decode')]
    namespace = {'torch': torch, 'time': time}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return type('BackendProbe', (), {n.name: namespace[n.name] for n in methods})


class HostValues:
    def __init__(self, values):
        self.values, self.reads = values, 0

    def tolist(self):
        self.reads += 1
        return list(self.values)


def batch(reqs=(0, 1), lengths=(2, 3), cpu=True, decode=True):
    return SimpleNamespace(req_pool_indices=HostValues(reqs),
                           seq_lens=HostValues(lengths),
                           seq_lens_cpu=torch.tensor(lengths) if cpu else None,
                           forward_mode=SimpleNamespace(is_decode=lambda: decode))


def backend():
    obj = backend_methods()()
    obj.index_pools = {0: {'row_of': {}, 'S': [], 'free': []}}
    obj._p1_cache = None
    obj._p1_lens = None
    obj._use_graph_path = False
    obj.head_dim, obj.num_kv_heads, obj.dense_threshold = 2, 1, 32
    obj.token_to_kv_pool = SimpleNamespace(get_kv_buffer=lambda _: (None, None))
    obj.req_to_token = torch.arange(64).view(2, 32)
    obj._get_indexer = lambda _: None
    obj.timer = SimpleNamespace(tick=lambda: 0, add=lambda *args: None,
                                maybe_report=lambda: None)
    obj.observed = []

    def dense(q, locs, *args):
        obj.observed.append(locs.tolist())
        return torch.full_like(q, len(locs))

    obj._dense_attn = dense
    return obj


def decode(obj, fb):
    q = torch.zeros(2, 2)
    return obj.forward_decode(q, q, q, SimpleNamespace(layer_id=0), fb,
                              save_kv_cache=False)


class HostMetadataCacheTests(unittest.TestCase):
    def test_request_identity_mismatch_does_not_use_old_lengths(self):
        obj = backend()
        obj.init_forward_metadata(batch(lengths=(1, 1)))
        current = batch(reqs=(1, 0), lengths=(3, 2))
        actual = decode(obj, current)
        torch.testing.assert_close(actual, torch.tensor([[3., 3.], [2., 2.]]))
        self.assertEqual(obj.observed, [[32, 33, 34], [0, 1]])

    def test_early_exits_clear_both_snapshots(self):
        for mode in ('empty_pools', 'extend', 'missing_mode', 'bad_requests'):
            with self.subTest(mode=mode):
                obj = backend()
                obj.init_forward_metadata(batch())
                fb = batch()
                if mode == 'empty_pools':
                    obj.index_pools = {}
                elif mode == 'extend':
                    fb.forward_mode = SimpleNamespace(is_decode=lambda: False)
                elif mode == 'missing_mode':
                    fb.forward_mode = None
                else:
                    fb.req_pool_indices = object()
                obj.init_forward_metadata(fb)
                self.assertIsNone(obj._p1_cache)
                self.assertIsNone(obj._p1_lens)

    def test_without_cpu_mirror_reads_lengths_once_for_multiple_layers(self):
        obj, fb = backend(), batch(cpu=False)
        obj.init_forward_metadata(fb)
        for _ in range(4):
            decode(obj, fb)
        self.assertEqual(fb.seq_lens.reads, 1)
        self.assertEqual(fb.req_pool_indices.reads, 1)

    def test_cpu_mirror_and_next_forward_refresh(self):
        obj, fb = backend(), batch()
        obj.init_forward_metadata(fb)
        decode(obj, fb)
        self.assertEqual(fb.seq_lens.reads, 0)

        fb.seq_lens_cpu.copy_(torch.tensor([4, 5]))
        obj.init_forward_metadata(fb)
        actual = decode(obj, fb)
        torch.testing.assert_close(actual, torch.tensor([[4., 4.], [5., 5.]]))
        self.assertEqual(fb.seq_lens.reads, 0)

    def test_graph_dispatch_bypasses_host_values(self):
        obj, fb = backend(), batch()
        obj._use_graph_path = True
        sentinel = object()
        obj._forward_decode_graph = lambda *args: sentinel
        self.assertIs(decode(obj, fb), sentinel)
        self.assertEqual(fb.req_pool_indices.reads, 0)
        self.assertEqual(fb.seq_lens.reads, 0)

    def test_existing_decode_row_recycling_is_preserved(self):
        obj = backend()
        pool = {'row_of': {0: 1, 1: 2, 7: 3}, 'S': [-1, 2, 3, 5], 'free': []}
        obj.index_pools = {0: pool}
        obj.init_forward_metadata(batch())
        self.assertEqual(pool['row_of'], {0: 1, 1: 2})
        self.assertEqual(pool['S'], [-1, 2, 3, -1])
        self.assertEqual(pool['free'], [3])


if __name__ == '__main__':
    unittest.main()
