"""Execute production host-routing methods with NumPy test doubles, without CUDA.

These tests count host reads and check routing/lengths; they do not validate
Torch attention arithmetic, CUDA synchronization cost, or graph execution.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import numpy as np


class HostValues:
    def __init__(self, values):
        self.values = list(values)
        self.reads = 0

    def tolist(self):
        self.reads += 1
        return list(self.values)


class TensorDouble(np.ndarray):
    def view(self, *shape):
        return self.reshape(*shape)


def backend():
    path = Path(__file__).with_name("backend.py")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "TLISparseAttnBackend")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("init_forward_metadata", "forward_decode")]
    shim = SimpleNamespace(Tensor=np.ndarray, nn=SimpleNamespace(Module=object),
                           empty_like=np.empty_like)
    ns = {"torch": shim, "time": time}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), ns)
    obj = type("BackendProbe", (), {n.name: ns[n.name] for n in methods})()
    obj.index_pools = {0: {"row_of": {}, "S": [], "free": []}}
    obj._p1_cache = obj._p1_lens = None
    obj._use_graph_path = False
    obj.head_dim, obj.num_kv_heads, obj.dense_threshold = 2, 1, 32
    obj.token_to_kv_pool = SimpleNamespace(get_kv_buffer=lambda _: (None, None))
    obj.req_to_token = np.arange(64).reshape(2, 32)
    obj._get_indexer = lambda _: None
    obj.timer = SimpleNamespace(tick=lambda: 0, add=lambda *a: None,
                                maybe_report=lambda: None)
    obj.seen = []

    def dense(q, locs, *args):
        obj.seen.append(locs.tolist())
        return np.full_like(q, len(locs))

    obj._dense_attn = dense
    return obj


def batch(mirror):
    return SimpleNamespace(req_pool_indices=HostValues([1, 0]),
                           seq_lens=HostValues([3, 2]), seq_lens_cpu=mirror,
                           forward_mode=SimpleNamespace(is_decode=lambda: True))


def decode(obj, fb):
    q = np.zeros((2, 2)).view(TensorDouble)
    return obj.forward_decode(q, q, q, SimpleNamespace(layer_id=0), fb,
                              save_kv_cache=False)


class ShortHostLengthsTests(unittest.TestCase):
    def test_short_mirrors_fall_back_once_for_all_layers(self):
        for mirror in ([], [99], np.array([99])):
            with self.subTest(mirror=repr(mirror)):
                obj, fb = backend(), batch(mirror)
                obj.init_forward_metadata(fb)
                for _ in range(4):
                    np.testing.assert_array_equal(decode(obj, fb), [[3, 3], [2, 2]])
                self.assertEqual(fb.seq_lens.reads, 1)
                self.assertEqual(fb.req_pool_indices.reads, 1)
                self.assertEqual(obj.seen, [[32, 33, 34], [0, 1]] * 4)

    def test_complete_or_padded_mirror_never_reads_device_lengths(self):
        for mirror in ([3, 2], [3, 2, 99], np.array([3, 2])):
            obj, fb = backend(), batch(mirror)
            obj.init_forward_metadata(fb)
            np.testing.assert_array_equal(decode(obj, fb), [[3, 3], [2, 2]])
            self.assertEqual(fb.seq_lens.reads, 0)

    def test_missing_or_malformed_mirror_preserves_single_fallback(self):
        for mirror in (None, object(), [object(), 2]):
            obj, fb = backend(), batch(mirror)
            obj.init_forward_metadata(fb)
            for _ in range(4):
                np.testing.assert_array_equal(decode(obj, fb), [[3, 3], [2, 2]])
            self.assertEqual(fb.seq_lens.reads, 1)

    def test_next_forward_refreshes_lengths(self):
        obj, fb = backend(), batch([])
        obj.init_forward_metadata(fb)
        fb.seq_lens.values = [4, 5]
        obj.init_forward_metadata(fb)
        np.testing.assert_array_equal(decode(obj, fb), [[4, 4], [5, 5]])
        self.assertEqual(fb.seq_lens.reads, 2)

    def test_graph_dispatch_does_not_consume_host_snapshot(self):
        obj, fb = backend(), batch([])
        obj._use_graph_path = True
        sentinel = object()
        obj._forward_decode_graph = lambda *a: sentinel
        self.assertIs(decode(obj, fb), sentinel)
        self.assertEqual(fb.seq_lens.reads, 0)

    def test_early_exit_still_resets_both_snapshots(self):
        obj, fb = backend(), batch([])
        obj.init_forward_metadata(fb)
        fb.forward_mode = None
        obj.init_forward_metadata(fb)
        self.assertIsNone(obj._p1_cache)
        self.assertIsNone(obj._p1_lens)


if __name__ == "__main__":
    unittest.main()
