"""CPU dispatch/selection regression; DeepSelect and capture are bounded mocks."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

TOPK = torch.topk


def production(available, ds):
    path = Path(__file__).with_name("indexer.py")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "TLIIndexer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "select_decode_batched")
    ns = {"torch": torch, "_ds_available": available, "ds_topk": ds}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[fn.name]


class DecodeTopKDispatchTests(unittest.TestCase):
    def invoke(self, enabled=True, available=True, capture=False, skip=False, n=3):
        generator = torch.Generator().manual_seed(123)
        profile = SimpleNamespace(
            taskmd=False, block_size=4, k1_blocks=2, coarse_dim=4,
            use_ds_topk=enabled, sliding_blocks=1, sink_blocks=1,
            near_len=8, sliding_window=4, token_budget=8, far_tokens=2,
        )
        obj = SimpleNamespace(profile=profile, nd2=4, idx1=torch.arange(4),
                              skip_far=skip, _q_refine=lambda q: q)
        pool = {
            "kq_q": torch.randint(0, 16, (n, 32, 2, 4), generator=generator,
                                  dtype=torch.uint8),
            "kq_sc": torch.rand(n, 32, 2, generator=generator),
            "kq_mn": torch.randn(n, 32, 2, generator=generator),
            "kmin": torch.randn(n, 8, 2, 4, generator=generator),
            "kmax": torch.randn(n, 8, 2, 4, generator=generator),
        }
        q = torch.randn(n, 4, 4, generator=generator)
        ds_calls = []

        def ds(scores, k):
            # A deterministic CPU stand-in, NOT DeepSelect numerical validation.
            ds_calls.append(tuple(scores.shape))
            return TOPK(scores, k, dim=-1).indices, scores

        with patch.object(torch, "topk", wraps=TOPK) as spy, patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=capture
        ):
            output = production(lambda: available, ds)(
                obj, pool, torch.arange(n - 1, -1, -1),
                [32 - i * 3 for i in range(n)], q,
            )
        l1_calls = sum(tuple(c.args[0].shape) == (n, 2, 8)
                       for c in spy.call_args_list)
        return output, l1_calls, ds_calls

    def test_deepselect_does_not_also_execute_torch_l1_topk(self):
        for n in (1, 3):
            for skip in (False, True):
                with self.subTest(n=n, skip=skip):
                    output, calls, ds = self.invoke(skip=skip, n=n)
                    self.assertEqual(calls, 0)
                    self.assertEqual(ds[0], (n * 2, 8))
                    reference, _, _ = self.invoke(enabled=False, skip=skip, n=n)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)

    def test_fallbacks_execute_one_torch_l1_topk_and_no_deepselect(self):
        for kwargs in ({"enabled": False}, {"available": False}, {"capture": True}):
            for skip in (False, True):
                with self.subTest(kwargs=kwargs, skip=skip):
                    output, calls, ds = self.invoke(skip=skip, **kwargs)
                    self.assertEqual(calls, 1)
                    self.assertEqual(ds, [])
                    reference, _, _ = self.invoke(enabled=False, skip=skip)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
