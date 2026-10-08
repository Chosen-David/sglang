#!/usr/bin/env python3
"""Read-only CPU/source-fragment review. Never imports SGLang or starts a model.

GPU selector and kernel execution are explicitly outside this verification.
"""
import ast
import contextlib
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'sglang-readonly'
OUT = Path(__file__).resolve().parent
start = time.perf_counter()
findings = {}

def tree(rel):
    return ast.parse((ROOT / rel).read_text())

def source_function(rel, name):
    node = next(n for n in ast.walk(tree(rel))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), rel, 'exec'), scope)
    return scope[name]

# Actual host-only region function; candidate/attention arithmetic below is NumPy.
idx = 'python/sglang/srt/layers/attention/tli/indexer.py'
regions = source_function(idx, '_taskmd_regions')
p = SimpleNamespace(block_size=64, sink_blocks=2, sliding_window=128,
                    alpha=.125, beta=.375, gamma=.625, k1_blocks=128,
                    token_budget=1024)
r = regions(SimpleNamespace(profile=p), 4096)
near_width = r['swa_lo_tok'] - r['near_blks'] * p.block_size
pad = r['nt_near'] - near_width
assert (near_width, pad, r['far_budget']) == (384, 384, 0)
selected = np.concatenate([np.arange(r['near_blks']*64, r['swa_lo_tok']),
                           np.zeros(pad, dtype=int), np.arange(128),
                           np.arange(4096-128, 4096)])
assert selected.size == 1024
v = np.zeros(4096); v[0] = 1
multiset_output = float(v[selected].mean())  # q=0 -> uniform weights over slots
set_output = float(v[np.unique(selected)].mean())
assert np.count_nonzero(selected == 0) == 385
assert abs(multiset_output - set_output) > .37
findings['prefill_padding_bias'] = {
    'verification': 'actual host region function + mathematical/NumPy consumer counterexample; no GPU selector execution',
    'S': 4096, 'alpha_beta_gamma': [.125, .375, .625], 'near_valid_tokens': near_width,
    'near_quota': r['nt_near'], 'invalid_zero_slots': pad,
    'token0_multiplicity': 385, 'distinct_selected_tokens': int(np.unique(selected).size),
    'unmasked_slot_attention': multiset_output, 'unique_token_attention': set_output}

K2, L = 1024, 1000
reps = max(K2 // L, 1); cut = reps * L; j = np.arange(K2)
grid = np.where(j < cut, j // reps, j - cut)
v = np.zeros(L); v[:24] = 1
assert grid.max() < L
assert float(v[grid].mean()) == .046875
findings['early_grid_not_dense'] = {
    'verification': 'NumPy of source grid formula; no CUDA', 'K2': K2, 'causal_length': L,
    'duplicated_first_tokens': K2 % L, 'grid_output': float(v[grid].mean()),
    'dense_output': float(v.mean())}

pred_rel = 'two-level-attention/benchmark/LongBench/pred.py'
pred_tree = tree(pred_rel)
branch = next(n for n in ast.walk(pred_tree) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Compare) and ast.unparse(n.test) == "dataset == 'samsum'")
pred_assignment = next(n for n in ast.walk(pred_tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'pred' for t in n.targets)
                       and 'tokenizer.decode(generated_content' in ast.unparse(n))
fragment = ast.parse('def fragment(dataset, model, tokenizer, input, max_gen, context_length):\n    pass').body[0]
fragment.body = [branch, pred_assignment, ast.Return(value=ast.Name(id='pred', ctx=ast.Load()))]
scope = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fragment], type_ignores=[])), pred_rel, 'exec'), scope)
tok = SimpleNamespace(eos_token_id=2, encode=lambda *a, **kw:[3], decode=lambda *a, **kw:'x')
try:
    scope['fragment']('samsum', SimpleNamespace(generate=lambda **kw:[[0, 1]]), tok, {}, 4, 10)
    raise AssertionError('expected missing local generated_content')
except UnboundLocalError as e:
    findings['samsum_runtime_error'] = {'verification':'executed unmodified AST branch and decode statement with mock model',
                                       'exception':type(e).__name__, 'message':str(e)}

class TinyTensor:
    def __init__(self, x): self.x = np.asarray(x)
    def __getitem__(self, k): return TinyTensor(self.x[k])
    def argmax(self, dim): return TinyTensor(self.x.argmax(axis=dim))
    def unsqueeze(self, dim): return TinyTensor(np.expand_dims(self.x, dim))
    def item(self): return self.x.item()

class TinyModel:
    def __init__(self): self.ids = iter([2, 1, 2]); self.calls = 0
    def __call__(self, **kwargs):
        self.calls += 1
        scores = np.full((1, 1, 3), -100.0)
        scores[0, 0, next(self.ids)] = 1
        return SimpleNamespace(logits=TinyTensor(scores), past_key_values=object())

manual = next(n for n in branch.orelse if isinstance(n, ast.With))
env = {'torch':SimpleNamespace(no_grad=contextlib.nullcontext),
       'model':TinyModel(), 'input':SimpleNamespace(input_ids=TinyTensor([[0]])),
       'q_input':SimpleNamespace(input_ids=[[]]), 'max_gen':4, 'tokenizer':tok}
exec(compile(ast.fix_missing_locations(ast.Module(body=[manual], type_ignores=[])), pred_rel, 'exec'), env)
assert env['generated_content'] == [2, 1, 2]
findings['first_eos_not_stopped'] = {'verification':'actual manual-generation AST with mock logits, not a model benchmark',
                                   'eos_id':2, 'generated_ids':env['generated_content'], 'model_calls':env['model'].calls}

eager_rel = 'two-level-attention/sparse_attn/ops/eager_decoding.py'
per_qh = next(n for n in ast.walk(tree(eager_rel)) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Name) and n.test.id == 'per_qh')
mask_assignment = next(n for n in per_qh.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'b_mask' for t in n.targets))
def repeat_np(m, pattern, g, bs):
    h = m.shape[0] // g
    return np.repeat(m.reshape(h, g, -1), bs, axis=-1)
env = {'m0':np.ones((32, 4160), dtype=bool), 'g_m':4, 'block_size':1,
       'bos_k':0, 'eos_k':4097, 'repeat':repeat_np}
exec(compile(ast.fix_missing_locations(ast.Module(body=[mask_assignment], type_ignores=[])), eager_rel, 'exec'), env)
assert env['b_mask'].shape == (8, 4, 4160)
try:
    np.where(env['b_mask'], np.zeros((8, 4, 4097)), -np.inf)
    raise AssertionError('expected incompatible dimensions')
except ValueError as e:
    findings['per_q_head_mask_axis'] = {'verification':'executed actual mask-assignment AST with NumPy repeat; no torch',
                                       'mask_shape':list(env['b_mask'].shape), 'score_shape':[8, 4, 4097],
                                       'exception':type(e).__name__, 'message':str(e)}

info_rel = 'two-level-attention/sparse_attn/info.py'
method_name = source_function(info_rel, 'get_method_name_with_info')
cfg = dict(method='tli', tli_enable_subspace=True, tli_enable_kmeans=False,
           tli_enable_layer_skip=False, tia_block_size=64, tia_level1_topk=128,
           tia_level2_topk=1024, tia_level2_cmp_ratio=4)
a = SimpleNamespace(**cfg, tli_alpha=.125, tli_beta=.375, tli_gamma=.625)
b = SimpleNamespace(**cfg, tli_alpha=.875, tli_beta=.875, tli_gamma=.5)
assert method_name(a) == method_name(b)
findings['configuration_name_collision'] = {
    'verification':'executed actual method-name function', 'different_configs_same_name':method_name(a),
    'condition':'same output directory, task and --t causes overwrite; run-specific postfix can avoid it'}

with tempfile.TemporaryDirectory(prefix='sglang-review-') as td:
    temp = Path(td); bin_dir = temp/'bin'; bin_dir.mkdir()
    stub = bin_dir/'python'
    stub.write_text('#!/bin/sh\nprintf "MOCK_PYTHON_FAILURE\\n"\nexit 23\n')
    stub.chmod(0o755)
    shell_env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
    run = subprocess.run(['bash', str(ROOT/'two-level-attention/benchmark/RULER/run_ruler.sh'), 'none','0','1'],
                         cwd=temp, env=shell_env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 0 and 'ALL DONE none' in run.stdout
    assert run.stdout.count('MOCK_PYTHON_FAILURE') == 33
    findings['runner_false_success'] = {'verification':'actual run_ruler.sh, only python binary replaced by exit-23 stub; no GPU/model commands',
                                       'python_failed_tasks':33, 'shell_exit_code':run.returncode,
                                       'completion_marker':'ALL DONE none', 'cd_also_failed':'cd:' in run.stderr}
    log = temp/'old.log'; log.write_text('old run: ALL DONE none\n')
    check = subprocess.run(['grep','-q','ALL DONE none',str(log)])
    assert check.returncode == 0
    findings['stale_marker'] = {'verification':'same grep completion predicate on a previous-run fixture',
                                'stale_log_accepted':True, 'not_a_live_remote_process_test':True}

    pred_dir = temp/'ruler'/'L4096'/'pred_1024'; pred_dir.mkdir(parents=True)
    def write_rows(name, answers, pred, count):
        row=json.dumps({'answers':answers, 'pred':pred, 'length':4096, 'budget':1024})+'\n'
        (pred_dir/name).write_text(row*count)
    write_rows('niah_single_1-none-01010000.jsonl', ['target'], 'wrong', 3)
    write_rows('niah_single_1-none-10080900.jsonl', ['target'], 'target', 1)
    write_rows('niah_single_2-none-01010000.jsonl', ['target'], 'wrong', 3)
    out_json = temp/'score'/'result.json'
    score = subprocess.run([sys.executable, str(ROOT/'two-level-attention/benchmark/RULER/score_ruler.py'),
                            '--root',str(temp/'ruler'),'--out',str(out_json)], capture_output=True, text=True, timeout=20)
    assert score.returncode == 0
    score_result = json.loads(out_json.read_text())
    assert score_result['L4096/none'] == {'niah_single_1':100., 'niah_single_2':0.}
    assert '| AVG | 50.00 |' in score.stdout
    findings['scorer_mixed_incomplete_runs'] = {'verification':'executed actual score_ruler.py against isolated synthetic JSONL fixtures',
                                               'result':score_result, 'new_partial_task_rows':1,
                                               'other_task_from_old_run':True, 'missing_tasks_rejected':False,
                                               'reported_AVG':50., 'exit_code':score.returncode}

# Syntax checks are deliberately separate from behavioral checks.
py_files = sorted(set((ROOT/'python/sglang/srt/layers/attention/tli').glob('*.py')) |
                  set((ROOT/'two-level-attention/sparse_attn').rglob('*.py')) |
                  set((ROOT/'two-level-attention/benchmark/LongBench').glob('*.py')) |
                  set((ROOT/'two-level-attention/benchmark/RULER').glob('*.py')))
py_errors = []
for path in py_files:
    try: ast.parse(path.read_text())
    except SyntaxError as e: py_errors.append({'file':str(path.relative_to(ROOT)), 'error':str(e)})
sh_files = sorted(set((ROOT/'two-level-attention/benchmark').rglob('*.sh')) |
                  set((ROOT/'two-level-attention/exp/trace/run_scripts').glob('*.sh')))
sh_errors = []
for path in sh_files:
    checked = subprocess.run(['bash','-n',str(path)], capture_output=True, text=True)
    if checked.returncode: sh_errors.append({'file':str(path.relative_to(ROOT)), 'error':checked.stderr})
assert not py_errors and not sh_errors
relevant = [idx, 'python/sglang/srt/layers/attention/tli/backend.py', pred_rel, eager_rel, info_rel,
            'two-level-attention/sparse_attn/indexer/tli_indexer.py',
            'two-level-attention/benchmark/RULER/run_ruler.sh',
            'two-level-attention/benchmark/RULER/chain_ruler.sh',
            'two-level-attention/benchmark/RULER/score_ruler.py']
result = {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'verification_scope':'read-only CPU/source fragments and controlled shell/scorer fixtures; no torch, CUDA, model or remote-process validation',
          'findings':findings,
          'syntax':{'python_files':len(py_files),'python_errors':py_errors,'shell_files':len(sh_files),'shell_errors':sh_errors},
          'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in relevant},
          'elapsed_seconds':round(time.perf_counter()-start,3)}
(OUT/'evidence_by_gpt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))
