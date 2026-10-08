#!/usr/bin/env python3
"""Readonly follow-up review of 15eea9a. No torch/CUDA/model imports.

Source fragments and controlled shell/scorer/merger fixtures only.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'sglang-latest-readonly'
OUT = Path(__file__).resolve().parent
start = time.perf_counter()
results = {}

def parse(rel): return ast.parse((ROOT/rel).read_text())
def assignment(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
def run_nodes(nodes, env, rel):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),rel,'exec'),env)
def function(rel, name):
    f=next(n for n in ast.walk(parse(rel)) if isinstance(n,ast.FunctionDef) and n.name==name)
    f.decorator_list=[];f.returns=None
    for a in f.args.args: a.annotation=None
    env={};run_nodes([f],env,rel);return env[name]

class Tensor(np.ndarray):
    def __new__(cls,x): return np.asarray(x).view(cls)
    def view(self,*shape): return Tensor(np.asarray(self).reshape(*shape))

torch_np=SimpleNamespace(where=lambda a,b,c:Tensor(np.where(a,b,c)),
                         cat=lambda xs,dim=-1:np.concatenate(xs,axis=dim),
                         ones_like=np.ones_like, no_grad=contextlib.nullcontext)
idx='python/sglang/srt/layers/attention/tli/indexer.py'
tree_idx=parse(idx)
select=next(n for n in ast.walk(tree_idx) if isinstance(n,ast.FunctionDef) and n.name=='_select_batched_taskmd')
p=SimpleNamespace(block_size=64,sink_blocks=2,sliding_window=128,
                  alpha=.125,beta=.375,gamma=.625,k1_blocks=128,token_budget=1024)
region_fn=function(idx,'_taskmd_regions')
S,t=4096,2047
r=region_fn(SimpleNamespace(profile=p),t+1)
width=r['swa_lo_tok']-r['near_blks']*64
assert width==128 and r['nt_near']==768
keep=np.zeros((1,1,768),dtype=bool);keep[...,:width]=True
values=np.zeros((1,1,768),dtype=int);values[...,:width]=np.arange(r['near_blks']*64,r['swa_lo_tok'])
env={'torch':torch_np,'keep_n2':keep,'S_r':Tensor([t+1]),'i_n2':Tensor(values)}
run_nodes([assignment(select,'sel_n2')],env,idx)
sel=np.concatenate([env['sel_n2'],np.arange(128).reshape(1,1,-1),
                    np.arange(t+1-128,t+1).reshape(1,1,-1)],axis=-1)
backend='python/sglang/srt/layers/attention/tli/backend.py'
bfunc=next(n for n in ast.walk(parse(backend)) if isinstance(n,ast.FunctionDef) and n.name=='_sparse_extend_one')
env2={'sel':sel,'S':S}
run_nodes([assignment(bfunc,'valid')],env2,backend)
future=sel>t
assert (future & env2['valid']).sum()==640
v=np.zeros(S);v[t+1]=1
actual=float(v[sel[env2['valid']]].mean())
correct=float(v[sel[(sel<=t) & env2['valid']]].mean())
assert actual==.625 and correct==0
results['F01_row_sentinel_causal_leak']={
 'verification':'actual region function, selection sentinel assignment and consumer valid assignment; NumPy attention, no GPU',
 'chunk_length':S,'query_position':t,'row_sentinel':t+1,
 'unmasked_future_slots':640,'q_zero_output':actual,'causal_output':correct,
 'early_repair_applies':t<1024,'last_row_test_misses':True}

# B02 fix works for the former early-grid example (source expression execution).
grid_assign=next(n for n in ast.walk(select) if isinstance(n,ast.Assign)
                 and any(isinstance(x,ast.Name) and x.id=='grid' for x in n.targets))
ge={'torch':torch_np,'j':Tensor(np.arange(1024).reshape(1,-1)), 'L':Tensor([1000]),'S':4096}
run_nodes([grid_assign],ge,idx)
assert np.array_equal(ge['grid'][0,:1000],np.arange(1000)) and np.all(ge['grid'][0,1000:]==4096)
results['old_B02_identity_fix']={'source_fragment_pass':True,'GPU_not_run':True}

pred='two-level-attention/benchmark/LongBench/pred.py'
tp=parse(pred)
sam=next(n for n in ast.walk(tp) if isinstance(n,ast.If) and ast.unparse(n.test)=="dataset == 'samsum'")
env={'torch':torch_np,'input':SimpleNamespace(input_ids=np.array([[10,20]])),
     'q_input':SimpleNamespace(input_ids=np.array([[30,40]])), 'max_gen':4,
     'tokenizer':SimpleNamespace(eos_token_id=2,encode=lambda *a,**k:[3])}
seen={}
def generate(**kw):
    seen['ids']=kw['input_ids'].tolist()
    return np.concatenate([kw['input_ids'],np.array([[7,8]])],axis=-1)
env['model']=SimpleNamespace(generate=generate)
run_nodes(sam.body,env,pred)
assert seen['ids']==[[10,20,30,40]] and env['generated_content']==[7,8]
results['old_B03_local_fix']={'source_branch_pass':True,'already_sliced_q_input_is_rejoined':True,'real_tokenizer_not_run':True}

# The unconditional first-token deletion remains before that fix.
drop=next(n for n in ast.walk(tp) if isinstance(n,ast.Assign)
          and any(isinstance(a,ast.Attribute) and a.attr=='input_ids' for a in n.targets)
          and 'q_input.input_ids[:, 1:]' in ast.unparse(n))
de={'q_input':SimpleNamespace(input_ids=np.array([[701,702,703]]))}
run_nodes([drop],de,pred)
assert de['q_input'].input_ids.tolist()==[[702,703]]
results['F07_qwen_non_bos_token_drop']={
 'verification':'actual slicing statement on synthetic non-BOS IDs; official Qwen config verified separately, not a tokenizer/model execution',
 'input_synthetic_ids':[701,702,703],'after_slice':[702,703],
 'official_config_url':'https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json',
 'official_config_add_bos_token':False,'official_config_bos_token':None}

eager='two-level-attention/sparse_attn/ops/eager_decoding.py'
peq=next(n for n in ast.walk(parse(eager)) if isinstance(n,ast.If) and ast.unparse(n.test)=='per_qh')
ee={'m0':np.ones((32,4160),dtype=bool),'g_m':4,'block_size':1,'eos_k':4097,'bos_k':0,
    'repeat':lambda m,pattern,g,bs:np.repeat(m.reshape(8,g,-1),bs,axis=-1)}
run_nodes([assignment(peq,'b_mask')],ee,eager)
assert ee['b_mask'].shape==(8,4,4097)
results['old_B05_mask_axis_fix']={'source_fragment_pass':True,'mask_shape':[8,4,4097],'GPU_not_run':True}

# Projection still bases the forced SWA on padded fine-score width.
hf='two-level-attention/sparse_attn/indexer/tli_indexer.py'
cm=next(n for n in ast.walk(parse(hf)) if isinstance(n,ast.FunctionDef) and n.name=='compute_mask')
forces=[n for n in ast.walk(cm) if isinstance(n,ast.Assign)
        and any(isinstance(x,ast.Subscript) and isinstance(x.value,ast.Name) and x.value.id=='topk_mask' for x in n.targets)
        and ('swa_lo_tok:' in ast.unparse(n) or ':sink_tok' in ast.unparse(n))]
assert len(forces)==2
T,Tpad=4097,4160
mask=np.zeros((1,1,8,Tpad),dtype=bool)
fe={'topk_mask':mask,'sink_tok':128,'swa_lo_tok':Tpad-128}
run_nodes(forces,fe,hf)
metric_fn=function('two-level-attention/sparse_attn/metrics.py','add_select_result')
metrics=SimpleNamespace(threshold=2048,select_tokens=[])
metric_fn(metrics,np.array([T-1]),mask,1)
reported=metrics.select_tokens[0];consumed=int(mask[0,0,0,:T].sum())
assert reported==256 and consumed==193
results['F06_projection_padding_budget']={
 'verification':'actual final forced-mask assignments and Metrics.add_select_result on a K2=256 zero-mid-budget case; no full selector/GPU',
 'real_length':T,'padded_length':Tpad,'K2':256,
 'reported_selected_tokens':reported,'consumer_real_tokens':consumed,
 'forced_real_SWA_tokens':int(mask[0,0,0,Tpad-128:T].sum()),'expected_SWA':128}

with tempfile.TemporaryDirectory(prefix='sglang-followup-') as td:
    temp=Path(td);bindir=temp/'bin';bindir.mkdir()
    stub=bindir/'python';stub.write_text('#!/bin/sh\nprintf "MOCK_FAILURE\\n"\nexit 23\n');stub.chmod(0o755)
    penv=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'])
    runner=ROOT/'two-level-attention/benchmark/RULER/run_ruler.sh'
    # Override only hardcoded cd; execute the actual shell body with exit-23 Python.
    rr=subprocess.run(['bash','-c','cd(){ return 0; }; source "$1" "${@:2}"','review',str(runner),'none','0','1'],
                      cwd=temp,env=penv,capture_output=True,text=True,timeout=20)
    assert rr.returncode==1 and rr.stdout.count('MOCK_FAILURE')==33
    marker=next(l for l in rr.stdout.splitlines() if l.startswith('ALL DONE'))
    log=temp/'failed.log';log.write_text(rr.stdout)
    assert subprocess.run(['grep','-q','ALL DONE none',str(log)]).returncode==0
    penv['REVIEW_LOG']=str(log)
    chain=ROOT/'two-level-attention/benchmark/RULER/chain_ruler.sh'
    wrapper='cd(){ return 0; }; grep(){ if [ "${3:-}" = "/tmp/ruler_fullkv.log" ]; then command grep "$1" "$2" "$REVIEW_LOG"; else command grep "$@"; fi; }; source "$1"'
    cr=subprocess.run(['bash','-c',wrapper,'review',str(chain)],cwd=temp,env=penv,capture_output=True,text=True,timeout=20)
    assert cr.returncode==1 and cr.stdout.count('MOCK_FAILURE')==66
    assert 'ALL DONE quest (FAILED=33)' in cr.stdout
    results['F02_failed_marker_and_dependency']={
      'verification':'actual runner and chain scripts; cd and input log path intercepted, python replaced by failure stub; no GPU',
      'runner_exit_now_correct':rr.returncode,'failure_marker':marker,
      'chain_grep_accepts_failure':True,'chain_runs_quest_then_tia_despite_quest_failure':True,
      'chain_failed_python_calls':66,'chain_final_exit':cr.returncode}

    scorer=ROOT/'two-level-attention/benchmark/RULER/score_ruler.py'
    tasks=next(n.value for n in parse('two-level-attention/benchmark/RULER/score_ruler.py').body
               if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='TASKS' for x in n.targets))
    tasks=ast.literal_eval(tasks)
    def write_rows(root,L,task,pred,n,stamp='10081043'):
        d=root/f'L{L}'/'pred_1024';d.mkdir(parents=True,exist_ok=True)
        row=json.dumps({'pred':pred,'answers':['target'],'length':L,'budget':1024})+'\n'
        (d/f'{task}-none-{stamp}.jsonl').write_text(row*n)
    # Only 2 of 33 cells; both meet min_samples. Current scorer still emits a number and exit 0.
    small=temp/'small'
    write_rows(small,4096,tasks[0],'target',100)
    write_rows(small,4096,tasks[1],'wrong',100,'01010000')
    out1=temp/'score1'/'scores.json'
    sr=subprocess.run([sys.executable,str(scorer),'--root',str(small),'--out',str(out1)],capture_output=True,text=True,timeout=20)
    d1=json.loads(out1.read_text())
    assert sr.returncode==0 and '| AVG | 50.00 |' in sr.stdout and d1['incomplete_cells']==[]
    # One partial task at L4096 drops its ten complete siblings from AVG too.
    big=temp/'big'
    for L in (4096,8192,16384):
        for i,task in enumerate(tasks):
            write_rows(big,L,task,'target' if L==4096 else 'wrong',1 if (L==4096 and i==0) else 100)
    out2=temp/'score2'/'scores.json'
    s2=subprocess.run([sys.executable,str(scorer),'--root',str(big),'--out',str(out2)],capture_output=True,text=True,timeout=20)
    assert s2.returncode==0 and '| AVG | 0.00 |' in s2.stdout
    d2=json.loads(out2.read_text())
    good=[v for k,vs in d2['scores'].items() for task,v in vs.items() if d2['n'][k][task]>=100]
    assert len(good)==32 and float(np.mean(good))==31.25
    results['F05_scorer_incomplete_and_wrong_grouping']={
      'verification':'actual score_ruler.py executed on isolated fixtures',
      'two_of_33_cells_exit':sr.returncode,'two_cells_AVG':50,'missing_cells_in_json_incomplete_list':False,
      'one_short_cell_complete_siblings_dropped':10,'reported_AVG':0,'complete_cell_diagnostic_mean':31.25,
      'note':'formal AVG should be blocked on an incomplete frozen matrix; 31.25 is only the diagnostic mean of complete cells'}

    # Execute full merger source; change only ROOT to the isolated fixture directory.
    merge_rel='two-level-attention/exp/trace/run_scripts/merge_ruler_b7.py'
    mt=parse(merge_rel);assignment(mt,'ROOT').value=ast.Constant(value=str(temp/'project'))
    data_dir=temp/'project'/'exp'/'results_ruler';data_dir.mkdir(parents=True)
    base={};b7={}
    method='tli_64_128_1024_c4_A'
    for L in (4096,8192,16384):
        for m,score in [('none',80),('quest_64_16',70),('tia_64_128_1024_c4',60),(method,10)]:
            base[f'L{L}/{m}']={task:score for task in tasks}
        b7[f'L{L}/{method}']={task:90 for task in tasks}
    (data_dir/'ruler_table_final.json').write_text(json.dumps(base))
    (data_dir/'ruler_b7.json').write_text(json.dumps(b7))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(mt),merge_rel,'exec'),{'__name__':'__main__'})
    merged=json.loads((data_dir/'ruler_table_e71_final.json').read_text())
    assert merged['TLI_C0']['AVG']==90 and merged['TLI_B7']['AVG']==90
    results['F04_control_overwrite_in_legacy_merge']={
      'verification':'actual full merger AST, only ROOT redirected to temporary fixtures',
      'input_C0':10,'input_B7':90,'output_C0':90,'output_B7':90}
    (data_dir/'ruler_b7.json').write_text(json.dumps({'scores':b7,'n':{},'incomplete_cells':[]}))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(mt),merge_rel,'exec'),{'__name__':'__main__'})
        raise AssertionError('new schema should break old merger')
    except IndexError as e:
        results['F03_score_schema_consumer_break']={'verification':'same actual merger with current scorer envelope',
                                                  'exception':type(e).__name__,'message':str(e)}

pyfiles=sorted(set((ROOT/'python/sglang/srt/layers/attention/tli').glob('*.py'))|
               set((ROOT/'two-level-attention/sparse_attn').rglob('*.py'))|
               set((ROOT/'two-level-attention/benchmark/LongBench').glob('*.py'))|
               set((ROOT/'two-level-attention/benchmark/RULER').glob('*.py')))
for f in pyfiles: ast.parse(f.read_text())
shfiles=sorted(set((ROOT/'two-level-attention/benchmark').rglob('*.sh'))|
               set((ROOT/'two-level-attention/exp/trace/run_scripts').glob('*.sh')))
for f in shfiles: subprocess.run(['bash','-n',str(f)],check=True,capture_output=True)
sources=[idx,backend,pred,eager,hf,'two-level-attention/sparse_attn/metrics.py',
         'two-level-attention/benchmark/RULER/score_ruler.py',
         'two-level-attention/benchmark/RULER/run_ruler.sh',
         'two-level-attention/benchmark/RULER/chain_ruler.sh',
         'two-level-attention/exp/trace/run_scripts/merge_ruler_b7.py']
data={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'scope':'source fragments, NumPy, isolated actual shell/scorer/merger; no GPU/real model/remote process',
      'checks':results,'syntax':{'python_files':len(pyfiles),'shell_files':len(shfiles),'errors':[]},
      'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
      'elapsed_seconds':round(time.perf_counter()-start,3)}
(OUT/'evidence_latest_by_gpt.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
