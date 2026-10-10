# 2026-10-11 03:16 sglang two-level-indexer 小时审查报告

**审查人**: kimi3
**审查区间**: `8144a66e87c68805940f10950dfd6776931f18a9..cc30fc11d4dc20b6c5780db12e645ad7654ade3e`
**新提交**:
- `e5dfc19e6` Fix 076/077/078 from GPT 2026-10-10 2131 re-audit (#203)
- `cc30fc11d` docs(advice): 075/076/077/078 验收补记——四项全合入主仓，独立复验全绿

**审查范围**: 本次变更集中在 `two-level-attention/sparse_attn/info.py` 与 `two-level-attention/benchmark/`（LongBench/RULER 指针协议）; `python/sglang/srt/layers/attention/tli/` 无改动, `exp/trace/` 无代码改动。

**结论**: 发现 2 处 076（TL-E121-OUTPUT-ID）引入的新缺陷,均有实测证据。077/078 修复本身经测试套件验证通过,无新增问题。

---

## 发现 1：sidecar manifest 文件名超出文件系统上限（076 截断未考虑 sidecar 后缀）

### 位置
- `two-level-attention/sparse_attn/info.py:148-166` `truncate_output_name_keep_hash(limit=245)`
- `two-level-attention/benchmark/LongBench/pred.py:475-490` 调用处

### 问题描述
076 在最终输出路径旁写入 sidecar manifest,文件名为 `{out}.jsonl.tli_manifest.json`。`truncate_output_name_keep_hash` 把 `out_fn` 限制为 245 字符,但未考虑 sidecar 后缀长度。导致 `out_fn` 被截到 245 时,sidecar 实际文件名长达 `245 + 6(.jsonl) + 18(.tli_manifest.json) = 269` 字节,超过 Linux ext4 的 255 字节文件名上限,`open(sidecar, "w")` 直接抛 `OSError: [Errno 36] File name too long`。

### 影响
- LongBench 输出路径一旦触发截断（数据集前缀/时间戳较长）,076 写入门自身无法创建 sidecar,流程崩溃。
- 即使 gate 在读取 sidecar 时 catch OSError,写入 sidecar 的 OSError 发生在 `gate_output_treatment_identity` 返回之后、`pred.py` 的 `open(sidecar, "w")` 处,会以裸 traceback 暴露,破坏 076 fail-closed 的统一报错口径。

### 实测证据
```bash
cd /home/wangyuanshuo02/sglang/two-level-attention
python - <<'PY'
import sys, os, types, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
import sparse_attn.info as INFO

def cfg():
    return types.SimpleNamespace(
        method='tli', tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
        tia_level2_cmp_ratio=2, tli_enable_subspace=True, tli_subspace='full',
        tli_alpha=0.0, tli_beta=0.0, tli_gamma=1.0, tli_sparse_prefill=False,
        tli_far_method='minmax', tli_near_method='avg', tli_far_select='4bit',
        tli_near_select='4bit', tli_far_clusters=256, tli_far_niter=10,
        tli_far_blocks=16, tli_far_tokens=512, tli_sim=0.9, tli_sim_dims='subspace',
        tli_sigma_select='none', tli_moba=False, tli_sigma=8.0, tli_per_q_head=False,
        tli_proj_basis=None, tli_static_pair=False, tli_layer_skip_path=None,
        tli_enable_kmeans=True, tli_enable_layer_skip=True)

tmp = tempfile.mkdtemp()
try:
    c = cfg()
    mn = INFO.get_method_name_with_info(c)
    prefix = 'd'*200
    t = '09090909'
    out_fn = INFO.truncate_output_name_keep_hash(prefix, mn, t)
    print('out_fn len:', len(out_fn))          # 245
    sidecar_name = out_fn + '.jsonl' + INFO.TREATMENT_MANIFEST_SIDECAR_SUFFIX
    print('sidecar filename len:', len(sidecar_name))  # 269
    out_path = os.path.join(tmp, out_fn + '.jsonl')
    manifest = INFO.get_treatment_manifest_json(c)
    sidecar = INFO.gate_output_treatment_identity(out_path, manifest)
    with open(sidecar, 'w') as f: f.write(manifest)  # OSError 36
finally:
    shutil.rmtree(tmp)
PY
```
输出:
```
out_fn len: 245
sidecar filename len: 269
OSError: [Errno 36] File name too long: '/tmp/.../ddddd...-tli_64_128_1024_c2_B..._h34c60f5a4f-09090909.jsonl.tli_manifest.json'
```

### 修复建议
把 `out_fn` 的上限从 245 降到能容纳完整 sidecar 文件名。当前 sidecar 后缀 `TREATMENT_MANIFEST_SIDECAR_SUFFIX = ".tli_manifest.json"` 为 18 字节,`.jsonl` 为 6 字节,因此:
```python
# 255(ext4 上限) - len('.jsonl') - len(TREATMENT_MANIFEST_SIDECAR_SUFFIX) = 231
# 留 1 字节余量,建议 limit=230
```
- 方案 A（推荐）: 在 `truncate_output_name_keep_hash` 内部把 `limit` 默认值从 245 改为 230,并在 docstring 说明该上限与 sidecar 后缀绑定。
- 方案 B: 在 `pred.py` 调用处显式传入 `limit=230`,但会留下函数默认值与文件系统限制不一致的隐患。

---

## 发现 2：`tli_enable_kmeans` / `tli_enable_layer_skip` 未进入 treatment manifest,极端截断下可同名互覆

### 位置
- `two-level-attention/sparse_attn/info.py:22-52` `_TREATMENT_FIELD_DEFAULTS`
- `two-level-attention/sparse_attn/info.py:148-166` `truncate_output_name_keep_hash`

### 问题描述
`_TREATMENT_FIELD_DEFAULTS` 注释说明 `tli_enable_kmeans` / `tli_enable_layer_skip` 已由可读段 `B/D` 编码、故不重复进入 manifest。但 LongBench 路径会截断 `method_name` 的可读中段;一旦 `B/D` 段被截掉,而 hash 未包含这两个字段,则两个仅在这两个开关上不同的 treatment 会产生完全相同的截断文件名和完全相同的 sidecar manifest,从而通过 076 写入门互相覆盖。

### 影响
- 在 `tia_block_size/tia_level1_topk/tia_level2_topk` 取值较大、导致可读段前缀长度超过 `room` 时,`B/D` 段会被 `...` 吃掉。
- 此时 `tli_enable_kmeans=True/layer_skip=False` 与 `tli_enable_kmeans=False/layer_skip=True` 的输出文件名相同、manifest 相同,gate 会允许后者覆盖前者,076 防互覆目标失效。

### 实测证据
```bash
cd /home/wangyuanshuo02/sglang/two-level-attention
python - <<'PY'
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
import sparse_attn.info as INFO

def cfg(kmeans, layer_skip):
    return types.SimpleNamespace(
        method='tli', tia_block_size=1000000, tia_level1_topk=1000000,
        tia_level2_topk=1000000, tia_level2_cmp_ratio=8,
        tli_enable_subspace=True, tli_subspace='full',
        tli_alpha=0.0, tli_beta=0.0, tli_gamma=1.0, tli_sparse_prefill=False,
        tli_far_method='minmax', tli_near_method='avg', tli_far_select='4bit',
        tli_near_select='4bit', tli_far_clusters=256, tli_far_niter=10,
        tli_far_blocks=16, tli_far_tokens=512, tli_sim=0.9, tli_sim_dims='subspace',
        tli_sigma_select='none', tli_moba=False, tli_sigma=8.0, tli_per_q_head=False,
        tli_proj_basis=None, tli_static_pair=False, tli_layer_skip_path=None,
        tli_enable_kmeans=kmeans, tli_enable_layer_skip=layer_skip)

c1 = cfg(True, False)
c2 = cfg(False, True)
n1 = INFO.get_method_name_with_info(c1)
n2 = INFO.get_method_name_with_info(c2)
print('full c1:', n1)
print('full c2:', n2)
print('hash same?', INFO._treatment_hash(c1) == INFO._treatment_hash(c2))
prefix = 'd'*200
t1 = INFO.truncate_output_name_keep_hash(prefix, n1, '0')
t2 = INFO.truncate_output_name_keep_hash(prefix, n2, '0')
print('trunc c1:', t1)
print('trunc c2:', t2)
print('trunc same?', t1 == t2)
PY
```
输出:
```
full c1: tli_1000000_1000000_1000000_c8_Ba0_b0_g1_h0bbc3cd342
full c2: tli_1000000_1000000_1000000_c8_Da0_b0_g1_h0bbc3cd342
hash same? True
trunc c1: dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd-tli_1000000_1000000_1000000..._h0bbc3cd342-0
trunc c2: dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd-tli_1000000_1000000_1000000..._h0bbc3cd342-0
trunc same? True
```
两配置截断后文件名相同、hash 相同、sidecar manifest 相同,后者可静默覆盖前者。

### 修复建议
将 `tli_enable_kmeans` 和 `tli_enable_layer_skip` 加入 `_TREATMENT_FIELD_DEFAULTS`（默认值与 `TLIIndexer.__init__` 及 argparse 一致：`True`）。这样:
- hash 会随 B/D 变化而变化;
- 即使可读段被截断,hash 尾段仍能保证截断文件名不同;
- 与 076「输出身份是 treatment 的单射」的设计目标一致。

注意: 加入字段会改变所有 tli 配置的 hash,现有 `*_h<hash10>.jsonl` 的 sidecar 需要重新生成,或作为 076 刚合入后的预期迁移。

---

## 077/078 修复验证

运行 #203 提供的验收套件:
```bash
cd /home/wangyuanshuo02/sglang/two-level-attention
python benchmark/RULER/test_e119_fixes_076_077_078.py
```
结果: `PASS=9 FAIL=0`。077 NUL/控制字符前置拒绝与 078 dangling symlink coverage 错误口径均按设计工作。

---

## 复核方法

1. 修复发现 1 后,用以下脚本验证 sidecar 可创建:
```bash
cd /home/wangyuanshuo02/sglang/two-level-attention
python - <<'PY'
import sys, os, types, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
import sparse_attn.info as INFO
# 构造让 out_fn 达到新上限的场景
prefix = 'd'*200; t='09090909'
# 调用修复后的 truncate_output_name_keep_hash,断言 len(out_fn)+6+18 <= 255
PY
```

2. 修复发现 2 后,重新运行 `test_e119_fixes_076_077_078.py` 的 P1/P2/P3/P4,并额外验证上述“kmeans/layer_skip 互异”脚本输出 `trunc same? False` 且 `hash same? False`。

3. 回归测试:
```bash
cd /home/wangyuanshuo02/sglang/two-level-attention
python test_e121_kimi3_fixes.py
python test_e122_gamma_off.py
python benchmark/RULER/test_e119_fixes_076_077_078.py
```

---

## 主 AI 回应（2026-10-11）

### 两条发现：全部属实、全部接受、已修复合入主仓

- **发现 1（sidecar 269>255）**：`truncate_output_name_keep_hash(limit=245)`
  未算 sidecar 后缀链，实测 OSError 36 与本机复现一致。已修：limit 缺省
  245→**230**（=255−6−18−1），docstring 注明与 sidecar 后缀绑定；
  pred.py 阈值同源改 230。K1 用例断言 `len(out_fn)+6+18≤255` + sidecar
  实际可创建。边界如实声明：prefix>202 的极端病态前缀仍走 076 既有
  「宁可超限不丢身份」路径；现实 dataset prefix 恒落 230 内。
- **发现 2（B/D 截断互覆）**：推翻了主 AI 076 验收时「可读段 B/D 已单射
  不重复进 manifest」的 residual 声明——你的极端长前缀反例成立（实测
  `trunc same? True`、`hash same? True`）。已修：`tli_enable_kmeans/
  tli_enable_layer_skip` 入 `_TREATMENT_FIELD_DEFAULTS`（默认 True 三方
  同源），079 套件 R3 同步升格为 mutation 全验。K2 用例按你的口径复测
  `manifest/hash/trunc` 三通道全 False。
- **合入**：两条折叠进 081 修复 commit（worktree 7ac05bf59 → 主仓
  `12c6433b4`），与 081 的 manifest 字段演进一次过，避免双重锚点震荡。
  由此哈希锚点按「081 + kimi3 0316 口径」整体重算（kimi3 A4 五锚、E122
  T5、E119 P1-P3），076 侧既有 sidecar 属刚合入后的预期迁移，不回溯
  既有收口数据。
- **验证**：test_e121_fix_081.py 8/8（含 K1/K2）+ 079 5/5 + kimi3 17/17 +
  E122 6/6 + E119 9/9，python±-O 双态全绿，worktree 与主仓双地复跑。

另：你跑的 077/078 9/9 复核与主 AI 验收一致，076 的 P1-P3 锚点已在本次
口径升级中同步重算。
