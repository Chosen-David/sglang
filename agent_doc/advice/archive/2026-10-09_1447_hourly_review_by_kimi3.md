# 2026-10-09 14:47 kimi3 小时审查报告

## 审查区间

- **基准 SHA**：`8b9fcd0bc6d87ebd17643c67aefc289195a86e87`
- **远端 SHA**：`b48ecfca650344c9051365c12211117c3032cbcc`
- **新提交**（`git log --oneline`）：
  - `b48ecfca6 docs: audit E116g publish and E119 gates`
  - `96d51ce10 fix: 发布成功后报告型 I/O 错误误触发破坏性回滚（kimi3 1404）`
  - `841e2e1a5 docs: E116g 主 AI 回应 GPT 1326 审计 040/041（发布锁 + E119 跨臂身份门禁落地）`
  - `d8cd5b680 E116g: GPT 1326 审计 TL-RULER-CONCURRENT-PUBLISH-040 / TL-RULER-CROSSARM-IDENTITY-041 两项修复`
  - `ac3b30e2f docs: respond to single-pool vs L1-bypass clarification (S-T009)`
- **工作树**：`git status --porcelain` 中仅有 `agent_doc/`、`paper/`、`research/docs/` 等文档/数据产物改动，以及 `exp/results`、`ref/figs` 等数据产物未跟踪文件；**无 `python/sglang/srt/layers/attention/tli/` 或 `two-level-attention/sparse_attn/` 等核心代码的未提交改动**，故未纳入本轮代码审查对象。

## 变更文件与重点

本轮变更集中在 RULER 正式打分入口及其回归测试：

- `two-level-attention/benchmark/RULER/score_ruler_formal.py`（E116g：040 并发发布锁 + 1404 报告型错误不回滚）
- `two-level-attention/benchmark/RULER/test_e116f_publish_atomic.py`
- `two-level-attention/exp/trace/analyze_e119_ruler64k_formal.py`（E119 跨臂身份门禁汇总）
- `two-level-attention/exp/trace/test_e119_crossarm_identity.py`

## 回归测试结果

已在本机执行相关测试，全部通过：

```text
E116f ALL PASS (8/8)
E119 crossarm identity ALL PASS (8/8)
```

## Bug 发现

### B1：备份清理失败会导致已成功的发布被回滚

**位置**：`two-level-attention/benchmark/RULER/score_ruler_formal.py:245-255`（`_locked_publish`）

**问题描述**：

在 `_locked_publish` 中，四镜像安装完成并通过 SHA 终验后，代码先执行备份文件删除（`os.remove(bak)`），**然后**才设置 `rollback_state["committed"] = True`：

```python
# ---- 全部镜像安装 + 终验成功 → 提交完成，锁内清理备份 ----
for dst, bak in backups.items():
    if bak is not None:
        try:
            os.remove(bak)
        except FileNotFoundError:
            pass
# ---- 事务已提交（kimi3 1404）：置 committed 标记 ...
rollback_state["committed"] = True
```

如果备份清理阶段因权限不足、外部并发清理、文件系统瞬态错误等原因抛出 `OSError`，`_locked_publish` 会在未置 `committed` 的情况下异常退出。外层 `main()` 的 `except (SystemExit, OSError)` 看到 `committed=False`，便会把**已经发布成功的公开镜像**回滚到旧代际（或首轮场景下删除新镜像），并生成 failure receipt。这实质上破坏了“发布成功后不得回滚”的语义，与 1404 修复目标相矛盾。

**实测证据**：

使用仓库内 `benchmark/RULER/testdata/e116e` 构造旧产物后，通过 monkeypatch 让 `os.remove` 对 `.bak-*` 文件抛出 `PermissionError`，复现结果如下：

```text
old out sha      d84fd21c588e5f06
new out sha (reference) c8b5e8f5b544d675
injected run rc= 1
current out sha  d84fd21c588e5f06
BUG REPRODUCED: successful publish was rolled back because backup cleanup failed; final out == old generation
```

即：新代际本已成功安装（reference run SHA = `c8b5e8f5...`），但因备份清理被注入失败，最终公开产物回退到旧代际 SHA（`d84fd21c...`），并留下 failure receipt。

复现脚本核心片段（可直接执行）：

```python
import os, sys, subprocess, tempfile, shutil, json
REPO = "/home/wangyuanshuo02/sglang/two-level-attention"
TESTDATA = os.path.join(REPO, "benchmark", "RULER", "testdata", "e116e")

base = tempfile.mkdtemp(prefix="e116g_backup_cleanup_")
root = shutil.copytree(os.path.join(TESTDATA, "pred_root"),
                       os.path.join(base, "root"))
out = os.path.join(base, "out.json")

# 第一轮：生成旧产物
subprocess.run([sys.executable, "-m", "benchmark.RULER.score_ruler_formal",
                "--root", root, "--pred-postfix", "_fx",
                "--data-root", os.path.join(TESTDATA, "data_root"),
                "--out", out, "--min-samples", "2"],
               cwd=REPO, env={**os.environ, "PYTHONPATH": REPO})

# 篡改 pred 使新代际可辨
tgt = os.path.join(root, "L32768", "pred_fx", "vt-fxm-01010000.jsonl")
rows = [json.loads(l) for l in open(tgt, encoding="utf-8")]
rows[0]["pred"] = "tampered pred (CLEANUP)"
with open(tgt, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 注入 backup 删除失败
argv = ["--root", root, "--pred-postfix", "_fx",
        "--data-root", os.path.join(TESTDATA, "data_root"),
        "--out", out, "--min-samples", "2"]
code = (
    "import os, sys\n"
    "sys.argv = ['score_ruler_formal'] + " + repr(argv) + "\n"
    "_orig = os.remove\n"
    "def _remove(p):\n"
    "    if '.bak-' in p:\n"
    "        raise PermissionError('INJECTED remove failure on backup ' + p)\n"
    "    return _orig(p)\n"
    "os.remove = _remove\n"
    "import benchmark.RULER.score_ruler_formal as m\n"
    "m.main()\n"
)
r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": REPO},
                   capture_output=True, text=True)
print(r.returncode, r.stderr[-800:])
# 观察：out 会回退到旧代际 SHA
```

**影响**：

在并发或自动化环境中，备份文件可能被外部清理进程删除、或当前进程缺少删除权限。此时已验证通过并成功发布的正确结果会被静默回滚，造成“旧代际残留”或“成功发布丢失”，属于 040/1404 发布语义缺陷的剩余窗口。

**修复建议**：

把 `rollback_state["committed"] = True` 上移到 SHA 终验成功之后、备份清理之前；备份清理改为 best-effort（捕获所有 `OSError` 而非仅 `FileNotFoundError`），失败不回滚：

```python
# ---- 全部镜像安装 + 终验成功 → 事务已提交 ----
rollback_state["committed"] = True
# ---- 提交后的清理：备份删除失败不得破坏已发布产物 ----
for dst, bak in backups.items():
    if bak is not None:
        try:
            os.remove(bak)
        except OSError:
            pass
```

这样外层 `except` 会把 backup 清理失败当作 post-publish 报告型错误处理（参考 1404）：产物保留、generation 保留、rc=0，failure receipt 记录 `publish_committed=True`。

**复核方法**：

1. 运行上述复现脚本，确认 bug 存在。
2. 修复后重新运行复现脚本，确认最终 `out` 保持新代际 SHA，且 `rc=0`。
3. 继续运行 `python3 -m benchmark.RULER.test_e116f_publish_atomic` 与 `python3 exp/trace/test_e119_crossarm_identity.py`，确认 038/039/040/041 回归测试仍全过。

## 性能/优化点

本轮变更集中在打分、发布、汇总脚本，**未触及 GPU 热路径**；未发现以下类型问题：

- 明显的重复计算或可缓存的中间结果
- `.item()` / `.tolist()` / `.cpu()` 等 GPU 同步调用
- 可批量化的逐请求循环
- kernel 旁路条件退化

因此本轮**无性能优化建议**。

## 纪律声明

- 未修改任何业务代码；未执行 `git commit` / `git push`。
- 已确认旧问题 C1（eager `-inf` top-k）、C2（taskmd graph 短行重复）、C3（near 簇分缺 scale）相关代码在本轮区间内未改动，故不重复报告。
- 报告仅包含经实测证据确认的 bug，无占位或推测性条目。
