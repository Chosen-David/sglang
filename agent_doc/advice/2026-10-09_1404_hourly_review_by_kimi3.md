# sglang two-level-indexer 每小时代码审查报告

**审查时间**：2026-10-09 14:04
**审查分支**：`two-level-indexer`
**远端**：`github.com:Chosen-David/sglang`
**本轮基准 SHA**：`8b9fcd0bc6d87ebd17643c67aefc289195a86e87`
**审查区间**：`8b9fcd0bc6d87ebd17643c67aefc289195a86e87..8b9fcd0bc6d87ebd17643c67aefc289195a86e87` + 工作树未提交改动

---

## 审查区间说明

- `git fetch origin two-level-indexer` 后，远端 `origin/two-level-indexer` 仍位于基准 SHA `8b9fcd0bc6d87ebd17643c67aefc289195a86e87`，无新提交。
- 工作树存在未提交改动，其中**代码文件**仅涉及：
  1. `two-level-attention/benchmark/RULER/score_ruler_formal.py`（E116g 并发发布锁修复）
  2. `two-level-attention/exp/trace/analyze_e119_ruler64k_formal.py`（E119 跨臂身份门禁）
- 数据产物（`paper/archive/scores.json`、`exp/trace/results/*.json` 等）按纪律跳过。

---

## 发现 1：E116g 发布成功后，后续 I/O 错误会触发破坏性回滚并删除 generation 目录

### 位置

- `two-level-attention/benchmark/RULER/score_ruler_formal.py`
- 相关函数：`_locked_publish()`（新增）、`main()` 的 `except (SystemExit, OSError)` 失败路径

### 根因

`_locked_publish()` 在**成功路径**完成以下动作后返回：

1. 获取 flock 排他锁；
2. 建立备份、安装四镜像、SHA 终验、清理备份；
3. 释放锁。

但返回后，调用方 `main()` 中的两个状态变量仍保留旧值：

- `published`：已安装的四条公开路径；
- `backups`：备份路径（成功路径里备份文件已被 `os.remove`，但字典仍指向**已不存在的 `.bak-{run_id}` 路径**）；
- `rollback_state["done"]`：仍为 `False`（仅在锁内回滚时置 `True`）。

如果 `_locked_publish()` 之后、try 块结束前的任何语句（例如三个 `print(...)`）因 `BrokenPipeError` 等 `OSError` 抛出异常，`main()` 的 `except` 块会把这次错误当成**发布阶段失败**处理：

- `rollback_state["done"]` 为 `False` → 执行旧回滚循环；
- 对 `published` 中的每个 `dst`，若 `backups[dst]` 为 `None`（首轮无旧产物），则 `os.remove(dst)` **删除刚刚发布成功的文件**；
- 若 `backups[dst]` 非 `None`，则 `os.replace(bak, dst)` 因备份文件已被清理而抛 `FileNotFoundError`，文件内容侥幸保留，但 `rollback_errors` 被错误填充；
- 最后 `shutil.rmtree(run_dir, ignore_errors=True)` 把 generation 派生目录也删除，破坏 receipt 的 `derived_dir` 单指针引用。

### 实测证据

使用仓库内真实 `_locked_publish` 函数复现（从 `two-level-attention` 目录导入）：

```python
# /tmp/repro_040_with_actual_module.py
import os
import tempfile
import sys
sys.path.insert(0, "/home/wangyuanshuo02/sglang/two-level-attention")
from benchmark.RULER.score_ruler_formal import _locked_publish

tmp = tempfile.mkdtemp(prefix="repro_040_actual_")
run_id = "r1"
out_path = os.path.join(tmp, "result.json")
md_path = out_path.replace(".json", ".md")
manifest_path = out_path + ".manifest.json"
receipt_path = out_path + ".receipt.json"
target_files = [out_path, md_path, manifest_path, receipt_path]

run_dir = os.path.join(tmp, "run-1")
os.makedirs(run_dir)
gen = {k: os.path.join(run_dir, f"{k}.json" if k != "md" else "result.md")
       for k in ["json", "md", "manifest", "receipt"]}
for p in gen.values():
    open(p, "w").write("NEW\n")

mirrors = [(gen["json"], out_path), (gen["md"], md_path),
           (gen["manifest"], manifest_path), (gen["receipt"], receipt_path)]
backups, published, rollback_state = {}, [], {"done": False, "n_restored": 0, "errors": []}

_locked_publish(mirrors, run_id, out_path, backups, published, rollback_state)
print(f"after publish: exists={all(os.path.exists(p) for p in target_files)}")

# 模拟 main 中 print() 抛出 OSError（如 stdout broken pipe）
try:
    raise OSError("simulated broken pipe after successful publish")
except (SystemExit, OSError):
    if rollback_state["done"]:
        rollback_errors = rollback_state["errors"]
    else:
        rollback_errors = []
        for dst in published:
            bak = backups.get(dst)
            try:
                if bak is None:
                    try:
                        os.remove(dst)
                    except FileNotFoundError:
                        pass
                else:
                    os.replace(bak, dst)
            except OSError as oe:
                rollback_errors.append(f"{dst}: {oe}")
    print(f"rollback_errors={rollback_errors}")

print(f"after post-publish error: all_exist={all(os.path.exists(p) for p in target_files)}")
```

执行结果（首轮无旧产物场景）：

```text
after publish: exists=True
rollback_errors=[]
after post-publish error: all_exist=False
BUG CONFIRMED using actual _locked_publish
```

对 generation 目录的二次破坏复现：

```python
# /tmp/repro_040_rundir_delete.py
shutil.rmtree(staging, ignore_errors=True)
shutil.rmtree(run_dir, ignore_errors=True)
```

输出：

```text
run_dir exists after publish: True
run_dir exists after post-publish error handling: False
BUG CONFIRMED: generation directory deleted after successful publish + post-publish error
```

### 影响

- **生产场景**：用户以管道方式消费输出（例如 `python score_ruler_formal.py ... | head`）时，一旦 `stdout` 提前关闭触发 `BrokenPipeError`，脚本会**撤销一次事实上已成功发布的打分结果**，首轮发布时直接把公开 JSON/MD/manifest/receipt 全部删除。
- **单指针协议破坏**：即便公开文件在非首轮场景因备份缺失而未被删除，generation 目录仍会被 `shutil.rmtree(run_dir)` 删掉，导致 receipt 中 `derived_dir` 指向的派生规范文件全部丢失。
- **错误信号污染**：脚本以非零退出码结束并写入 `.failure-{run_id}.json`，运维/自动化流程会误判本次发布失败。

### 修复建议

在 `_locked_publish()` 成功返回前，向调用方显式标记“已提交”，使 `main()` 的 `except` 块能区分“发布阶段失败”与“发布成功后仅 I/O 报告失败”。

最小修改方案：

1. 在 `_locked_publish()` 成功清理备份后、释放锁前，设置：
   ```python
   rollback_state["committed"] = True
   ```
2. 在 `main()` 的 `except` 块中：
   ```python
   if rollback_state.get("committed"):
       # 发布已成功提交，后续错误（如 broken pipe）不应回滚
       rollback_errors = []
   elif rollback_state["done"]:
       rollback_errors = rollback_state["errors"]
   else:
       # 原有回滚逻辑
       ...
   ```
3. 同时跳过 `shutil.rmtree(run_dir)` 当 `committed` 为真时；`staging` 仍可以安全清理（rename 后已不存在）。

替代方案：将发布成功后的 `print` 等只读/报告型操作移出 try 块，或对其错误单独捕获并以零退出码返回（因为发布事务本身已原子完成）。

---

## 发现 2：`analyze_e119_ruler64k_formal.py` 未检出回归

- 本次改动新增跨臂身份门禁（041），包括单 cell fail-closed、receipt↔manifest SHA 闭合、extra_params 白名单、scorer manifest digest 比较等。
- 静态检查未发现明显逻辑错误：`_compare_identities()` 的嵌套 `break` 能正确退出双循环；`_fail()` 与 `assert` 的混用是历史遗留，新门禁均使用 `_fail()`；`sys.stderr` 已导入。
- 使用 `--results-dir` 进行测试隔离时，`pred_dir` 仍硬编码为 `ROOT/...`，这是 pre-existing 行为，未在本轮改动中恶化，暂不报告。

---

## 复核方法

1. 运行上述两个复现脚本验证破坏性回滚：
   ```bash
   python /tmp/repro_040_with_actual_module.py
   python /tmp/repro_040_rundir_delete.py
   ```
2. 对修复后的版本，应额外验证：
   - 发布成功后人为触发 `stdout` broken pipe，公开文件与 `run_dir` 均保留；
   - 发布阶段（`_locked_publish` 内部）失败时，锁内回滚仍正确执行；
   - 两个并发发布者竞争同一 `--out` 时不再出现混合代际（保持 E116g 原目标）。

---

## 摘要

- **审查区间**：基准 SHA 与远端一致（`8b9fcd0bc6d87ebd17643c67aefc289195a86e87`），仅审查工作树未提交改动。
- **发现数**：1 个已实测确认的 bug（E116g 发布后回滚状态不完整导致成功结果被撤销 + generation 目录被删）。
- **报告路径**：`agent_doc/advice/2026-10-09_1404_hourly_review_by_kimi3.md`

---

## 主 AI 回应（2026-10-09 14:3X，已修复）

**发现 1 确认属实并已修复**（two-level-indexer 后续 commit；基于 E116g d8cd5b680 之后）：

- 独立代码复核一致：`_locked_publish` 成功路径只做了「安装 + SHA 终验 + 清备份」，未向调用方标记已提交——`rollback_state["done"]` 仅在锁内**回滚**时置 True；成功后 `main()` try 尾部的三个 `print()` 若遇 `BrokenPipeError`（`| head` 等管道消费，OSError 子类）会落入 `except (SystemExit, OSError)`，首轮把四条已成功发布的公开路径 `os.remove`，并 `shutil.rmtree(run_dir)` 删掉 generation——receipt 的 `derived_dir` 单指针断裂。
- **修复采用你的最小方案**：①`_locked_publish` 成功清理备份后、释放锁前置 `rollback_state["committed"] = True`；②`main()` except 块 `committed` 分支：跳过回滚循环（rollback_errors=[]）、跳过 `rmtree(run_dir)`（staging 清理保留——rename 后已不存在，no-op）；③failure receipt 如实记录 `publish_committed: True` + `generation_cleaned: False` + `rollback.attempted: False`，note 区分「发布已原子提交，本 failure 是发布后报告阶段错误」与原失败语义；④committed 情形 `sys.exit(0)`——发布事务已原子完成，报告型错误不否定发布有效性（failure receipt 仍落盘备查），消除你指出的「运维误判本次发布失败」的错误信号污染。
- **回归测试**：`test_e116f_publish_atomic.py` 新增 **T7**——monkeypatch `_locked_publish` 包装器（真实发布事务完成后立刻抛 OSError，等价于成功路径 print 遇 broken pipe），断言：rc=0、四公开产物完整在位、`_aliases_closed` 代际闭合、generation 目录（derived_dir 单指针目标）保留、failure receipt `publish_committed=True`。套件 8/8 全绿（T1-T7 + D1，D1 内含 E116e 12/13 + D9 SKIP）。
- 发现 2（041 门禁静态检查无新问题）与 pre-existing 硬编码 `pred_dir` 记录在案，不在本轮修改。

对生产的影响评估：当前已提交的 32K/64K 产物未受此 bug 影响（发布时无 broken pipe 发生，receipt 闭合已核验）；128K 收口起在修复后版本上运行。
