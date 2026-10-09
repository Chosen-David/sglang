# TwoLevel E119 汇总发布原子性补充审计（`ebe965a34`）

## 审查范围与版本

- 审查分支：`two-level-indexer`
- 审查提交：`ebe965a34bcdde795347852118153797a96b6494`
- 相对上一份审计提交 `b48ecfca6` 的变化只有
  `agent_doc/advice/2026-10-09_1429_twolevel_e116g_e119_followup_audit_96d51ce.md`
  追加主 AI 回应；没有实现或测试代码变化。
- 目标文件：`two-level-attention/exp/trace/analyze_e119_ruler64k_formal.py`
  （SHA256 `de9d348245c736f6f724be8c01b23417821ad574363df4e3a43691ff22f05ad8`）。
- 本轮只做 CPU 控制流故障注入，没有使用 GPU、没有重跑模型、没有修改实现、
  实验数据或既有结果。

## 新发现

| ID | 状态 | 严重度 | 位置 | 结论 |
|---|---|---:|---|---|
| `TL-E119-SUMMARY-ATOMICITY-049` | **confirmed / 隔离故障注入** | P1（收口产物可恢复性） | `analyze_e119_ruler64k_formal.py:321-322` | 汇总器以 `open(p_out, "w")` 直接截断旧 summary，再调用 `json.dump`。进程终止、磁盘写满或写入异常会毁掉上一份已通过门禁的 summary，留下截断/非法 JSON；当前没有同目录临时文件、`fsync`、校验后 `os.replace` 或 summary 发布锁。 |

### 违反的契约

该脚本把 `e119_ruler64k_formal_summary.json` 作为三臂正式收口产物，并在注释、
测试语义及既有审计回应中要求失败运行不得覆盖旧 summary。当前实现仅保证
**门禁在写入前失败**时旧文件不变；门禁全部通过后，最终输出阶段一旦失败，
旧的 last-known-good summary 已先被 `"w"` 截断，无法恢复。

预期行为是：新 summary 完整序列化、落盘、同步并重新解析/校验成功以前，公开
路径始终指向上一份完整有效文件；任意写入异常只遗留未提交临时文件或被清理，
不破坏旧 summary。

### 触发条件与实际行为

触发条件：三臂读取门禁全部通过，执行到 `line 322` 后，在 JSON 完整写完并关闭
之前发生 `OSError`、磁盘写满、进程被杀或机器掉电。直接写固定公开路径的并发
执行也可能暴露同类半写窗口。

本轮在临时目录复制三臂 result/manifest/receipt/scorer manifest；由于仓库未提交
production prediction 与 derived prediction（这是已记录的 048 fixture 缺口），
仅用 receipt 内已记录的对应 SHA 虚拟满足这些只读文件检查，使执行可靠到达唯一的
summary 写入点。随后把 `json.dump` 故障注入为“先写入 `{"partial":` 并 flush，
再抛 `OSError`”。结果：

```text
rc=1
error=injected-write-interruption
before_sha256=5bcd09ab80635fc08e9ac5bead154f19a27ef2dfbddbf30d8ef016a6bfae229f
after_sha256=875687fdaf12961b993bd559158bb56a6def15bc74d6b805ea88303818de659a
after_bytes=b'{"partial":'
after_valid_json=False
```

这里的预测文件存在性/SHA 虚拟化仅用于绕过已知 048 缺失 fixture，未篡改 result、
manifest、receipt、scorer manifest 或汇总算法；故障直接作用于源码 `line 322` 的
公开 summary 写入。反例原始输出如上，输出文本 SHA256 可由所列字段确定；未把临时
目录或生产数据提交仓库。

不依赖生产文件的最小可运行 primitive 复现（与 `line 322` 的打开模式和故障窗口
相同）：

```bash
python3 - <<'PY'
import json, os, tempfile
p = os.path.join(tempfile.mkdtemp(), "e119_ruler64k_formal_summary.json")
old = b'{"sentinel":"previous-valid-summary"}\n'
open(p, "wb").write(old)
try:
    with open(p, "w") as f:       # 源码 line 322 的直接截断写
        f.write('{"partial":')
        f.flush()
        raise OSError("injected-write-interruption")
except OSError:
    pass
after = open(p, "rb").read()
print(after)
assert after != old
try:
    json.loads(after)
    raise AssertionError("截断文件不应是合法 JSON")
except json.JSONDecodeError:
    pass
PY
```

### 对现有数据与结论的影响

- 当前仓库内 `e119_ruler64k_formal_summary.json` 是有效 JSON，SHA256 为
  `ccd6d81ee1f2e90e9d77f8545f0de178fddcc5fe4c118f1e31292a3488a8aa4f`；没有证据
  表明这次历史写入实际中断。
- 因此本发现不撤销现有 64K 数值 `mavg 49.42 > FullKV 48.54 > aavg 47.51`，也不
  证明评分数据被污染。
- 风险落在后续重汇总和 128K 收口：失败运行可能同时没有新结论、又删除最后一份
  可用旧结论；消费者若只检查路径存在，甚至可能读到截断文件。

## 建议修复与最小验收

1. 在 `results_dir` 同一文件系统创建唯一临时文件，完整 `json.dump` 后
   `flush + fsync`，关闭后重新解析并验证必要字段；全部通过才用 `os.replace` 原子
   替换公开 summary。异常路径清理本轮临时文件，旧 summary 字节必须不变；如需
   抵抗目录项掉电丢失，再同步父目录。
2. 给 summary 发布使用专用锁，或把它纳入版本化 generation + 单指针提交协议；
   至少保证两个汇总器并发时不会出现半文件。若采用 last-writer-wins，summary 中
   必须保留三臂不可变输入引用，使后写者的输入代际可审计。
3. 新增两个隔离负例：
   - `json.dump` 写出前缀后抛 `OSError`；断言 exit 非零、旧 summary SHA 不变、公开
     文件仍可解析。
   - 两个进程同时向同一 summary 发布不同但各自闭合的 fixture；断言最终文件只能是
     某一完整代际，不得混写/截断，并核对其输入引用。
4. 修复 048 后，把这些用例纳入干净检出可运行的 E119 fixture 套件；普通解释器与
   `python -O` 都应执行。原子写修复不能替代 045-047 的显式门禁和口径闭包。

## 旧发现状态与未覆盖项

- `042-048` 已在 `ebe965a34` 的主 AI 回应中全部接受并派发 E116h，但当前实现提交
  尚未出现，故状态仍是“accepted / not implemented”，不能称约束已经生效。
- 本轮实际运行 `test_e119_crossarm_identity.py`，在干净检出仍于 P1 正例因未提交
  production prediction 失败（已知 048），没有把该失败重复计为新 bug。
- GPU two-level attention、near/far 与 L1/L2 选择、kernel/e2e 精度、128K 实际结果
  本轮未验证；049 只覆盖 E119 最终 summary 的发布/恢复边界。

## 下一检查点

等待 E116h 实现提交后，先按 042-049 逐项映射到源码与测试，重点复跑完整写集锁、
备份清理状态机、普通/`python -O` 消费门禁、可移植 fixture 和 summary 写入故障；
只有旧版失败、修复版通过且远端内容一致，才把对应发现标为 fixed/rechecked。
