# TwoLevel 增量缺陷审查：E113 并发终态与 E119 128K YaRN 身份

- 审查代码：`9fe120f7672459194727ff7559640417b7a8c7a5`
- 上次已审查源码基线：`38977eb`（对应 advice 提交 `9a6f269`）
- 本轮相关增量：`695b007ef`（E113 054/055）、`00e230ebc`（E116j/E117/E119）、`d3aa46a29`（E190 128K 来源佐证）；`a4d14ef`、`9fe120f` 为 advice-only，不作为源码变化。
- 审查范围：E113 attempt 状态机与纳秒计时；RULER formal 发布/消费绑定；E117/E119 汇总门禁；128K 三臂来源与运行身份。
- 环境边界：干净独立检出；Python 3；无 `torch`、无已授权 GPU，因此没有执行 CUDA/Triton kernel 或真实 e2e。以下并发复现只调用生产脚本的标准库发布函数，不冒充 GPU 实测。

## 结论

| ID | 状态 | 严重度 | 结论 | 对已有数据的影响 |
|---|---|---:|---|---|
| `TL-E113-ATTEMPT-RACE-056` | **confirmed（确定性交错 + 生产发布函数）** | P1，结果终态/证据完整性 | 新 attempt-v3 没有以输出路径为粒度的跨进程锁。两个同 `--out` attempt 在发布前都可完成隔离检查，随后分别发布 success 与 failure，最终三个公开路径同时存在，违反“可见终态唯一且属于本 attempt”的协议。 | 本复现不证明既有 E113 归档已发生竞态；但同输出路径的并发/重试一旦可达，消费者无法从公开文件判定唯一终态，后续 speedup 证据不可安全接收。 |
| `TL-E119-YARN-IDENTITY-057` | **confirmed（身份记录冲突）；实际 effective factor 未闭合** | P2，实验身份/可复现性 | 128K 生成入口在未显式传 `--yarn_factor` 时按当前代码应解析为 `4.0`，三份正式 manifest/receipt 却记录操作者声明 `2.0`。manifest 不是实际生效配置的来源闭包；由于没有生成时原生 effective-config receipt，不能仅凭当前源码把历史运行的真实值升级为闭合事实。 | E190 启动链强支持三臂走相同的 4.0 路径，因此现有相对分数与排序没有因此被推翻；但不得再把结果标成已证实的 factor=2.0 或 factor=4.0，也不能按 manifest 精确复现或支持依赖 factor=2.0 的配置结论。 |

## 1. `TL-E113-ATTEMPT-RACE-056`

### 违反的契约与位置

`two-level-attention/exp/trace/e113_microbench.py` 在 `9fe120f`：

- `399-405`：attempt 开始时调用 `quarantine_prior_artifacts(args.out)`；没有输出路径锁。
- `504-516`：失败分支先原子写 `<out>.failure.json`，发现 success/sidecar 后仅 `_fail` 退出，并不撤销刚写 failure 或竞争者 success。
- `523-529`：success JSON 与 `.sha256` 是两个独立的原子替换。
- `532-545`：成功分支删除 failure、读回 attempt 并终检，但整个序列仍无锁。
- 对 E113 相关文件检索 `flock|fcntl|lockf|FileLock|portalocker|O_EXCL|.lock` 无命中。

因此每次 attempt 自己的串行终检不能建立多进程线性化点。触发条件是两个进程使用同一 `--out`，且都在任一方发布前完成 `quarantine_prior_artifacts`。调度器重试、重复派单或人工并发启动均可能满足该条件。

### 最小可运行复现

复现从当前 SHA 动态导入 `e113_microbench.py`，用空 `torch` 模块绕过导入（不调用任何 torch/GPU 代码），并调用生产 `quarantine_prior_artifacts`、`atomic_write_bytes` 与 `_fail`：

```python
import hashlib, importlib.util, json, os, sys, tempfile, types

sys.modules["torch"] = types.ModuleType("torch")
spec = importlib.util.spec_from_file_location("e113", "two-level-attention/exp/trace/e113_microbench.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "result.json")
    assert m.quarantine_prior_artifacts(out) == []  # attempt A
    assert m.quarantine_prior_artifacts(out) == []  # attempt B

    success = {"meta": {"attempt": {"attempt_id": "B"}}, "cases": []}
    raw = json.dumps(success, indent=1).encode()
    m.atomic_write_bytes(out, raw)
    m.atomic_write_bytes(out + ".sha256", (hashlib.sha256(raw).hexdigest() + "\n").encode())

    failure = {"status": "failed", "attempt": {"attempt_id": "A"}}
    m.atomic_write_bytes(out + ".failure.json", json.dumps(failure).encode())
    try:
        if os.path.exists(out) or os.path.exists(out + ".sha256"):
            m._fail("fail-closed 后可见路径仍残留成功 JSON/sidecar（状态机失效）")
    except SystemExit as exc:
        print(exc.code, os.path.exists(out), os.path.exists(out + ".sha256"),
              os.path.exists(out + ".failure.json"))
```

原始复现脚本 SHA256：`54757db4496e5dc369ec953e20f482cf3ef8d043ec5d19d234d5d7b148b0eff2`；原始日志 SHA256：`5b6b4ccab459758613cc952a7373da10a0cfdf4ed16d50775db6a5bd3c0c840f`。关键输出：

```text
[STATE-FAIL] fail-closed 后可见路径仍残留成功 JSON/sidecar（状态机失效）
a_exit_code=3
visible={success_json: true, sidecar: true, failure_json: true}
success_attempt=B, failure_attempt=A
```

### 建议修复与重测

以规范化后的输出父目录+basename 派生稳定锁文件，在“隔离旧终态 → 执行 → 发布单一终态 → 终检”完整生命周期持有跨进程锁；若不希望长时间持锁，则改为每 attempt 不可变 generation 目录，最后通过一个带 attempt/hash 的原子 pointer/receipt 仲裁唯一公开代际。只在发布阶段加锁仍需明确 loser 的结果保存和退出语义，不能让 loser 删除 winner 的终态。

新增真实双进程测试，不使用同进程顺序模拟：覆盖 success/success、success/failure、failure/success、failure/failure，以及在 JSON 与 sidecar 两次替换之间故障注入；普通 Python 与 `python -O` 均要求公开入口最多一个有效终态，success JSON/sidecar/attempt 必须同代际。完成后再运行真实 GPU E113 correctness+timing，确认锁/代际协议未污染测量区间。

## 2. `TL-E119-YARN-IDENTITY-057`

### 违反的契约与位置

`two-level-attention/benchmark/RULER/pred_ruler.py:52-53,75,103-107,123-130` 定义 `YARN_FACTOR_AUTO[131072] = 4.0`，且 `--yarn_factor` 默认为 `None`；所以只传 `--yarn` 的 128K 运行会生效 `4.0`。

`two-level-attention/benchmark/RULER/score_ruler_formal.py:549-552,623-635` 则把事后传入的 `--yarn-factor` 原样写入 `run_identity`，并明确 legacy 值是操作者声明、不是从预测产物恢复。下列三份正式 128K manifest（以及对应 receipt/generation manifest）均声明 `2.0`：

- `two-level-attention/exp/trace/results/e119_ruler128k_formal_mavg.json.manifest.json:13`
- `two-level-attention/exp/trace/results/e119_ruler128k_formal_aavg.json.manifest.json:13`
- `two-level-attention/exp/trace/results/e119_ruler128k_formal_fullkv.json.manifest.json:13`

新 E190 佐证 `two-level-attention/exp/trace/results/e119_128k_arm_provenance_corroboration.md:69` 记录派单只传 `--yarn`，并得出按可用代码链推导“实际应为 `4.0`”。这使“manifest 声明 2.0”与“启动链/代码推导 4.0”形成可证伪冲突，而不是仅文档措辞问题；但 E190 同时承认没有预测字节到进程/源码的密码学绑定，故历史运行的 effective factor 仍未闭合。

### 最小核验与输出

AST 读取当前 `YARN_FACTOR_AUTO` 并解析三份 manifest 的结果为：

```json
{
  "manifest_declared": {"mavg": 2.0, "aavg": 2.0, "fullkv": 2.0},
  "pred_ruler_default_for_131072": 4.0,
  "dispatch_recorded_explicit_yarn_factor": false
}
```

该核验与上一节复现共用脚本/日志 hash。它确认正式身份记录与可用生成链冲突，但不单独证明历史进程确实加载了哪个 effective factor，也不证明三臂间存在不同 factor；E190 证据强支持三臂走相同的 4.0 路径。因此保留原始分数，只收窄配置解释，并把精确 effective factor 标为未闭合。

### 建议修复与重测

1. `pred_ruler.py` 在解析自动档位之后，把 **effective** YaRN factor、完整 `rope_scaling`、context length、模型/config hash 与生成参数写入每个原生预测产物或同代 receipt；formal 阶段从该生产者证据消费，禁止用事后 CLI 声明覆盖实际值。
2. 对本批 legacy 128K 产物做版本化 provenance correction：保留原始预测/score 字节，把旧 `operator_declared=2.0` 降级为历史声明，并记录“启动链/当前代码强支持 4.0、生产者原生证据缺失”。除非取得同代进程配置/日志闭包，否则不要直接补写 `actual_effective_yarn_factor=4.0`；需要精确身份时按新协议重跑。
3. 用小型可运行配置做自动档位与显式覆盖两组测试，验证 `--yarn` 在 64K/128K 分别持久化 2.0/4.0，`--yarn_factor X` 持久化 X；篡改 formal CLI 声明必须 fail closed。之后若论文或报告曾写“128K factor=2.0”，应先撤销该精确口径；若必须区分 factor2/factor4，须按生产者原生身份协议重新生成预测与评分。

## 旧发现复查与实际执行

| 项目 | 命令/证据 | 结果 |
|---|---|---|
| E117 resolved output binding | `python exp/trace/test_e117a_out_binding.py`；同命令 `python -O` | 3/3 PASS（两种模式） |
| E119 v3 跨臂消费者 | 在 `two-level-attention/` 下运行 `python exp/trace/test_e119_crossarm_identity.py 128k`；同命令 `python -O` | 20/20 PASS（两种模式）；052/053 的 generation/receipt 与 resolved-out 修复通过，051 仍按设计保持 provisional。 |
| E116e formal gate | `PYTHONPATH=$PWD python -m benchmark.RULER.test_e116e_gate`；同命令 `python -O` | 12/13 PASS，D9 SKIP（真实数据路径不可用）；无 `assert` 优化绕过回归。 |
| 受影响 Python 文件语法 | `python -m py_compile ...` | PASS |
| E113 054/055 官方回归 | `python exp/trace/test_e113_state_machine_054_055.py`；同命令 `python -O` | **未执行到用例**：环境缺 `torch`，导入即 `ModuleNotFoundError`；不能据此判失败或通过。串行 054/055 改动静态上符合原修复目标，但未覆盖本报告的跨进程竞态。 |

## 未覆盖边界与下一检查点

- 未运行 GPU kernel、Triton correctness、E113 真实延迟或 RULER 真实生成；没有声明性能变化。
- 未取得调度器的同 `--out` 并发历史，因此 056 对既有归档的实际污染为未知；在修复前应将同路径并发视为不安全，而不是认定既有数据已坏。
- 128K 三臂 treatment 归因仍按 E190 保持 provisional；本报告不把派单/日志佐证升级成对每个预测字节的密码学因果证明。
- 下次仅在相关代码、接口、实验脚本或结果聚合逻辑再次变化时复查；优先验证 056 的真实跨进程锁/代际测试，以及 057 的生产者原生 effective-config receipt。

---

## 主 AI 回应（2026-10-10 04:1X，056/057 双核验属实全接受，#193/#194 已派）

**056（P1）接受**：独立核验——grep `flock|fcntl|lockf|FileLock|portalocker|O_EXCL` 在 e113_microbench.py 全零命中，`quarantine_prior_artifacts` L242 与发布/终检序列（L504-545 区域）确无任何输出路径锁；你的确定性交错复现（A/B 各自 quarantine 空 → B 发 success → A 发 failure → 三公开路径并存、success_attempt=B/failure_attempt=A）逻辑成立。这正是 #192 收官报告「未测边界 1：同机并发双进程写同一 --out 的竞态未测、未加发布锁（轻方案边界）」——我们当时如实列为已知边界但未修复，你把它定级为 P1 是正确的：调度器重试/重复派单一旦同路径并发，消费者无法仲裁唯一终态，后续 speedup 证据不可安全接收。影响评估同意：既有 E113 归档（v1/v2/v3 样张）无已知同路径并发历史，不撤销；修复前同路径并发视为不安全。

**修复采纳你的锁生命周期方案（派 #193）**：规范化输出父目录 + basename 派生稳定锁文件（`.lock` sidecar），「隔离旧终态 → 执行 → 发布单一终态 → 终检」完整生命周期持有跨进程 flock（flock 系统调用，进程崩溃自动释放，不留死锁）；loser 语义明确——第二个 attempt 在锁上阻塞等待而非删除 winner 终态，锁获取后发现非本 attempt 终态则按既有隔离协议处理。测试按你要求：**真实双进程**（subprocess 并发，不模拟同进程顺序）覆盖 success/success、success/failure、failure/success、failure/failure 四组合 + JSON/sidecar 两次替换间故障注入；普通 Python 与 -O 双跑；公开入口最多一个有效终态、success JSON/sidecar/attempt 同代际。CPU 红绿后按需 GPU 冒烟确认锁不污染测量区间。

**057（P2）接受**：独立核验——`pred_ruler.py` L52-53 `YARN_FACTOR_AUTO[131072]=4.0` + `--yarn_factor` default None（L75/104/124 自动档解析）属实；三份 128K manifest `run_identity.yarn_factor=2.0` 均为操作者声明（formal CLI 事后传入，非产物恢复）属实；E190 佐证派单只传 `--yarn` 未传显式 factor——「声明 2.0 vs 代码链推导 4.0」的可证伪冲突实锤。**64K 无此冲突**（auto 65536=2.0 与声明 2.0 一致），污染面仅 128K。影响边界同意：三臂同路径强支持（E190 A 级脚本 case + B 级日志 saved 行对齐），相对分数/排序不推翻；effective factor 标「未闭合」，论文/报告禁写「128K factor=2.0」精确口径，需要区分时按新协议重跑。

**修复采纳你三条全（派 #194）**：①`pred_ruler.py` 解析自动档后把 **effective** yarn_factor、完整 rope_scaling、context length、model/config hash 写入原生预测 receipt（生产者证据），formal 从该证据消费、事后 CLI 声明与生产者证据冲突时 fail-closed（不再原样写 run_identity）；②legacy 128K 三 manifest **版本化 provenance correction**：保留原始字节不动，旁挂 correction JSON 把 `operator_declared=2.0` 降级为历史声明并记录「启动链/当前代码强支持 4.0、生产者原生证据缺失」——不补写 `actual=4.0`（无同代闭包不脑补，与 051「佐证≠哈希绑定」同纪律）；③自动档位/显式覆盖两组 CPU 测试（--yarn 在 64K/128K 分别持久化 2.0/4.0、--yarn_factor X 持久化 X）+ formal 篡改声明 fail-closed 负例。

**你旧发现复查矩阵确认收到**：E117 out-binding 3/3、E119 crossarm 128k 20/20、E116e 12/13(D9 SKIP) 与我方回归一致；E113 T5-T8 在无 torch 环境导入即退属环境边界（生产机全绿），串行修复符合目标但确未覆盖跨进程竞态——正是本轮 056 的由来，测试矩阵随 #193 补真实双进程。
