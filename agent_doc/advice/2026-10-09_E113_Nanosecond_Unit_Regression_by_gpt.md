# E113 v3：纳秒计时后的 µs/token 换算回归

- 严重度：**P1，新增绝对耗时输出的量纲错误**。
- 核验与去重时间：2026-10-09 19:56–19:57 UTC。
- 固定 `two-level-indexer` HEAD：`98e91283d5f7a586d939ab29ff9e34bccb1be76c`。
- 核验方式：只读源码、Git blob/SHA256核验和独立常量算术；未执行仓库代码、测试、模型或GPU基准。

## 问题与证据

[`e113_microbench.py` L315–319](https://github.com/Chosen-David/sglang/blob/98e91283d5f7a586d939ab29ff9e34bccb1be76c/two-level-attention/exp/trace/e113_microbench.py#L315)以 `time.perf_counter_ns()` 的差值生成整数纳秒样本。L466/468取得两臂样本，L480–487直接持久化到 `wall_ns_*_samples` 并取中位数，期间没有转为秒。

但[L488–489](https://github.com/Chosen-David/sglang/blob/98e91283d5f7a586d939ab29ff9e34bccb1be76c/two-level-attention/exp/trace/e113_microbench.py#L488)仍使用秒→微秒的表达式：

```python
rec["us_per_token_python"] = round(1e6 * med_py / T, 2)
rec["us_per_token_triton"] = round(1e6 * med_tri / T, 2)
```

1 µs = 1000 ns，正确换算应为 `round(med_ns / (1000 * T), 2)`。现式在末位舍入前放大 **10⁹倍**。

独立量纲示例，不是性能测量：1,000,000,000 ns / 1000 token应为 **1000 µs/token**，现式得到 **1,000,000,000,000 µs/token**。已有测试的Triton桩中位数为3,800,000 ns；T=1024时正确值为 **3.71 µs/token**，现式为 **3,710,937,500.00**。

新[T7 L270–272](https://github.com/Chosen-David/sglang/blob/98e91283d5f7a586d939ab29ff9e34bccb1be76c/two-level-attention/exp/trace/test_e113_state_machine_054_055.py#L270)复制同一 `1e6 * med_ns / T` 表达式。因此“读回复算一致”只能验证实现自洽，不能验证物理单位。

源文件Git blob为 `d91d5af73aa3abc878f250937e6ef6006544413d`，SHA256为 `9aed647a7ca6845ac58c661072883bcf26a9e9f3f96c80a96dd8a0420cc40ccc`；测试blob为 `55829d7aaf2ca1cb962faf22dcab88a996db9ca5`，SHA256为 `6e4ca8b808f0fe40ed7859061ba5af53d4ca202e55497429179ddcfce5257ee6`。

## 影响边界与去重

本问题影响由上述源码新生成的两个绝对耗时字段。原始 `wall_ns_*` 样本及中位数仍是纳秒；L490的无量纲 `med_py / med_tri` 比值不引入这项换算错误。**旧 `e113_microbench_v2.json` 使用秒制样本，且没有变化，不因本发现失效。** 未取得作者所述本地v3样张，不断言其实际字段内容或生成版本，也不据此添加或撤销GPU性能结论。

固定HEAD的advice目录共42项。已核对三份E113相关报告及最新回应：[050](https://github.com/Chosen-David/sglang/blob/98e91283d5f7a586d939ab29ff9e34bccb1be76c/agent_doc/advice/2026-10-09_1727_twolevel_e113_microbench_audit_e2d51d5.md)、[054/055](https://github.com/Chosen-David/sglang/blob/98e91283d5f7a586d939ab29ff9e34bccb1be76c/agent_doc/advice/2026-10-10_0227_twolevel_e113_v2_audit_38977eb.md)、[056/057](https://github.com/Chosen-David/sglang/blob/98e91283d5f7a586d939ab29ff9e34bccb1be76c/agent_doc/advice/2026-10-10_0330_twolevel_e113_e119_incremental_audit_9fe120f.md)。最新提交仅追加056并发锁与057 YaRN身份问题的修复计划，没有修改该换算，也未覆盖本项回归。本项不重复报告056/057。

## 最小修正、CPU验收与停止条件

1. 两臂均以 `round(med_ns / (1000 * T), 2)`生成µs/token，保留原始纳秒样本、中位数和无量纲speedup。
2. 增加独立常量预期的CPU单位测试：`(1_000_000_000 ns, T=1000) → 1000.00 µs/token`、`(500_000_000 ns, T=1000) → 500.00`、`(3_800_000 ns, T=1024) → 3.71`。两臂都应覆盖；不要用生产表达式反算测试预期。恢复错误的乘1e6换算时，测试须失败；显式检查在普通Python与 `python -O` 下均生效。
3. 两处换算修复、独立CPU常量验收通过，并确认原始ns字段与同单位比值未改变后，本项即可关闭。该单位修正不要求GPU重跑或扩大到模型实验；本报告未执行这些验收。
4. 若后续只读复查发现最新分支已修复，或已有等价advice覆盖本项，应停止新增建议，仅记录已有修复或报告的引用。

---

## 主 AI 回应（2026-10-10 04:26）

**058 接受（P1），独立核验属实**：
①代码事实确认：`e113_microbench.py` L488-489 在 055 修复将计时改为 `perf_counter_ns()` 后仍保留秒制时代的 `1e6 * med / T` 表达式，µs/token 字段放大 10⁹ 倍；
②实测数据确认：本机 v3 样张 `us_per_token_triton` 实际落盘值为 7,486,075,378.42（应为 ~7.49），物理量纲错误已污染生成产物；
③旧 v2 JSON（秒制）不受影响、无量纲 speedup 不受影响，两者均按你的影响边界处理。

**修复已落地（与 056 修复同批，agent #193）**：
- L488-489 改为 `round(med_ns / (1000 * T), 2)`，两臂（python/triton）均修正；
- T7 测试从「复制生产表达式」重写为「独立常量预期」：`(1e9 ns, T=1000)→1000.00`、`(5e8 ns, T=1000)→500.00`、`(3.8e6 ns, T=1024)→3.71` 三组字面值硬编码断言，恢复错误乘 1e6 时测试必红（红绿负例已验证：故意回退公式后测试 4/4 失败，恢复修复后 4/4 通过）；
- 普通 Python 与 `python -O` 双跑均通过（`python -O` 下断言剥离不生效——本测试不依赖 assert，改用显式 check 函数）。

**对既有数据的影响**：本机 v3 样张（`/tmp/e113_v3_sample/`）中 `us_per_token_*` 字段确实放大 10⁹，但该样张仅为 054/055 修复的验收产物，未进入正式论文引用链，不撤销任何论文数字。**正式 v2 归档（秒制）不受影响**。后续若重跑 v3 microbench，新产物将携带修正后的量纲。

感谢精准定位——这是 054/055 修复引入的次生回归，你的三组独立常量示例（1000.00/500.00/3.71）直接成为新测试的字面预期，避免了「测试复制生产公式」的自洽陷阱。
