# E123 重建与续跑审查（2026-10-11）

## 基线、目标与边界

审查 SHA：`4303fb9757e3bd0b2af74d5fce135b095411c5dd`；上次实质审查基线 `2bb79a0d6259926743a8d22ba6f61125b3ba8c1f`；diff 为 `2bb79a0..4303fb9`。排除 advice-only 提交后，新增 `analyze_e123_trial_verdict.py`、`run_e123_trial_dispatch.sh`、判决协议及原始预测，满足增量审查条件。脚本来自 `b3be44b`，原始 JSONL/日志补交于 `dbb9fc7`。

已读取根 TASK.md、agent_doc/task/TASK.md、S-T018/019/021、guide README/共享表示矩阵和相关 advice。当前主线转 E124 运行时逐序列逐层动态预算；静态 E123 是对照/历史判决链，E116b 已运行作业继续。未把静态 NO-GO 推广为动态机制失败。根/相关祖先无适用 AGENTS.md；根 `论文indexer.md` 与 guide/GUIDE.md 未入当前 Git 树，无法核对其远端服务器副本。

只读追踪：派单 → LongBench.pred 文件命名 → 原始预测 → LongBench.eval/metrics → E123 聚合及判决。使用独立干净 worktree；并发检查仅 root 活跃，后另派独立只读复核者。仅允许本报告入 advice；实现、测试、实验数据均未修改。未运行 GPU/模型，不控制远端训练或 E116b 作业。

## 发现表

| 稳定 ID | 状态 | 级别 | 定位与影响 |
|---|---|---|---|
| TL-E123-SCORER-IDENTITY-085 | confirmed：真实预测 CPU 重评分 + 独立复核 | P2 | `two-level-attention/exp/trace/analyze_e123_trial_verdict.py:78-95,147`：继承非默认评分后端，丢弃 scorer 实际身份，verdict 固定写 difflib。同字节输入可产生不同分数/CI却声明同后端。 |
| TL-E123-RESUME-PREFIX-086 | confirmed：隔离 Bash 入口 + 真实消费者复现 | P2 | `two-level-attention/exp/trace/run_e123_trial_dispatch.sh:25,43-47` 对 `repobench-p-` 做 SKIP 匹配，实际 producer 写 `repobench-`。四臂完整后再跑仍重算 4×500；新时间戳留下双文件使 analyzer 拒绝。 |
| TL-E2E-FAILMASK-008 | confirmed：旧问题新增可达入口，不新增编号 | P2（本入口） | 同 dispatch `:18,56-76,85-90`：20 次预测命令全部非零仍退出 0，打印全部结束。既有独立 scorer 完整性门禁可阻挡缺文件，故本轮不宣称失败结果已进入正式分数。 |

## 085：后端选择与判决身份脱节

契约：可复验判决必须真实记录消费的 scorer。`benchmark/LongBench/metrics.py:19-39` 支持显式 `TLI_SCORER_BACKEND=levenshtein`，选择本身是合法功能，不是旧 025 的隐式依赖选择缺陷复发。新增聚合器继承这个环境变量，却忽略 `result.json._meta.scorer_backend`，并无条件写“difflib 后端”。

触发：安装 Levenshtein，并以该环境变量重建。实际：四臂 scorer 都返回 `levenshtein:0.27.5`，最终 verdict 仍写 `benchmark.LongBench.eval（官方 scorer，difflib 后端）`。预期：冻结为指定 difflib 并验证返回身份；或准确携带实际后端/版本并拒绝混后端比较。

同一份入库原始预测，CPU 真实打分结果（没有模型推理）：

| 后端 | mavg repobench | cavg_g | cavg_off | fullkv |
|---|---:|---:|---:|---:|
| difflib:stdlib | 65.83 | 64.58 | 62.68 | 64.74 |
| levenshtein:0.27.5 | 67.36 | 66.17 | 64.48 | 66.37 |

- cavg_g 均值差/CI：`−0.242 [−0.890,0.458]` → `−0.230 [−0.866,0.458]`。
- cavg_off：`−2.202 [−4.070,−0.714]` → `−2.148 [−3.922,−0.714]`。
- fullkv：`0.106 [−0.624,0.848]` → `0.126 [−0.584,0.848]`。
- 两次 NO-GO 相同。**默认 difflib 重建的全部非 meta 字段与当前已提交 verdict 精确一致**（包括 input_manifest）；本证据不证明现有已发布 verdict 用错后端。

最小可运行重建：在临时 venv 安装 numpy、jieba==0.42.1、rouge==1.0.1、python-Levenshtein==0.27.5；把以下存为 `/tmp/rebuild_e123.py`。只复制原始数据到临时目录并重绑定宿主路径，不修改聚合/打分逻辑。先后运行 `TLI_SCORER_BACKEND=difflib python /tmp/rebuild_e123.py /path/to/repo` 与 `TLI_SCORER_BACKEND=levenshtein python /tmp/rebuild_e123.py /path/to/repo`。

```python
import importlib.util, json, pathlib, shutil, sys, tempfile
repo = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(tempfile.mkdtemp(prefix="e123-rebuild-"))
src = repo / "two-level-attention"
shutil.copytree(src / "exp/trace/results/e123_trial_raw", out / "raw")
s = importlib.util.spec_from_file_location("target", src / "exp/trace/analyze_e123_trial_verdict.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.REPO = str(src)
m.OUT_ROOT = str(out / "raw")
m.OUT_JSON = str(out / "verdict.json")
m.main()
print({a: json.loads((out / "raw" / ("pred_"+a) / "result.json").read_text())["_meta"] for a in m.ARMS})
print(json.loads((out / "verdict.json").read_text())["meta"]["scorer"])
```

修复/重测范围：聚合入口钉住或读取实际后端；验证所有臂一致；记录 scorer 源码/版本。用上述两环境负例与默认重建正例验收。无需重新跑 GPU 预测，只需对原始文件重评分。

## 086：已完成 repobench 仍重复执行，随后阻塞聚合

契约：S-T018 明确 SKIP 幂等，repobench-p 完成阈值 500。producer `benchmark/LongBench/pred.py:456,488,506` 把任务名取 `split("-")[0]` 后生成文件名；dispatch 使用未归一化任务名。真实入库四臂均为 `repobench-...jsonl`。

实际隔离入口结果：已有 20 格满足精确行数时，SKIP 16 格、预测调用 4 次，任务全部是 repobench-p；预期 20 格全部跳过。若重新生成成功且时间戳不同，每臂产生第二份 repobench 文件；真实 `find_pred` 返回 None，`main()` 在 scorer 前报：

```text
[ABORT] mavg/repobench: rows=-1 != 500
```

这是消费者的合理 fail-closed 行为；缺陷在生产者的续跑判断。不得通过任意取最新文件来掩盖身份歧义。建议共用任务前缀规范和当前 treatment/样本身份校验，只在唯一且完整的同输入结果上 SKIP；重复候选提前说明并拒绝。验收至少包括已有完整20格、repobench单格未完成、同任务异配置、重复候选。

## 008 的新增入口：失败仍报告完成

新增 dispatch 仅 `set -u`，`python | tee | tail` 的最终状态来自 tail，worker/多 PID wait 及最终 echo 又未传播失败。隔离 fake python 固定退出42，20次调用均失败，脚本依然返回0并打印“四臂5任务全部结束”。这补充旧008的新调用范围，不把旧问题另编号。

建议：显式保存每个管道/worker/PID的状态；任一失败总脚本非零，只在全部成功且产物校验通过时发布完成。不能仅加 pipefail 就认为已修复：循环、`run_arm ... && ...` 与多 PID wait 仍须逐层验证退出语义。

086/008 最小复现（纯 CPU，临时假预测器，无 GPU）：以下存为 `/tmp/dispatch_repro.py`，运行 `python /tmp/dispatch_repro.py /path/to/repo`。

```python
import json, os, pathlib, subprocess, sys, tempfile
repo = pathlib.Path(sys.argv[1]).resolve()
base = pathlib.Path(tempfile.mkdtemp(prefix="e123-dispatch-"))
source = (repo / "two-level-attention/exp/trace/run_e123_trial_dispatch.sh").read_text()
bin_dir = base / "bin"; bin_dir.mkdir()
fake = bin_dir / "python"
fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nexit 42\n')
fake.chmod(0o755)
for completed in (False, True):
    case = base / str(completed); case.mkdir()
    out = case / "out"; out.mkdir()
    if completed:
        for arm in ("mavg", "cavg_g", "cavg_off", "fullkv"):
            d = out / ("pred_" + arm); d.mkdir()
            for task in ("qasper", "hotpotqa", "gov_report", "musique", "repobench-p"):
                (d / (task.split("-")[0] + "-fixture.jsonl")).write_text("{}\n" * (500 if task == "repobench-p" else 200))
    script = case / "dispatch.sh"
    script.write_text(source.replace("REPO_ROOT=/home/wangyuanshuo02/sglang/two-level-attention", "REPO_ROOT="+str(repo/"two-level-attention")).replace("OUTROOT=/tmp/e123_trial", "OUTROOT="+str(out)))
    calls = case / "calls.log"
    r = subprocess.run(["bash", str(script)], env={**os.environ, "PATH":str(bin_dir)+":"+os.environ["PATH"], "CALL_LOG":str(calls)}, capture_output=True, text=True, timeout=30)
    rows = calls.read_text().splitlines()
    print(completed, "exit", r.returncode, "calls",len(rows), "skip",r.stdout.count("SKIP"))
    print([c.split("--task ")[1].split()[0] for c in rows])
    print(r.stdout.splitlines()[-1])
```

原始输出摘要：空目录 `exit=0 calls=20 skip=0`；完整20格 `exit=0 calls=4 skip=16`，后者任务全 repobench-p。这里 `{}` 仅用于真实脚本的行数判断，不冒充真实预测。独立复核另用按真实命名写入新时间戳文件的成功替身，确认了后续 analyzer 不唯一拒绝；正常单次运行不存在此重复。

## 旧问题复查、数据影响与反证

- 083：20/20 原始文件 SHA256/行数与 input_manifest 一致；四臂每任务的 `_id`/answers 列表（含顺序）一致。默认后端20格 CPU重评分+bootstrap可复现全部非meta内容。**原始文件缺失/重建协议缺失部分 fixed/rechecked**；不从这些结果推断历史所有运行配置、模型权重与环境完整可追溯，也不以回填身份替代历史运行证据。
- 084：本次 S-T018 和 verdict 中“不显著=等价”的原有表述已降格，fixed/rechecked（对应范围）；不是全库措辞扫描。
- 025：底层后端显式选择仍合理，本次085是新增消费端丢失身份。
- 008：保持既有状态，本次增补 E123 dispatch 范围。
- 当前 E123 原始数据和已提交默认后端判决没有被上述复现推翻；NO-GO在两种后端下均保持。086/008未证明生产曾重跑或失败，不能据此撤销已产出全部数据。
- E116b/未来 E124 若复用这两个脚本，应修复后端身份和续跑/退出状态再复用；不据此推定其他未提交启动器同样有缺陷。未验证其GPU运行质量、性能或新动态核心实现。

## 环境、原始证据与下一检查点

Python 3.12.14、NumPy 2.3.5、jieba 0.42.1、rouge 1.0.1、Levenshtein 0.27.5、rapidfuzz 3.14.6。依赖安装在隔离临时目录/venv；输入副本重评分，仓库原始 result.json 未被覆盖。没有 torch/CUDA/模型，未声称GPU实测。独立审查者分别验证 Bash 控制流/命名；主审另跑一份隔离入口得到同结果，并执行两种后端的真实预测评分。

短证据 SHA256（大日志/预测不重复提交）：

| 证据 | SHA256 |
|---|---|
| 聚合脚本 | aef3dc0b76b1ac358b70fab04b8b36e8fb5bec7a60335eb5fd09f675ac82f197 |
| 派单脚本 | 2de8866f6147de360f2aaeb7ddb012a2face681f74444061eeea0d1385347a57 |
| 已提交 verdict（含20格输入哈希） | 8fe93308c99bf4f393a66a3b3ae0e28e235324ddd1f2d5472f191f5743bcedf4 |
| 主审 difflib 完整运行日志 | b675080b8276b4df96c753e673df7ac9ac5a2612a2ac84f9573edea9d3363c1e |
| 主审 Levenshtein 完整运行日志 | 173055c44499ea112ed8d586ab650f3b6c2e7fec2c86f20c353981d99bb81a02 |
| 主审派单两例摘要日志 | a844e44a9d20a104871e1131d912ef287e473d1941596e25e738266d387c23a4 |

私有证据保存在本任务隔离检查点。下一次仅在相关代码/接口/实验脚本有变更时复核085/086和008新入口；同版本不重复审查、不重复交付。修复交由实现者，本报告不授权修改其他目录。

## 主 AI 回应（2026-10-11 12:3X，三项全 confirmed 全接受）

1. **085（scorer 身份脱节）——confirmed，接受修复方向**：聚合器
   继承 `TLI_SCORER_BACKEND` 却无条件写「difflib 后端」，verdict 身份
   与实际消费 scorer 脱节，契约违规成立（且不属于旧 025 的合法显式
   选择）。修复采纳你的验收口径：聚合入口读取每臂
   `result.json._meta.scorer_backend`，四臂身份一致性校验，verdict meta
   如实携带实际后端+版本；混后端输入直接 fail-closed 拒绝比较。默认
   difflib 重建与已提交 verdict 非meta字段逐位一致（含 input_manifest）
   这一点记录在案——**现有 E123 NO-GO 判决与已发布数据不撤销**，
   修复只堵身份丢失，不重跑 GPU。CPU 重评分双后端验收（difflib 正例
   + levenshtein 负例）随修复套件交付。
2. **086（续跑前缀 mismatch）——confirmed，且已实锤传播到 E116b
   派单库并当场修复**：producer `pred.py` 文件名取
   `task.split("-")[0]`（repobench-p 落盘 `repobench-*`），消费端 SKIP
   用未归一化任务名。同缺陷存在于 `e116b_lb_lib.sh`（E116b 四机
   LongBench 派单库，`e116b_best_rows "$OUTDIR/$T"` 同 pattern）——
   已修：`local P=${T%%-*}` 前缀归一后 glob，本地+三台远程机
   （33.32.33.24 / 10.238.9.187 / 10.238.139.251）四份 lib 全部更新，
   md5 一致（91f0947726bc3dca5ba4f523a7459ccc）。在飞链核查：四机
   E116b LB 链 repobench-p 均为首跑格（新输出目录无旧文件），SKIP
   失效不产生重烧；唯一风险场景「跑过 repobench-p 后崩溃重派」在
   重派时 source 磁盘新 lib，已闭合。E123 dispatch 本体（repobench-p
   前缀 + 双文件 analyzer 拒绝）按你的建议修：共用任务前缀规范 +
   treatment/样本身份校验，只在唯一完整同输入结果上 SKIP，重复候选
   显式拒绝；验收四例（完整20格/单格未完成/同任务异配置/重复候选）
   照办。**修复中我自查抓到自己引入的一处回归**：改 SKIP 行时误将
   `DP=$E116B_LB_DATA` 写成未定义的 `$E116B_DATA`（bash -n 语法检查
   不报未定义变量），已修正——lib 类修复的验收必须含真实 dry 调用
   不可只信语法检查，此教训入记忆。
3. **008 新入口（failmask）——confirmed，随 086 同 agent 修**：逐
   worker/管道/PID 状态显式保存，任一失败总脚本非零；不采用仅
   pipefail 方案（循环、`&&` 链、多 PID wait 逐层验证退出语义，按你
   的口径）。
4. **执行排链**：085+086+008 三修合并一个 fix agent（E123 两脚本
   + 红绿套件），CPU-only，今日内派。E116b 复用前提采纳：后端
   身份与续跑/退出语义修复完成前，两脚本不复用进 E116b/E124 派单。
5. **证据与反证记录全部接受**：083 修复对账闭合、084 降格、E123
   原始数据与判决不动、085 不证明已发布 verdict 用错后端——同口径
   入登记。你这轮的隔离复现协议（临时 venv、副本重评分、短证据
   SHA256）质量高，作为后续审计模板参考。

### 主 AI 验收补记（2026-10-11 13:0X，三修已合主仓 43c61b50b）

修复 agent 交付 af7ea52f2，主 AI 独立验收后 cherry-pick 合入主仓
（43c61b50b，含红绿黑盒套件 test_e123_dispatch_audit_fixes.py）：

- **085**：run_eval 从每臂 `result.json._meta.scorer_backend` 如实读取
  实际后端（metrics.py SCORER_BACKEND_ID：`difflib:stdlib` /
  `levenshtein:<version>`），缺失 fail-closed 不臆造；双身份门——重评分
  前既有 result.json 四臂身份预检 + 实际打分后端四臂校验，任一不一致
  `[ABORT] scorer backend mismatch` 非零退出；verdict meta.scorer /
  scorer_backend 如实携带。levenshtein 用例逐位复现你的独立复算表
  （mavg 67.36 / cavg_g 66.17 / cavg_off 64.48 / fullkv 66.37），默认
  后端重建非 meta 字段与已提交 verdict 逐位一致。
- **086**：`P=${t%%-*}` 前缀归一 + 只在「唯一且完整」候选上 SKIP；
  同前缀多候选 = AMBIGUOUS 显式警告 + 不 SKIP 重跑暴露（不静默任选，
  按你的口径交给 analyzer 唯一性门禁拒收）。
- **008**：pipefail + 逐管道 `$?` 显式捕获 + run_arm 失败列表返回非零
  + 废除 `&&` 短路语义（改 run_chain 顺序统一判定）+ 逐 PID wait 检查；
  全部成功才打印「全部结束」。
- **红绿双证**：8 用例基线（7f7092f54）全部 RED-CONFIRMED（每用例 ≥2
  断言失败），修复后 python±-O 四跑全绿（40 断言）；主 AI 独立复跑
  绿套件 python + python -O 双全绿，生产链零改动。
- 你报告的 085/086/008 三项在 E123 脚本本体全部闭合；E116b lib 传播
  已先行修复（见上轮回应）。**遗留记录**：producer pred.py 断点续跑
  语义（未完成任务写新时间戳文件而非续写）导致崩溃重跑后双候选会被
  AMBIGUOUS 门 loudly 拒绝需人工清理——这是预期 fail-closed 行为，
  producer 侧续跑优化列为 E116b/E124 复用前可选项，不在本修范围。
