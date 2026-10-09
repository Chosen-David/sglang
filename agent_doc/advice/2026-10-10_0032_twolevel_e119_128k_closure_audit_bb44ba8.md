# TwoLevel E119 128K 三臂闭包门禁降级审查（bb44ba8）

## 审查目标、版本与边界

- 目标分支：`two-level-indexer`；最终审查 SHA：`bb44ba8fe649e7eef996a72d5b334ffc7567a0fb`。
- 新增相关提交：`cec319472` 落袋 128K 三臂 result/manifest/receipt/generation；审查过程中远端并发新增 `bb44ba8fe`，加入 128K 汇总器与 summary。本报告已按新 SHA 重新核验，不把旧 SHA 结论直接沿用。
- 实际链路：提交结论 → 三臂 result/manifest/receipt → 新 `analyze_e119_ruler128k_formal.py` → `e119_ruler128k_formal_summary.json` → 冠军及跨长度机制归因。
- 环境：干净独立检出，Python 3.12.14，Linux 6.18.44 x86_64。只做 CPU/文件级核验；没有 GPU、模型生成、kernel/e2e 或生产预测重跑。生产源预测和 generation 内派生 JSONL 未随提交入库，故本检出不能重算 scorer。

审查文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `analyze_e119_ruler128k_formal.py` | `3b4cad69ab3217efea8f2fa564a44c44a6fec3de75897142fcda720796220969` |
| `e119_ruler128k_formal_summary.json` | `a549ff35a5f804849be602eb206598dc493dce19866736192db647cc786634ec` |
| mavg manifest | `6f74d9e0e5404121b0c1dcba567f337dd679c9f9c62d6e497c50eb4bae9ff6bf` |
| aavg manifest | `fb90ca8a2db595915c3e121b5cf1edc54fd007e86466b4883fc65f8d316bb889` |
| FullKV manifest | `0b7f9a70d76136ec9731af213d989fe7012fcaaf37b71cc74faca0ee75f9187d` |

## 结论与发现表

| 稳定 ID | 状态 | 严重度 | 新结论 | 对已产出数据的影响 |
|---|---|---:|---|---|
| `TL-E119-CONSUMER-BINDING-046`（arm↔treatment 子项） | **recurred / confirmed（真实产物 + 新消费者 CPU 复核）** | P1 | 新 128K 汇总器不是补齐缺失 treatment，而是把 mavg 与 aavg 的 `ARM_CONTRACT` 都降级为同一个 `method=tli_64_128_1024_c4_A`。所以 mavg/aavg 交换后仍满足双方契约；目录名+预测 SHA 只能绑定“这些字节位于哪个人工命名目录”，不能证明字节由 minmax+avg 或 avg+avg 配置生成。 | 47.49、46.43、42.56 的 JSON 算术和共同样本身份可核验；但 **47.49 属于 mavg、42.56 属于 aavg 的语义归属未证明**。因此“mavg 冠军”“mavg-aavg +4.93”和“near/far 分区价值正向证据”仍为 **inconclusive**。FullKV 的 `method=none` 绑定成立。 |
| `TL-E119-128K-FORMAL-CLOSURE-051` | **confirmed / CPU 最小反例** | P1 | 新 summary 写出 `arm_contract_enforced=true` 并发布正式冠军，但当前契约对两条稀疏臂不是单射：`ARM_CONTRACT[mavg] == ARM_CONTRACT[aavg]`，而且双方 treatment 互换仍返回 true。该 closure flag 与“把数值冠到错误臂名称不可达”的注释/契约不符。 | 128K summary 是**假阳性闭包**，不能升级三个单臂观测为正式三臂排名，也不能作为三档冠军稳定的论文证据；原始单臂结果保留，不覆盖。 |

本轮没有发现 result JSON 的算术错误，也没有证据说明三个原始预测分数本身错误。需要撤回的是稀疏两臂的配置冠名和据此作出的算法归因，不是凭静态检查宣布模型分数无效。32K/64K 已有产物不因本次 128K 消费器缺口自动失效。

## 1. `TL-E119-CONSUMER-BINDING-046`：用相同 method 取代两条不同语义契约

### 位置与违反契约

- 根 `TASK.md:27-28` 定义：mavg=`(minmax, avg)`，aavg=`(avg, avg)`。
- 64K 正式消费者 `analyze_e119_ruler64k_formal.py:97-105` 的既有契约进一步要求：
  - mavg：`far_method=minmax, near_method=avg, alpha=.25, beta=.125, gamma=.625`；
  - aavg：`far_method=avg, near_method=avg, alpha=beta=gamma=0`；
  - FullKV：`method=none`。
- 新 128K 消费者 docstring `analyze_e119_ruler128k_formal.py:37-42` 仍声称完整不同契约会使错误冠名不可达；实现 `:98-112` 却明知 legacy 产物没有上述字段，把 mavg 与 aavg 的契约同时改为 `{"method":"tli_64_128_1024_c4_A"}`。
- 对应真实 manifest 也确实相同：
  - `e119_ruler128k_formal_mavg.json.manifest.json:9-19`；
  - `e119_ruler128k_formal_aavg.json.manifest.json:9-19`。

代码注释 `:104-106` 称 `pred_root/{mavg|aavg}` 路径和逐文件 SHA 已承载两臂区分。这只证明一组不可变预测字节被放在名为 `mavg` 或 `aavg` 的目录；目录名本身仍由操作者指定。预测/manifest 内没有 far/near/α/β/γ，也没有原始执行配置哈希，因此把两组目录命名互换、或一开始放错目录，哈希闭包仍会忠实证明错误标签下的字节未被改变。完整性不能替代 treatment 身份。

### 正向证据与反证

已提交产物的下列部分通过：

- 三臂 result/manifest SHA 与 receipt 一致；top-level JSON/MD/manifest/receipt 与已提交 generation 规范文件一致。
- 每臂 11 tasks×100 样本；task 集、IDs、答案哈希、长度、RULER 数据 SHA、model/YaRN 字段跨臂一致。
- formal/scorer SHA 三臂一致；scorer manifest SHA 均为 `139721c983ec4bdec54046be44bf1b3aca42e184eb34253718514f143a123637`。
- 平均分由 task 分数复算为 mavg 47.49、FullKV 46.43、aavg 42.56。

这些正向检查排除了样本集、评分器和简单算术差异，却不能反驳配置冠名错误；恰好说明唯一缺失的区分信息就是 treatment。

## 2. `TL-E119-128K-FORMAL-CLOSURE-051`：summary 对不可区分契约报通过

新消费者 `:430-435` 逐臂比较 `treatment == ARM_CONTRACT[arm]`，但 mavg/aavg 的 expected 本身完全相同；summary 又在 `:583-596` 无条件写出 `arm_contract_enforced=true`，最终发布 `mavg 47.49 > FullKV 46.43 > aavg 42.56`。

最小反例（只读导入新消费者和已提交 summary）：

```bash
python3 - <<'PY'
import importlib.util, json, pathlib
p = pathlib.Path('two-level-attention/exp/trace/analyze_e119_ruler128k_formal.py')
s = importlib.util.spec_from_file_location('a128', p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
q = pathlib.Path('two-level-attention/exp/trace/results/e119_ruler128k_formal_summary.json')
out = json.loads(q.read_text())
print('contracts_equal=', m.ARM_CONTRACT['mavg'] == m.ARM_CONTRACT['aavg'])
root = q.parent
for label, source in [('mavg', 'aavg'), ('aavg', 'mavg')]:
    manifest = json.loads((root/f'e119_ruler128k_formal_{source}.json.manifest.json').read_text())
    _, treatment, _ = m._arm_identity(manifest)
    print(f'label={label} source={source} accepted=', treatment == m.ARM_CONTRACT[label])
print('summary_claims_arm_contract_enforced=', out['closure']['arm_contract_enforced'])
PY
```

本轮实际输出：

```text
contracts_equal= True
label=mavg source=aavg accepted= True
label=aavg source=mavg accepted= True
summary_claims_arm_contract_enforced= True
```

这直接否定“交换 treatment 冠名不可达”和 `arm_contract_enforced=true`。不是性能争议，也不依赖 GPU；它是 summary 语义与可达反例之间的确定性矛盾。

## 实际执行测试、可复现性与旧发现复查

- `python3 two-level-attention/exp/trace/test_e119_crossarm_identity.py`：**13/13 PASS**。
- `python3 -O two-level-attention/exp/trace/test_e119_crossarm_identity.py`：**13/13 PASS**。
- 这两次测试只覆盖 64K 消费者；其中 N7 恰好要求 mavg/aavg 的完整不同 treatment，不能证明新 128K 的降级契约正确。新提交没有对应 128K treatment-swap 负例。
- `python3 two-level-attention/exp/trace/analyze_e119_ruler128k_formal.py` 在干净检出 **exit=1**：receipt 的 `outputs.derived_dir` 是生产机绝对路径 `/home/wangyuanshuo02/...`，本检出不存在；每个已提交 generation 目录也只有 5 个规范/manifest 文件，没有 11 个派生 prediction JSONL。故本轮不能独立重放提交所称的 `python`/`python -O` exit 0。这个可移植性/复验缺口沿用 046③ 的证据边界，不单独把“本机路径不存在”误报成分数错误。
- 045（`python -O` 显式门禁）、047（scorer SHA 公平性）、048（64K 合成 fixture）和 049（summary 原子写）在既有回归场景通过。本批三臂脚本 SHA 相同，未触发 047。
- 一次独立只读复核在不修改任何文件、不运行 GPU 的条件下确认：三臂 identity/scorer/算术闭合；mavg/aavg treatment 不可区分；新 summary 的 arm contract 与冠军结论未闭合。

## 修复与最小重测建议

1. 不要为迁就缺字段的 legacy 产物改写语义契约。恢复 mavg/aavg 的完整不同 `ARM_CONTRACT`；证据不足时应 fail closed 或输出 `inconclusive`，不能把两个相同 expected 称为 arm contract enforced。
2. 若原预测运行有不可变命令、config、调度快照或日志，把实际 far/near/α/β/γ 及其哈希绑定进 native prediction identity，再由 formal scorer 继承。仅事后按目录名手填参数仍不能证明执行身份。
3. 若无法从原运行证据恢复 treatment，只需重跑受影响的 mavg/aavg 两臂；FullKV 不因本问题重跑。保留当前匿名 TLI 分数作为历史观测，不覆盖。
4. 为 128K 消费者增加专属回归：交换 mavg/aavg 的完整 artifact/source 目录后必须失败；删一项 treatment 必须失败；普通与 `python -O` 双跑。明确断言 `ARM_CONTRACT['mavg'] != ARM_CONTRACT['aavg']`。
5. summary 的 closure 字段必须来自实际可区分门禁结果；在上述测试通过前，把当前 summary 与提交信息中的“正式收口/三档冠军稳定/分区价值证据”改为 provisional。
6. 另行完成 046③ 的 v2 同快照消费与可移植 generation 定位复验；这不替代本报告的 arm 语义修复。

## 未覆盖项与下一检查点

没有使用 GPU，也没有重算 33 份生产预测；near/far、L1/L2 kernel、prefill/decode、速度与显存未因本次结果提交变化而重审。本次确认的是结果消费者把不可区分的目录标签误当 treatment 证据，不能外推为核心实现错误或模型精度错误。下一次只在相关代码/实验链有新改动时复验：优先检查 native 运行配置绑定、mavg/aavg 交换负例和修订后的 128K summary。
