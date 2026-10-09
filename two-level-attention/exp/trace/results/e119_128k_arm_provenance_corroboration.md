# E190 — 128K 三臂 treatment 同时代运行佐证（GPT 审计 TL-E119-128K-FORMAL-CLOSURE-051 回应材料）

- 收集时刻：2026-10-10 02:32 CST；方式：四机只读收割（/tmp 日志与派单脚本 grep、仓库脚本、formal manifest、.251 活进程 ps）。未修改任何生产数据/代码/进程。
- 机器：本地 / H20 / .187 / .251。详细 33 格对账表见同名 `.json`。

## 一、051 核验结论：属实，不推翻

`sparse_attn/info.py` 的 `get_method_name_with_info`（tli 分支）只编码
`tia_block_size(64) / level1_topk(128) / level2_topk(1024) / cmp_ratio(4)` 与开关位
`A(subspace) B(kmeans) D(layer_skip)`，**不编码** `tli_far_method / tli_near_method / tli_alpha / tli_beta / tli_gamma`。
因此 mavg 与 aavg 两臂的文件名方法段、manifest `extra_params` 完全同串 `tli_64_128_1024_c4_A`
——051 的根因属实。本文件只补充**运行侧同时代佐证**，证明两臂在派单与运行层面配置确实不同。

## 二、诚实边界（必须随任何引用传递）

1. **同时代佐证、非哈希绑定**：`saved ->` 日志行证明「某文件名在该时刻该机产出」，派单记录证明「该链进程 argv 配置」，靠文件名+时间戳+任务序对齐；没有 pred 字节→进程的密码学绑定。
2. **不能将 051 升级为 closed**：128K mavg/aavg 归因（+1.06 / −3.87）保持 **provisional**；closed 须按 E120 口径在未来数据上以原生身份字段重建。
3. 操作者会话派单记录（`~/.claude/projects/.../62aa2193-*.jsonl` tool_use Bash 条目，UTC 时间戳）为操作者自查材料，独立性弱于第三方证据，已按 evidence_level 如实分级。

## 三、完整性锚点（收集时实测）

对 33 个 best_file 逐个重算本地生产文件 SHA256，与
`e119_ruler128k_formal_{arm}.json.manifest.json` 的 `tasks[].source_sha256` 比对：
**33/33 逐字节一致**——本表对账的文件正是 formal 收口（mavg 47.49 / FullKV 46.43 / aavg 42.56）实际消费的字节。
（该锚点只证明「本地文件 == 收口输入」，不证明「文件字节 == 某配置进程输出」——后者正是 051 的缺口。）

## 四、对账结果：33/33 全配到证据

| 维度 | 计数 |
|---|---|
| best-file 格配到证据 | **33 / 33** |
| A 级（配置编码在脚本 per-arm case 内，脚本 SHA 在案） | 24 |
| B 级（long_tasks.sh EXTRA 派单：会话派单命令原文 + 远端日志起跑/saved 行 +（部分）ps argv 快照三方对齐） | 9 |
| 按来源机：.187 10 / H20 12 / .251 10 / 本地 1 | 33 |

### 四机脚本 SHA 对照

| 脚本 | 本地 | H20 | .187 | .251 | 角色 |
|---|---|---|---|---|---|
| `/tmp/e109_ruler128_tasks.sh` | b94128e4 | b94128e4（一致） | — | — | H20 128K 补位主力，per-arm case L15-17 |
| `/tmp/e109_ruler_long_tasks.sh` | cd21c840（05:37Z sed 修 MODEL 后） | — | 465d0314（仅 MODEL 行不同，逐行逻辑一致） | — | 长档包抄链，EXTRA 透传 |
| `/tmp/e109_251_ruler128k.sh` | — | — | — | 212e62b2 | .251 双卡，per-arm case 全 αβγ |
| `benchmark/RULER/run_ruler_e109.sh`（仓库 runner） | 461716ad（本地化 MODEL/cd） | 65a2f7a2 | 65a2f7a2（H20=.187 逐字节一致） | — | 早期 r128 三链（TS 10090643/10090704） |
| `/tmp/e109_ruler64_tasks.sh` | b7095f78 | b7095f78 | b7095f78 | — | 仅 64K 档，与 128K 无关（排除混淆用） |

### 三臂配置编码三源一致（+1 个进程级直接佐证）

- **mavg**：`--tli_far_method minmax --tli_near_method avg --tli_enable_kmeans false --tli_alpha 0.25 --tli_beta 0.125 --tli_gamma 0.625`
  见 run_ruler_e109.sh L34、e109_ruler128_tasks.sh L15、.251 脚本 case、long_tasks 派单 EXTRA（会话记录 2026-10-08T23:59:22Z / 00:08:18Z / 04:39:49Z 原文）。
- **aavg**：`--tli_far_method avg --tli_near_method avg --tli_enable_kmeans false --tli_alpha 0 --tli_beta 0 --tli_gamma 0`
  见 run_ruler_e109.sh L38、ruler128 L16、.251 case、long_tasks EXTRA；另有 **.251 活进程 ps argv**（收集时 pid 2514139，全参数可见）直接印证。
- **fullkv**：`--method none`（无任何 TLI 参数；long_tasks.sh 路径靠 argparse 后位覆盖硬编码 `--method tli`）
  见 run_ruler L40-42、ruler128 L17、.251 case；另有 .251 活进程 ps argv（pid 2524519）直接印证。

## 五、33 格对账速览（详情见 JSON per_file）

| 臂 | 格（best_file 时间戳 → 来源机/GPU → 证据级） |
|---|---|
| mavg | s1 10090643→.187/G6 A；s2 10090643→.187/G6 A；s3 10091107→H20/G6 A；mk1 10090808→.187/G4 B；mk2 10091039→H20/G2 A；mk3 10091240→.187/G3 B；mq 10091240→.187/G3 B；mv 10091533→H20/G7 A；cwe 10091039→H20/G2 A；fwe 10091224→H20/G7 A；vt 10090759→.187/G0 B |
| aavg | s1 10090704→H20/G4 A；s2 10090922→.251/G1 A；s3 10091218→.251/G1 A；mk1 10091516→.251/G1 A；mk2 10091813→.251/G1 A；mk3 10091240→.187/G5 B；mq 10091830→H20/G7 A；mv 10091039→H20/G5 A；cwe 10091039→H20/G5 A；fwe 10091039→H20/G5 A；vt 10090808→.187/G1 B |
| fullkv | s1 10090704→H20/G1 A；s2 10090808→.187/G7 B；s3 10090808→.187/G7 B；mk1 10091039→H20/G0 A；mk2 10090922→.251/G0 A；mk3 10091402→本地/G0 B；mq 10091332→.251/G0 A；mv 10091535→.251/G0 A；cwe 10091740→.251/G0 A；fwe 10091943→.251/G0 A；vt 10092148→.251/G0 A |

## 六、gap 与旁支观察（如实记录，不脑补）

1. **在飞件**：.251 收集时仍在跑 aavg/niah_multiquery-10100009（74 行）与 fullkv/niah_single_3-10100158（16 行），均**不属于收口输入**（33 格 best-file 全为 100 行且先于收口时点完成）；其 ps argv 恰好提供两臂配置的直接进程级佐证。
2. **失败派单·本地**：fullkv cwe/fwe 首派（12:54 CST）因本地脚本 MODEL 行为 .187 路径 11s 双任务失败；13:37 sed 修复后重派成功——解释本地/.187 long_tasks.sh SHA 差异仅 MODEL 行。
3. **失败派单·H20**：GPU7 mavg 派单表 "fwe mv" 中 mv 非法任务名被 argparse 秒拒（DONE failed=1）；fwe 正常落盘（10091224 即 best-file），无残留半文件。
4. **.187 r128 mavg 链首派失败**：22:39Z 相对路径 cd 失败，22:43Z 绝对路径重派成功；现存日志为成功链。
5. **旁支观察（与 051 无关）**：128K manifest `yarn_factor` 声明 2.0（operator-declared），而派单只传 `--yarn`，pred_ruler `YARN_FACTOR_AUTO[131072]=4.0` → 实际运行应为 4.0。三臂声明/行为同值，跨臂可比性不受影响；建议并入 GPT 后续审计项。
6. **多文件同格**：部分格有重复产物（SKIP 幂等 + best-file 仲裁），本表只对账 manifest 仲裁选中的 best_file；同格其余历史文件来源日志同样可查。

## 七、结论口径

四机派单脚本、时间戳化派单命令、远端日志起跑/saved 行、.251 活进程 argv、formal manifest
`source_sha256` 33/33 逐字节一致——五个独立层面相互印证：**128K mavg/aavg/fullkv 三臂在运行时确实按
`minmax/avg/α.25/β.125/γ.625` vs `avg/avg/α0/β0/γ0` vs `--method none` 三套不同配置产生**。
但按诚实边界第 2 条，此为 provisional 归因的运行佐证，051 保持非 closed。
