# kimi3《20×H20 多机多卡编排建议（BeeGFS 单一协同面 v2）》核验与取舍（2026-10-08）

来源：`agent_doc/advice/2026-10-08_kimi3_multi_gpu_orchestration.md`
核验者：主 AI。方法：逐条对照本会话三机（本机 2 卡 + H20 8 卡 + .251 2 卡）+ .187 8 卡接入的实测经历。

## 总判决：**adapt（方向采纳，切换时机推迟到 E109 收官后）**

建议对问题的诊断全部被本会话实测验证；但 E109 海选链剩余寿命 ~1-2 天，
调度架构切换的工程成本（dispatch.py ~150 行 + 三机验证）超过剩余收益。
**E110/full13/RULER 一律生在新架构里**（与建议 §6 迁移路径一致）。

## 实测验证表（建议的预判 vs 本会话实况）

| 建议论断 | 本会话实测证据 | 判定 |
|---|---|---|
| 代码漂移（§2：远程独立副本，无同步） | .187 部署走 rsync 快照（27M 代码非 git）；.24 远程副本 HEAD 与主树不同 lineage 已实锤；SKIP glob 补修须逐机 rsync 推送 | **验证** |
| 自由 SKIP + jobs 快照的结构缺陷（§2 协调行） | **两次同根因事故**：H20 GPU3/7 空闲（stride-8 倒序槽位全 SKIP）、.187 GPU2/3/7 立即 DONE——都是 jobs 数组按启动时刻快照生成，新完成臂不纳入 | **验证**（refill 重挂链是手工补丁，manifest 动态 claim 天然免疫） |
| marker grep 协议脆弱（B07/F02） | repobench-p 文件名前缀 bug（`t.split("-")[0]`）已让 SKIP/打分双失配一次 | **验证** |
| BeeGFS 是唯一健康通道（§2 末行） | 三机模型路径同一 dolphinfs 挂载（Qwen3-8B 全机直接读），唯一未利用的协同能力 | **验证** |
| 静态分片长尾风险（§4） | gov_report 2h54m/臂 vs qasper 17min，速率方差 ~10×，静态均分必有长尾机 | **验证** |

## 逐项取舍

| 项 | 判定 | 理由 |
|---|---|---|
| §3 代码单源（git archive + sha256 拒启） | **adopt**（E110 起） | 漂移已实测；sha256 启动校验把「靠记忆」变硬失败 |
| §3 结果直写 BeeGFS（.tmp+rename 原子发布） | **adopt**（E110 起） | 消灭拉回循环（本轮已挂 3 个：H20/.251/.187），收官落袋 rsync 保留一次性 |
| §3 manifest mkdir 原子 claim 调度 | **adopt**（E110 起） | 本会话两次 GPU 空闲事故的直接根治方案 |
| §4 预分片长尾 + 动态 claim 两层 | **adapt** | 方向对；E109 收官前的 sim 臂/mminmax 边界臂分摊仍用现有 refill 模式 |
| §6 E109 在跑链不动 + 最后一次手工 sync | **adopt** | 与本轮实际做法一致（拉回循环收官后 N1 重跑 score_v2） |
| §7 前置检查清单 | **adopt** | 启动 E110 前逐项过；F5（杀 wys2 坍塌 sim 臂）须先确认其实况 |
| §8 时间线 | **defer** | 依赖 §3 落地速度；E109 海选数据优先，时间线按实际收官滚动修正 |

## 与现有运维的衔接（E109 剩余期）

- 三机四链（本机正序 + H20 倒序+refill + .251 sim + .187 倒序+refill）+ 三个拉回循环维持不动；
- 每次 refill 式重挂是「手工 sync 不可持续」的又一实证，累计入 E110 调度器需求；
- 收官动作不变：全量 rsync 落袋 + score_v2 重跑（N1）+ commit push。

## E110 启动前的最小落地清单（从建议 §3/§7 派生）

1. `tli_runs/<stage>/` 目录骨架 + code/ 快照 + sha256 清单生成；
2. dispatch.py（枚举缺口 → mkdir claim → 调 pred → 写状态）首版；
3. 三台远程机 BeeGFS 挂载与写权限验证（mount | grep bgfuse + rename 测试）；
4. wys/wys2 ssh 包装脚本（.24 已有 deploy 经验可复制）。
