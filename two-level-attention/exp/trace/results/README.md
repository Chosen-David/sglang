# results/ — 实验落袋 JSON 目录

本目录存放任务链各实验的**最终判决 JSON**（eNN_ 前缀按实验代号命名，如 `e98_best_election.json`；跨方法主题实验无代号前缀，如 `m6_sinkguard_full.json`）。落袋 JSON 是论文与可视化脚本的唯一数据源——数字勿手改，修订须重跑产出脚本并在 commit message 写明口径。分片产出带 `_s0/_s1` 后缀，merge 后的主 JSON 不带后缀。`/tmp` 下的跑批脚本与中间 pred 输出**不入库**（管线按绝对路径引用 exp/trace/ 脚本，/tmp 属临时边界）；打分为会话内联时，落袋 commit 即审计锚点。脚本→数据→结果→论文引用的完整映射见仓库根 `CODEMAP.md`。
