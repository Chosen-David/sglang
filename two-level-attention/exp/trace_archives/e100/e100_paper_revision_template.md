# E100 收割轮论文修订模板（数字槽位待 e100_tail_ci.json 落袋后填）

## 判决分支选择（按 tail_vs_full_ci.contains_zero）
- **A 含 0（大概率）**：走「统计不可分」口径，主表臂口径注升级为双口径并报
- **B 不含 0**：走「维度敏感」口径，主表保 full-L1 + tail 消融行

## A 分支文案（替换 CN TLI_paper.tex L213 尾句 / EN TLI_paper_en.tex L216 尾句）

### CN 原句（要替换）：
「同一 $(\alpha,\beta)$ 的部署参照臂（$\gamma{=}0.125$）上，L1 上界取 tail32 子空间的成对校准给出 hotpotqa 54.83 对 54.43、musique 33.22 对 34.76（tail32/全维）——主表结论对 L1 上界维度不敏感。」

### CN 新句（A 分支）：
「主臂配置本身做了 tail32-L1 全 13 任务配对实测（{TAIL_AVG} 对 50.78，配对差 {DIFF}，95% CI [{CI_LO}, {CI_HI}] 含 0；逐样本 bootstrap B=10000）：L1 上界维度在主臂上是统计不可分的——但方向性结构存在，far 检索最重的 hotpotqa 掉 {HQ_DIFF} 分（$\gamma{=}0.625$ 的紧 far 预算放大上界信息损失），而代码与摘要任务差在 0.3 分内；早先 $\gamma{=}0.125$ 参照臂的成对校准（54.83 对 54.43 / 33.22 对 34.76）与之相容：维度敏感性随 far 预算收紧而增大。主表采用全维 L1 口径，tail32-L1 为同预算消融臂。」

### EN 原句（要替换）：
「on the same-$(\alpha,\beta)$ deployment-reference arm ($\gamma{=}0.125$), a paired calibration with the L1 bound on the tail32 subspace scores hotpotqa 54.83 versus 54.43 and musique 33.22 versus 34.76 (tail32/full), so the main-table conclusions are insensitive to the L1 bound dimension.」

### EN 新句（A 分支）：
「the main-arm configuration itself was re-measured with a tail32 L1 bound across all 13 tasks ({TAIL_AVG} versus 50.78, paired difference {DIFF}, 95\% CI [{CI_LO}, {CI_HI}] containing 0; per-sample bootstrap, $B{=}10000$): the L1 bound dimension is statistically indistinguishable on the main arm -- yet a directional structure exists, with far-retrieval-heavy hotpotqa dropping {HQ_DIFF} points ($\gamma{=}0.625$'s tight far budget amplifies the bound's information loss) while code and summarization tasks stay within 0.3 points; the earlier paired calibration on the $\gamma{=}0.125$ reference arm (54.83 versus 54.43 / 33.22 versus 34.76) is consistent with this: dimension sensitivity grows as the far budget tightens. The main table keeps the full-dimensional L1 protocol, with tail32-L1 as a same-budget ablation arm.」

## B 分支：保留原校准句 + 追加 E100 显著差披露（主表保 full-L1），模板同上但「statistically indistinguishable」改「a significant difference ({DIFF}, CI excludes 0)」

## 收割 checklist
1. python -u exp/trace/analyze_e100_tail_ci.py（cd two-level-attention）→ e100_tail_ci.json
2. 填槽位：TAIL_AVG/DIFF/CI_LO/CI_HI/HQ_DIFF（hotpotqa tail_minus_full = {HQ}）
3. 双语替换两处 + 检查 §4.2/消融是否有「维度不敏感」复述点（grep 不敏感/insensitive）
4. 编译双语 + pypdfium2 抽验
5. two-level commit（e100_tail_full.json + e100_tail_ci.json + analyze 脚本）+ sglang commit（tex）+ 双 push
