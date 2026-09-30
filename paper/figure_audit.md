# E77：论文图清单审计（2026-09-30）

tex 引用 6 图 vs exp/figures 现存 19 图。原则：一图一论点（顶会审美：
Fig1 做成 architecture+teaser 复合图；实验图统一 house style；架构图留
draw.io 源文件）。

## 现有 6 图引用审计

| tex 引用 | 论点 | 判决 | 动作 |
|---|---|---|---|
| fig7_tli_architecture | Fig1 主架构图 | **留（核心）但重做** | matplotlib 手绘稿 → draw.io 重制 + 留 .drawio 源文件；内容需更新（终局配置 β.25、far/near 分区 L2、无「E4c negative」小字注记——那是脚注不是图内容） |
| fig2_e3_subspace | tail32 子空间质量（E3） | 留但合并 | 与 E73 互补机理（0.576/0.383/0.796）合并为一张两 panel 图——贡献 #1 的核心证据图 |
| fig3_e4c_strict_budget | 严格预算候选质量（E4c） | 留 | house style 统一（可并入消融图族） |
| fig4_e6_layer_skip | D' 层跳过轮廓（E6） | 留 | gate 是「保守开关」定位，图随叙事降级为支撑图 |
| fig9_e2e_boundaries | e2e 收益区边界 | 留（速度故事核心） | 需更新至 #65 终值（1.285×） |
| fig10_m8_kernels | M8 kernel 微基准 | 留 | house style 统一 |

## 未引用但值得进的图

| 候选 | 论点 | 判决 |
|---|---|---|
| e64a_alpha_curves / beta_curves | α/β 平坦性（E75 无杠杆的支撑证据） | 选一张进消融（β 曲线更相关） |
| e64g_grid_heatmap | α×β 网格热力图 | 与 α/β 曲线二选一（热力图信息密度更高） |
| e72 trace vs e2e 排名反转 | 口径鸿沟方法论（贡献 #4） | **新做**：五臂双排名反转对比图（trace mass rank vs e2e F1 rank）——目前只有 PNG 在 ref/figs |
| e65_dim_reduction | 降维三段判决 | **新做或改**：E65+E73 合并的降维故事图（tail32 互补 + PCA d16 + 训练投影鸿沟） |
| e74 任务族 Δ | far 让渡边界（β 任务形态依赖） | 可选：Δ 柱状图按族分组 |
| fig1_h1 / fig5_e8 / fig6 / fig8 | 旧叙事图 | 不进（信息已被新图覆盖） |

## 终版图规划（8 图封顶）

1. **Fig1**：TLI 架构图（draw.io 重制，留源）——两级索引 + far/near 分区 + 系统栈
2. **Fig2**：子空间与降维（E3 + E73 两段互补 + E65/E76 PCA）——贡献 #1 证据
3. **Fig3**：严格预算候选质量（E4c）+ B' 分区防挤出
4. **Fig4**：α×β 热力图（平坦性 = E75 判决支撑）
5. **Fig5**：E72 五臂 trace vs e2e 排名反转（口径鸿沟）
6. **Fig6**：D' 层跳过 + gate 梯度（保守开关）
7. **Fig7**：kernel 微基准（M8 + 三方对比）
8. **Fig8**：e2e 收益区边界（1.285× + 长度梯度）

house style：okabe_ito 色盲友好色板（publication-figures skill 的 spec）、
Arial ≥8pt、无多余网格、PDF+PNG 双格式、图内英文（无 CJK 字体）。

## 待办

- [ ] draw.io 架构图（Fig1）+ .drawio 源文件
- [ ] E72 排名反转图新做
- [ ] 降维故事图（等 E76 数据落袋后终版）
- [ ] 全图 house style 统一重制
