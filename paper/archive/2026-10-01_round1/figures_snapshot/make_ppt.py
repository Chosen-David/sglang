# 新版进展汇报 PPT（16:9，python-pptx）
# 数据/图全部来自 two-level-attention/exp/figures 与 results（真实实测）
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

FIG = "/home/wangyuanshuo02/two-level-attention/exp/figures"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/figures/TLI_progress_v3.pptx"

# 配色
BLUE = RGBColor(0x2F, 0x6F, 0x9F)
RED = RGBColor(0xC1, 0x44, 0x3C)
GREEN = RGBColor(0x3A, 0x7D, 0x44)
GRAY = RGBColor(0x55, 0x55, 0x55)
DARK = RGBColor(0x22, 0x22, 0x22)
LIGHT = RGBColor(0xEF, 0xF3, 0xF8)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def title_bar(s, text, sub=None, color=BLUE):
    box = s.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.9))
    tf = box.text_frame
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(28); r.font.bold = True; r.font.color.rgb = color
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run(); r2.text = sub
        r2.font.size = Pt(13); r2.font.color.rgb = GRAY


def bullet_box(s, x, y, w, h, items, fs=15):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        if isinstance(it, tuple):
            txt, hl = it
        else:
            txt, hl = it, DARK
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        r = p.add_run(); r.text = "• " + txt
        r.font.size = Pt(fs); r.font.color.rgb = hl
        p.space_after = Pt(6)
    return box


def pic(s, name, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    s.shapes.add_picture(os.path.join(FIG, name), Inches(x), Inches(y), **kw)


def stat_chip(s, x, y, w, h, big, small, color=BLUE):
    from pptx.enum.shapes import MSO_SHAPE
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = LIGHT
    shp.line.color.rgb = color; shp.line.width = Pt(1.5)
    tf = shp.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = big
    r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = color
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = small
    r2.font.size = Pt(11); r2.font.color.rgb = GRAY


# ============ S1 封面 ============
s = slide()
box = s.shapes.add_textbox(Inches(1), Inches(2.2), Inches(11.3), Inches(2.8))
tf = box.text_frame
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "TLI: Two-Level Indexer"
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = BLUE
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run(); r2.text = "Training-free 稀疏注意力索引器：A 子空间 + B' 分区预算 + D' 层跳过"
r2.font.size = Pt(18); r2.font.color.rgb = DARK
p3 = tf.add_paragraph(); p3.alignment = PP_ALIGN.CENTER
r3 = p3.add_run(); r3.text = "进展汇报 v2 ｜ Qwen3-8B 真实权重 × 10 条真实 trace × 2×H20 ｜ 2026-09-23"
r3.font.size = Pt(13); r3.font.color.rgb = GRAY

# ============ S2 论文定位与贡献 ============
s = slide()
title_bar(s, "论文定位与贡献", "全部结论有真实 trace 实测支撑（脚本+数据可复现）")
bullet_box(s, 0.6, 1.4, 7.4, 5.6, [
    ("A 位置稳定子空间粗筛：L1 块 min/max 只在低频尾维 d'=32 计算（4× 削减）", BLUE),
    "E3: mass recall 0.729 ≈ 全维 0.732；random 0.32 / highfreq 0.14 崩溃",
    ("B' far/near 分区 L2 预算：远端独立 top-K2 池，防被近端高分挤出", GREEN),
    "E5b debug: far-heavy 层 TLI 0.9995/0.9997 vs TIA 0.9990/0.9929",
    ("D' 离线校准层跳过：13/36 层免远端检索（静态掩码，零在线开销）", RED),
    "E6: precision 0.92–1.00，far 质量损失 <0.3%",
    "索引侧收益（E8）: FLOP 2.57×（跳层 4.88×）；Triton fused L1 kernel 3.6×/1.6×",
    "Negative results 护城河：严格预算下聚类代表无优势（E4c）+ 在线信号失效（E5）",
], fs=14)
stat_chip(s, 8.6, 1.6, 4.2, 1.5, "2.57×", "索引 FLOP 全层平均（E8-1）", BLUE)
stat_chip(s, 8.6, 3.3, 4.2, 1.5, "0.9998", "TLI mass 覆盖（36 层均值，vs TIA 0.9990）", GREEN)
stat_chip(s, 8.6, 5.0, 4.2, 1.5, "3.6×", "fused L1 kernel vs eager（S=128K）", RED)

# ============ S3 H1 动机 ============
s = slide()
title_bar(s, "动机：注意力的真实形态是三层分解（H1）", "「近端集中」叙事在 mass 口径下不成立；dense top-1024 mass 覆盖恒为 1.0000")
pic(s, "fig1_h1_decomposition.png", 0.7, 1.5, w=7.8)
bullet_box(s, 8.9, 1.7, 4.0, 5.0, [
    "sink（前 64 tok）：0.37–0.71 质量大头",
    "near（末 2048 tok）：0.16–0.29",
    "far（中间区）：0.02–0.29，任务相关",
    ("TIA@1024≈FullKV 根因：dense top-1024 已覆盖全部 mass", RED),
    "→ 论文动机：far 层的挤出风险 + 索引开销，而非「召回远端」",
], fs=13)

# ============ S4 A 子空间 ============
s = slide()
title_bar(s, "创新点 A：位置稳定子空间粗筛", "E3: 低频尾维 d'=32 ≈ 全维 128（选错维度崩溃）")
pic(s, "fig2_e3_subspace.png", 1.2, 1.6, w=10.8)
stat_chip(s, 0.8, 6.1, 3.8, 1.1, "0.729 vs 0.732", "d'=32 vs d'=128 mass recall", GREEN)
stat_chip(s, 5.0, 6.1, 3.8, 1.1, "4×", "L1 索引维度（HBM 同步减）", BLUE)
stat_chip(s, 9.2, 6.1, 3.8, 1.1, "0.32 / 0.14", "random / highfreq 子空间（崩溃）", RED)

# ============ S5 B' 重定位（E4c） ============
s = slide()
title_bar(s, "创新点 B'：分区预算（含 E4c 口径修正）", "E4b 的「kmeans 4–10×」含整簇超选 bug：budget=512 实取 2769 tok", RED)
pic(s, "fig3_e4c_strict_budget.png", 0.7, 1.7, w=8.6)
bullet_box(s, 9.6, 1.8, 3.4, 5.2, [
    "严格预算 + per-head 加权重测（E4c）：",
    ("km_blk 0.09–0.39（最差）", RED),
    "km_tok 0.45–0.79（无一致优势）",
    ("4bit token 级精筛 ≈ oracle 0.999", GREEN),
    "→ B 重定位为 far/near 分区预算",
    "聚类降级为消融 + negative result",
], fs=12.5)

# ============ S6 D' 层跳过 ============
s = slide()
title_bar(s, "创新点 D'：离线校准层跳过", "far mass 层轮廓双峰 → 静态掩码；在线信号版已实测否定")
pic(s, "fig4_e6_layer_skip.png", 0.9, 1.6, w=7.6)
bullet_box(s, 8.9, 1.8, 4.0, 5.0, [
    "13/36 层可跳（τ=0.02）",
    "precision 0.92–1.00",
    "far 质量损失 <0.3%",
    ("在线 L1 信号与 far 质量相关 ≈0（E5）", RED),
    "→ D' 必须离线校准（结论写入 negative results）",
], fs=13)

# ============ S7 架构图 ============
s = slide()
title_bar(s, "TLI 架构", "A + B' + D' 数据流（全部继承 TIA 两级语义）")
pic(s, "fig7_tli_architecture.png", 0.8, 1.4, w=11.8)

# ============ S8 E8 开销 ============
s = slide()
title_bar(s, "索引开销与 fused kernel 原型（E8）", "D' topk 截断到有效块数 = 真正兑现 L2 计算节省")
pic(s, "fig5_e8_speedup.png", 1.0, 1.6, w=11.2)
bullet_box(s, 1.0, 5.9, 11.5, 1.4, [
    ("E8-1: 跳层 4.88× / 非跳层 1.27×（A 子空间贡献）/ 全层平均 2.57×；attention 主计算量不变（K2 固定）", DARK),
    "E8-2: L1 fused 3.6×/1.6×（块 id 对拍一致）+ L2 级联 fused 1.63×（分区 topk 对拍 4096/4096）——完整两级仅 2 launches",
], fs=13)

# ============ S8b M3 系统集成 ============
s = slide()
title_bar(s, "M3：sglang 系统集成实测（2026-09-24，H20 + Qwen3-8B 真实上下文）",
          "tli backend 全链路：fused kernel 对拍精确一致 + e2e 逐字一致 + 高并发曲线暴露批量化缺口")
pic(s, "fig8_m3_system.png", 0.6, 1.35, w=12.2)
bullet_box(s, 0.6, 5.95, 12.2, 1.35, [
    ("L2 级联 fused kernel 接入：池边界即因果边界 + 滑窗精确复制；对拍 jaccard 1.0000（S=9.9K/131K），e2e 输出与 eager 逐字一致", GREEN),
    ("增量索引预分配几何扩容（cat 版 S=131K 时 4.8GB/步 memcpy）→ update 0.128ms 与 S 无关，增量==全量仍逐位一致", DARK),
    ("结构性发现：forward_decode 逐请求 Python 循环 → ~38ms/req/step 线性放大（triton 批量化仅 3.2×）→ M4 批量化 = 吞吐主表前置条件", RED),
], fs=11.5)

# ============ S9 TLI vs TIA 逐层 ============
s = slide()
title_bar(s, "TLI vs TIA 逐层质量（E5b debug，hotpotqa）", "全 36 层 diff < 0.001；far-heavy 层反超（分区防挤出）")
pic(s, "fig6_tli_vs_tia_layers.png", 1.5, 1.6, w=10.2)
stat_chip(s, 1.5, 6.2, 3.6, 1.0, "36/36 层", "diff < 0.001 vs TIA", GREEN)
stat_chip(s, 5.5, 6.2, 3.6, 1.0, "+0.0068", "L05 反超 TIA（0.9997 vs 0.9929）", BLUE)

# ============ S10 E5b 主表 ============
s = slide()
title_bar(s, "E5b：LongBench 13 子集主表（已完成）", "TLI −0.14 vs TIA / −0.44 vs FullKV，同时索引 FLOP 2.57×（E8-1）")
box = s.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.7), Inches(4.3))
tf = box.text_frame
tf.word_wrap = True
rows = [
    ("task", "FullKV", "Quest", "TIA", "TWI", "TLI"),
    ("hotpotqa", "53.48", "45.74", "53.89", "48.98", "53.96"),
    ("2wikimqa", "38.29", "38.46", "38.27", "36.16", "39.07"),
    ("musique", "32.14", "27.25", "32.28", "23.60", "31.35"),
    ("passage_ret_en", "100.0", "98.50", "99.50", "98.50", "100.0"),
    ("qasper", "44.17", "40.13", "44.03", "38.96", "44.03"),
    ("multifieldqa_en", "53.40", "51.01", "53.19", "48.11", "52.98"),
    ("gov_report", "33.17", "32.13", "33.43", "33.09", "32.41"),
    ("qmsum", "23.53", "22.21", "23.90", "23.98", "22.77"),
    ("multi_news", "24.93", "24.95", "24.73", "24.99", "24.66"),
    ("narrativeqa", "25.61", "20.64", "22.18", "26.19", "23.14"),
    ("triviaqa", "90.71", "87.55", "89.82", "89.59", "90.22"),
    ("lcc", "68.81", "68.34", "69.14", "66.22", "68.74"),
    ("repobench-p", "66.50", "63.41", "66.40", "63.87", "65.59"),
    ("AVG", "50.36", "47.72", "50.06", "47.86", "49.92"),
]
first = True
for row in rows:
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    first = False
    is_head = row[0] == "task"
    is_avg = row[0] == "AVG"
    r = p.add_run()
    r.text = f"{row[0]:16s}" + "".join(f"{v:>8s}" for v in row[1:])
    r.font.size = Pt(12.5)
    r.font.bold = is_head or is_avg
    r.font.name = "Consolas"
    if is_avg:
        r.font.color.rgb = BLUE if False else DARK
        # TLI 列高亮
        r2 = p.add_run(); r2.text = "  ← TLI"
        r2.font.size = Pt(11); r2.font.color.rgb = GREEN; r2.font.bold = True
    elif row[0] == "TLI" and not is_head:
        pass
    p.space_after = Pt(2)
bullet_box(s, 0.8, 5.95, 11.7, 1.3, [
    ("TLI = A+B'+D'-gated：far 总量 <0.3 的任务启用 D' 层掩码（4.88× 索引省算），多跳任务（far 0.55-0.95）不跳层", GREEN),
    ("本轮 negative result：D' 全局掩码跨任务不泛化（musique −4.83 等）；per-task 重校准也失败（长度失配）→ far 总量 gate 是正确设计", RED),
], fs=12)

# ============ S11 Negative results ============
s = slide()
title_bar(s, "Negative Results（论文护城河）", "每一项都有实测编号与脚本", RED)
bullet_box(s, 0.7, 1.5, 12.0, 5.6, [
    ("E4c：严格 token 预算下，kmeans 聚类代表（块 scatter-amax 0.09–0.39 / token topk 0.45–0.79）无一致优势", DARK),
    "   ↳ E4b 的 4–10× 占优含整簇超选 bug（5.4× 超预算）+ head 平均口径虚高——口径陷阱本身是方法论贡献",
    "E5：跨层 mask 复用 IoU 0.33；decode 步间 churn 22%；在线 L1 信号与 far 质量相关 ≈ 0",
    "   ↳ 否定「跨层复用」「增量 topk」「在线自适应」三条捷径，确立 D' 必须离线",
    "H1：dense top-1024 mass 覆盖恒为 1.0000",
    "   ↳ 该领域的 recall 口径会系统性高估稀疏方法差距，mass + 端到端分数才是有效口径",
], fs=14.5)

# ============ S12 待办 ============
s = slide()
title_bar(s, "待办与计划", "优先级序（M3 系统集成已完成 → 剩批量化 + H100 + 写作）")
bullet_box(s, 0.7, 1.4, 12.0, 5.8, [
    "① ~~E5b 主表~~ ✅ 49.92（−0.14 vs TIA）+ D' far 总量 gate 设计修正 + negative results 入册",
    "② ~~sglang M2/M3~~ ✅ paged 寻址 + O(n) 增量索引 + 稀疏 prefill + L1/L2 fused kernel 接入（对拍 1.0000 / e2e 逐字一致，6 commits）",
    "③ **M4 批量化 decode（当前主线）**：共享 index pool + kq 真 4bit 存储 + L1/L2 kernel grid 加 batch 维 + _sparse_attn 批量 gather + CUDA graph",
    "④ ~~消融表~~ ✅（trace 级四组合 + far_tokens 敏感性 + D' 三任务消融 3/3）",
    "⑤ ~~Qwen3-32B 泛化复验~~ ✅（A Go 强度递减如实报告 / D' 更强 41/64 层 precision 0.993 / gate 判据入 negative results）",
    "⑥ H100 吞吐主表（机器申请中；H20 已备好算力无关性论证层）",
    "⑦ 论文写作（骨架见 TWO_LEVEL_PAPER_REPORT.md，主表已齐）",
], fs=14)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
