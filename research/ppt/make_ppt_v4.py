# v4 进展汇报 PPT（16:9，python-pptx）——论文视角重组：
# 每个创新点一页讲清「是什么 + 探索到哪了（含最新证据）」，系统侧 M3–M7
# 数据融入对应创新点/系统贡献页，不按里程碑流水加页。
# 数据来源：two-level-attention/exp/figures + results、sglang TWO_LEVEL_PAPER_REPORT.md
# §4/§7/§8/§8b-2..§8b-5、tli_dim_sweep.json、tli_m5_e2e_results.json（全部实测）
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# 图资产只读自导师仓库（fig1-8 的 PDF/PNG 在那边）；输出写 sglang 目录
# （用户指示：工作目录 = ~/sglang，~/two-level-attention 只读不动）
FIG = "/home/wangyuanshuo02/two-level-attention/exp/figures"
OUT = "/home/wangyuanshuo02/sglang/TLI_progress_v4.pptx"

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


def mono_table(s, x, y, w, h, rows, fs=12.5):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    n = len(rows)
    for i, row in enumerate(rows):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        is_head, is_last = i == 0, i == n - 1
        r = p.add_run()
        r.text = f"{row[0]:22s}" + "".join(f"{v:>10s}" for v in row[1:])
        r.font.size = Pt(fs)
        r.font.name = "Consolas"
        r.font.bold = is_head or is_last
        r.font.color.rgb = BLUE if is_head else (RED if is_last else DARK)
        p.space_after = Pt(2)


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
r3 = p3.add_run(); r3.text = "进展汇报 v4 ｜ Qwen3-8B/32B 真实权重 × 真实 trace × 2×H20 ｜ 2026-09-25"
r3.font.size = Pt(13); r3.font.color.rgb = GRAY

# ============ S2 论文定位与贡献（四贡献 = 三个算法创新点 + 一个系统贡献） ============
s = slide()
title_bar(s, "论文定位与贡献", "三个算法创新点（全部 Go 且边界已画出）+ 一个系统贡献（sglang 全集成）")
bullet_box(s, 0.6, 1.4, 7.4, 5.6, [
    ("A 位置稳定子空间粗筛：L1 块 min/max 只在低频尾维计算 → 维度压缩边界已实测（M6）", BLUE),
    "8B/32B 双规模验证：机制存在、强度递减（诚实泛化结论）",
    ("B' far/near 分区 L2 预算：防远端被近端挤出（E4c 重定位后的正确形态）", GREEN),
    "far-heavy 层反超 TIA；far_tokens 128–256 即饱和",
    ("D' 离线校准层跳过：跳 13/36（32B 41/64）层，far 质量损失 <0.3%", RED),
    "gate 设计经两代模型验证，判据失败模式入 negative results",
    ("S 系统贡献：sglang 全链路原型→生产级（M3–M7），decode 6.6× + prefill 2.1× + 4bit 存储", BLUE),
    "对拍文化：replay 逐位 0.00e+00、e2e 逐字一致、集合 32/32 一致",
    "Negative results 护城河：聚类代表/在线信号/跨层复用/维度可压性边界",
], fs=13.5)
stat_chip(s, 8.6, 1.6, 4.2, 1.5, "2.57×", "索引 FLOP 全层平均（E8-1）", BLUE)
stat_chip(s, 8.6, 3.3, 4.2, 1.5, "6.6×", "sglang decode（M3 原型→M5，bs=32）", RED)
stat_chip(s, 8.6, 5.0, 4.2, 1.5, "3.2×", "kq 存储压缩（M6 4bit，零漂移）", GREEN)

# ============ S3 动机 H1 + 评估口径升级 ============
s = slide()
title_bar(s, "动机：注意力的真实形态是三层分解（H1）", "「近端集中」叙事不成立；评估口径升级——总 mass 被 sink 稀释，竞争区才是区分度")
pic(s, "fig1_h1_decomposition.png", 0.7, 1.5, w=7.8)
bullet_box(s, 8.9, 1.7, 4.0, 5.2, [
    "sink（前 64 tok）：0.37–0.71 质量大头",
    "near（末 2048 tok）：0.16–0.29",
    "far（中间区）：0.02–0.29，任务相关",
    ("总 mass 口径陷阱：竞争区（去 sink+滑窗）仅占总 mass 0.515", RED),
    "→ 论文质量口径改报剩余 mass coverage（0.953），区分度显著更高",
], fs=12.5)

# ============ S4 创新点 A：是什么 + 探索到哪了 ============
s = slide()
title_bar(s, "创新点 A：位置稳定子空间粗筛——探索已收敛，边界已画出", "是什么：L1 只在 RoPE 低频尾维算 min/max 上界；探索：维度能压到哪、跨规模是否成立")
pic(s, "fig2_e3_subspace.png", 0.7, 1.4, w=7.4)
bullet_box(s, 8.5, 1.6, 4.4, 5.4, [
    ("8B（E3）：d'=32 recall 0.729 ≈ 全维 0.732；random/highfreq 崩溃", GREEN),
    ("32B 泛化：lowfreq32 0.66–0.87 ≈ full128（差 ≤0.08）＞ random ＞＞ hifreq——机制存在、强度递减", DARK),
    ("M6 新证据（维度压缩扫描，真实 trace）：", BLUE),
    ("  L1 粗筛 d'=32→16：质量逐位不变（选中块集 172→194 块，L2 精筛吸收）→ 可再省一半索引存储/流量", GREEN),
    ("  L2 细筛 δ=16→8：cov剩 0.953→0.782（far-heavy 层最差 0.34）→ 不可压", RED),
    ("结论：两级索引维数压缩边界画在 L1/L2 之间——粗筛上界可粗，细筛分数必须精（论文可直接引用的边界结论）", DARK),
], fs=12)
stat_chip(s, 0.9, 6.35, 3.6, 1.0, "d'=16 免费", "L1 维度再减半（M6 实测）", GREEN)
stat_chip(s, 4.8, 6.35, 3.6, 1.0, "δ=8 崩溃", "L2 维度不可压（M6 实测）", RED)

# ============ S5 创新点 B'：是什么 + 探索到哪了 ============
s = slide()
title_bar(s, "创新点 B'：far/near 分区预算——重定位后的正确形态", "是什么：远端独立 top-K2 池防挤出；探索：聚类代表（原方案）被严格预算口径否定", RED)
pic(s, "fig3_e4c_strict_budget.png", 0.7, 1.7, w=8.6)
bullet_box(s, 9.6, 1.8, 3.4, 5.2, [
    "原方案（聚类代表）：严格预算 + per-head 口径下",
    ("  km_blk 0.09–0.39（最差）/ km_tok 无一致优势", RED),
    ("  4bit token 级精筛 ≈ oracle 0.999（TIA 原生）", GREEN),
    "→ B 重定位为分区预算（工程正确形态）",
    "far_tokens 敏感性：128–256 饱和",
    "far-heavy 层 TLI 反超 TIA（L05 +0.0068）",
    "聚类降级为消融 + negative result",
], fs=11.5)

# ============ S6 创新点 D'：是什么 + 探索到哪了 ============
s = slide()
title_bar(s, "创新点 D'：离线校准层跳过——跨规模复验成立，gate 设计两代验证", "是什么：far 质量低的层免远端检索（静态掩码）；探索：怎么 gate、跨模型是否泛化")
pic(s, "fig4_e6_layer_skip.png", 0.9, 1.6, w=7.6)
bullet_box(s, 8.9, 1.8, 4.0, 5.2, [
    "8B：跳 13/36 层，precision 0.92–1.00",
    ("32B 泛化更强：跳 41/64 层（64%），precision 0.993", GREEN),
    "far 层轮廓双峰跨模型同构（可跳层空间随深度增大）",
    ("gate 探索史（negative results 资产）：", RED),
    "  在线信号 corr≈0（E5）→ 必须离线",
    "  far 总量判据两代模型均重叠 → 口径混淆",
    "  真根因 = 校准集层轮廓与任务错位（corr 0.05–0.89）",
    "→ 最终形态：per-task 层轮廓校准（同长度同分布 trace）",
], fs=12)

# ============ S7 架构图 ============
s = slide()
title_bar(s, "TLI 架构", "A + B' + D' 数据流（全部继承 TIA 两级语义）+ sglang 系统栈")
pic(s, "fig7_tli_architecture.png", 0.8, 1.4, w=11.8)

# ============ S8 创新点 S（系统贡献）：从原型到生产级引擎 ============
s = slide()
title_bar(s, "系统贡献：sglang 全链路集成（M3–M7）——decode 6.6×，prefill 2.1×",
          "叙事：M3 曲线暴露「原型→生产级」工程鸿沟 → 批量化→图化→4bit→prefill，每步对拍零漂移", GREEN)
mono_table(s, 0.7, 1.5, 7.6, 2.5, [
    ("decode 阶梯", "bs=1", "bs=8", "bs=16", "bs=32"),
    ("M3 原型", "51.4", "369.7", "666.4", "1236.8"),
    ("M5 graph", "17.7-36.8", "70.3-75.5", "87.5-115.2", "188.6"),
    ("triton graph", "7.8", "12.8", "18.7", "31.7"),
], fs=12.5)
stat_chip(s, 8.7, 1.6, 4.2, 1.15, "6.6×", "decode 累计（bs=32，tok/s 25.9→169.7）", RED)
stat_chip(s, 8.7, 2.9, 4.2, 1.15, "2.08×", "prefill（select 快路径 5.2×@10K）", BLUE)
bullet_box(s, 0.7, 4.3, 12.2, 2.9, [
    ("M4 批量化：共享 index pool + 批量 select/增量（launch 数与 bs 无关）；线性项 38→5.2 ms/req", DARK),
    ("M5 CUDA graph：图内统一增量/稀疏路径零 host 同步（真 capture 通过）+ 短行 veto 回退；replay vs eager 逐位 0.00e+00", DARK),
    ("M6 kq 真 4bit：uint8+scale 三张量，128→40 B/token-head，重建≡量化（逐位零漂移）——S=131K 显存前提解锁", DARK),
    ("M7 prefill：归因修正（瓶颈是 select 而非 build）→ 池==因果区快路径；prefill 延迟随 S 亚线性（9.5→62.1 ms/层 @5K→130K）", DARK),
    ("诚实口径：S=10K 档仍慢于 triton 全量——稀疏收益在长 S 的 HBM 流量（bs=32×131K 时 triton 需 ~620GB/步纯 HBM），S=131K 主表待 M8/H100", RED),
], fs=11.5)

# ============ S9 索引开销与 kernel（E8） ============
s = slide()
title_bar(s, "索引开销与 fused kernel 原型（E8）", "D' topk 截断到有效块数 = 真正兑现 L2 计算节省")
pic(s, "fig5_e8_speedup.png", 1.0, 1.6, w=11.2)
bullet_box(s, 1.0, 5.9, 11.5, 1.4, [
    ("E8-1: 跳层 4.88× / 非跳层 1.27×（A 子空间贡献）/ 全层平均 2.57×；attention 主计算量不变（K2 固定）", DARK),
    "E8-2: L1 fused 3.6×/1.6×（块 id 对拍一致）+ L2 级联 fused 1.63×（分区 topk 对拍 4096/4096）——完整两级仅 2 launches",
], fs=13)

# ============ S10 TLI vs TIA 逐层质量 ============
s = slide()
title_bar(s, "TLI vs TIA 逐层质量（E5b debug，hotpotqa）", "全 36 层 diff < 0.001；far-heavy 层反超（分区防挤出）")
pic(s, "fig6_tli_vs_tia_layers.png", 1.5, 1.6, w=10.2)
stat_chip(s, 1.5, 6.2, 3.6, 1.0, "36/36 层", "diff < 0.001 vs TIA", GREEN)
stat_chip(s, 5.5, 6.2, 3.6, 1.0, "+0.0068", "L05 反超 TIA（0.9997 vs 0.9929）", BLUE)

# ============ S11 E5b 主表 ============
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
        r2 = p.add_run(); r2.text = "  ← TLI"
        r2.font.size = Pt(11); r2.font.color.rgb = GREEN; r2.font.bold = True
    p.space_after = Pt(2)
bullet_box(s, 0.8, 5.95, 11.7, 1.3, [
    ("TLI = A+B'+D'-gated：far 总量 <0.3 的任务启用 D' 层掩码（4.88× 索引省算），多跳任务不跳层", GREEN),
    ("negative result：D' 全局掩码跨任务不泛化（musique −4.83 等）；per-task 重校准也失败（长度失配）→ far 总量 gate 是正确设计", RED),
], fs=12)

# ============ S12 Negative results ============
s = slide()
title_bar(s, "Negative Results（论文护城河）", "每一项都有实测编号与脚本", RED)
bullet_box(s, 0.7, 1.5, 12.0, 5.6, [
    ("E4c：严格 token 预算下，kmeans 聚类代表（块 0.09–0.39 / token 0.45–0.79）无一致优势", DARK),
    "   ↳ E4b 的 4–10× 占优含整簇超选 bug（5.4× 超预算）+ head 平均口径虚高——口径陷阱本身是方法论贡献",
    "E5：跨层 mask 复用 IoU 0.33；decode churn 22%；在线 L1 信号与 far 质量相关 ≈ 0",
    "   ↳ 否定「跨层复用」「增量 topk」「在线自适应」三条捷径，确立 D' 必须离线",
    "H1/M4：dense top-1024 mass 覆盖恒为 1.0000；竞争区仅占总 mass 0.515",
    "   ↳ recall/mass 总口径系统性高估差距 → 论文改报剩余 mass coverage（0.953）+ 端到端分数",
    ("M6：L2 细筛维数不可压（δ→8 时 cov剩 0.953→0.782）——与聚类路线「细筛必须保全全维」独立同构，两级系统普适教训", DARK),
    ("M3-b：L2 级联纯 kernel 化 No-Go——L1 能容忍多选（下游吸收）、最末级不能（无兜底）→ topk 留给 torch 是正确分工", DARK),
], fs=13)

# ============ S13 待办 ============
s = slide()
title_bar(s, "待办与计划", "算法侧三个创新点全部收敛 → 剩系统侧收尾 + H100 主表 + 写作")
bullet_box(s, 0.7, 1.4, 12.0, 5.8, [
    ("算法侧 ✅：A（双规模泛化 + 维度边界）/ B'（重定位 + 敏感性）/ D'（跨规模 + gate）——探索已收敛，结论可写", GREEN),
    ("系统侧 ✅：M3–M7 落地（decode 6.6× + prefill 2.1× + 4bit 存储 3.2×），每步对拍零漂移，10 commits", GREEN),
    ("⑤ M8 H 卡优化（下一步）：TMA/Tensor Descriptor gather（_sparse_extend 随机行 gather 现仅 576GB/s）+ 141GB 大显存轴（bs=64×S=131K）", RED),
    "⑥ H100 吞吐主表（机器申请中）+ RULER/NIAH 补评测（对齐 Quest/SnapKV/HISA 论文口径）",
    "⑦ S=131K 长上下文主表（M6 已解锁显存前提；bs=32×131K 稀疏 1/8 HBM 流量的收益兑现位）",
    "⑧ 论文写作（骨架见 TWO_LEVEL_PAPER_REPORT.md；主表 + 边界结论 + negative results 已齐）",
], fs=14)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
