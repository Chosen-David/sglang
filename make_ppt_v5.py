# v5 进展汇报 PPT（16:9，python-pptx）
# 相对 v4 的两类更新：
# ① 采纳 GPT 批判性意见（TLI_progress_v4_critical_rework.pptx，19 页复盘）的合理项：
#    - claim 收紧：质量=守恒（≈TIA），主卖点=性能（同精度下索引侧降本+kernel 加速）
#    - 系统贡献分层表述：6.6× 是「原型→生产级」工程进步，vs triton 的 10K 档差距如实保留
#    - 负向任务逐项诊断（musique/gov_report/qmsum/repobench-p 不能只报平均）
#    - A 表述升级为「非对称压缩定律」（L1 可粗 L2 必精）
#    - negative results 逐条映射 final design decision
#    - venue 定位 MLSys/ASPLOS/SC（算法边界+系统集成+负结果）；Go/No-Go 硬阈值
# ② 本轮新实测数据（GPT 复盘时没有的）：
#    - 同机 kernel 级三方对比（NEW S9）：Quest 官方 decode_select_k + page GEMV / DSA 官方
#      tilelang fp8_index / TLI 两级 select 全链路——同机同口径诚实报告
#    - L1 分区语义澄清（far 区即 L1 唯一服务对象，near 不进 L1）
# 数据来源：two-level-attention/exp/figures、TWO_LEVEL_PAPER_REPORT.md、
# kernel_comparison_indexers.json（本轮实测）、dsa_vs_tli_indexer.json
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# 图资产只读自导师仓库；输出写 sglang 目录
FIG = "/home/wangyuanshuo02/two-level-attention/exp/figures"
OUT = "/home/wangyuanshuo02/sglang/TLI_progress_v5.pptx"

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
        r = p.add_run(); r.text = ("• " if not txt.startswith(" ") else " ") + txt
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


def mono_table(s, x, y, w, h, rows, fs=12.5, first_col=22):
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
        r.text = f"{row[0]:{first_col}s}" + "".join(f"{v:>10s}" for v in row[1:])
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
r2 = p2.add_run(); r2.text = "质量守恒（≈TIA）前提下的索引侧降本：A 子空间 + B' 分区预算 + D' 层跳过"
r2.font.size = Pt(18); r2.font.color.rgb = DARK
p3 = tf.add_paragraph(); p3.alignment = PP_ALIGN.CENTER
r3 = p3.add_run(); r3.text = "进展汇报 v5（含批判性复盘响应 + 同机 kernel 三方对比）｜ Qwen3-8B/32B × 真实 trace × sglang ｜ 2026-09-25"
r3.font.size = Pt(13); r3.font.color.rgb = GRAY

# ============ S2 论文定位与贡献（claim 收紧版） ============
s = slide()
title_bar(s, "论文定位与贡献（claim 收紧后）",
          "质量=守恒而非胜出（TLI 49.92 ≈ TIA 50.06，−0.44 vs FullKV）；主卖点=同精度下索引侧降本 + 系统加速", GREEN)
bullet_box(s, 0.6, 1.4, 7.4, 5.6, [
    ( "A 位置稳定子空间粗筛（低频尾维）→ 两级索引的「非对称压缩定律」：L1 可粗、L2 必精", BLUE),
    "8B/32B 双规模验证：机制存在、强度递减（诚实泛化结论）；L1 d'=16 免费 / L2 δ=8 崩溃（边界已实测）",
    ( "B' far/near 分区 L2 预算：远端独立配额防挤出；聚类代表降级为 negative result", GREEN),
    "far_tokens 128–256 饱和；far-heavy 层反超 TIA（L05 +0.0068）",
    ( "D' 离线校准层跳过：跳 13/36（32B 41/64）层；gate 泛化是最大风险 → held-out 验证列入待办", RED),
    ( "S 系统贡献（分层表述）：索引侧 2.57× FLOP + sglang 原型→生产级（M3–M7）；e2e 胜利位=131K/大 batch HBM 流量，非 10K 档", BLUE),
    "对拍文化：replay 逐位 0.00e+00、e2e 逐字一致、集合 32/32 一致",
    "Negative results = 护城河：每一项映射一个 final design decision",
    ( "Venue 定位：MLSys/ASPLOS/SC 风（算法边界 + 系统集成 + 负结果），非纯精度论文", DARK),
], fs=13)
stat_chip(s, 8.6, 1.6, 4.2, 1.5, "2.57×", "索引 FLOP 全层平均（E8-1）", BLUE)
stat_chip(s, 8.6, 3.3, 4.2, 1.5, "6.6×", "decode 原型→生产级（M3→M5，bs=32）", RED)
stat_chip(s, 8.6, 5.0, 4.2, 1.5, "32×↓", "TLI vs DSA 每 token 索引 MAC（本轮实测口径）", GREEN)

# ============ S3 动机 H1 + 评估口径三件套 ============
s = slide()
title_bar(s, "动机：注意力的真实形态是三层分解（H1）",
          "总 mass 口径被 sink 稀释 → 论文质量口径三件套：剩余 mass coverage + 远端 recall + 端到端分数")
pic(s, "fig1_h1_decomposition.png", 0.7, 1.5, w=7.8)
bullet_box(s, 8.9, 1.7, 4.0, 5.2, [
    "sink（前 64 tok）：0.37–0.71 质量大头",
    "near（末 2048 tok）：0.16–0.29",
    "far（中间区）：0.02–0.29，任务相关",
    ( "口径陷阱：dense top-1024 mass 覆盖恒 1.0000；竞争区仅占总 mass 0.515", RED),
    "→ 主口径=剩余 mass coverage（0.953）+ 远端 recall 单独报",
    "→ 补 RULER/NIAH/passkey 直接压力测试 far 检索（待办）",
], fs=12.5)

# ============ S4 创新点 A：非对称压缩定律 ============
s = slide()
title_bar(s, "创新点 A：非对称压缩定律——L1 可粗；L2 瓶颈在「表示」不在「维数」",
          "本轮扩展（300 行 + 机制分离 + 投影对照）：维度选择口径 L2 不可压；投影表示口径 L2 也可压")
pic(s, "fig2_e3_subspace.png", 0.7, 1.4, w=7.4)
bullet_box(s, 8.5, 1.5, 4.4, 5.5, [
    ("8B（E3）：d'=32 recall 0.729 ≈ 全维；random/highfreq 崩溃；32B 泛化强度递减", DARK),
    ("扩展测试（5 trace×5 层×3 t）：选择口径 δ=16→8 far rec 0.557→0.378；fp32 vs 4bit 仅差 2–5% → 病因=维度非量化", DARK),
    ( "投影对照（NEW）：PCA16 0.532 ≈ 选择32 0.557（打分维度砍半）；PCA32 0.682 = +0.125；跨任务 PCA 基迁移仅 −0.024（离线校准同 D' 哲学）", GREEN),
    ( "随机投影崩溃（0.147/0.246）：JL 保范数不保 GQA 求和点积排序——K 协方差结构是本质", RED),
    ( "修正结论：L2 的信息瓶颈在表示方式——投影可压、选择不可压；打分 FLOP 可再减半（工程集成待办）", BLUE),
], fs=10.5)
stat_chip(s, 0.9, 6.35, 3.6, 1.0, "d'=16 免费", "L1 维度再减半（M6）", GREEN)
stat_chip(s, 4.8, 6.35, 3.6, 1.0, "PCA16≈选32", "投影表示同质砍半维（本轮）", BLUE)

# ============ S5 创新点 B' ============
s = slide()
title_bar(s, "创新点 B'：far/near 分区预算——机制已实测定位（本轮消融）", "GPT 质询「far 在 L1 被剪掉 L2 救不回」→ 三版本消融 + survival curve 定量回答", RED)
pic(s, "fig3_e4c_strict_budget.png", 0.7, 1.7, w=8.6)
bullet_box(s, 9.6, 1.5, 3.4, 5.5, [
    "三版本消融（5 trace × 5 层 × 2 t）：",
    ( "  L1 存活(oracle far) 0.93–1.00——far 在 L1 被剪不发生", GREEN),
    ( "  hier(L1 分区) No-Go：far 池 94→16 块，mass cov 0.846→0.811", RED),
    ( "  B' 增益机制=近端名额保障（L05 cov剩 0.600→0.726，far mass 同值 0.999）", BLUE),
    "→ 正确形态=分区只在 L2 终选级（当前实现）",
    "far_tokens 128–256 饱和；聚类降级 negative result",
], fs=10.5)
bullet_box(s, 0.7, 5.6, 8.6, 1.6, [
    ( "名额口径陷阱（本轮发现）：global top-K2 下 far 平均占 569/1024 名额（4bit 噪声+基数效应）——far_rec 虚高但 cov剩 反降；B' 把名额分配从「分数噪声驱动」改为「预算驱动」，far 截到 256 质量不损（far mass cov 0.999 同值）", DARK),
], fs=10.5)

# ============ S6 创新点 D' + gate 风险 ============
s = slide()
title_bar(s, "创新点 D'：离线校准层跳过——跨规模复验成立，gate 泛化是最大风险", "强证据：8B 跳 13/36、32B 跳 41/64 precision 0.993；风险：gate 需 held-out 验证去 oracle 化")
pic(s, "fig4_e6_layer_skip.png", 0.9, 1.6, w=7.6)
bullet_box(s, 8.9, 1.8, 4.0, 5.2, [
    "8B：跳 13/36 层，precision 0.92–1.00",
    ( "32B 泛化更强：跳 41/64 层（64%），precision 0.993", GREEN),
    "far 层轮廓双峰跨模型同构",
    ( "失败史（写进论文）：在线信号 corr≈0 / far 总量判据重叠 / 全局掩码跨任务不泛化 / per-task 重校准长度失配", RED),
    "→ 最终 gate：离线 per-task 层轮廓校准（同长度同分布 trace）",
    ( "→ 补 held-out 任务/长度切分验证：报告 gate precision + 误跳损失（待办）", DARK),
], fs=11.5)

# ============ S7 架构图（分区语义澄清） ============
s = slide()
title_bar(s, "TLI 架构", "A + B' + D' 数据流 + 空间分区语义：far 区 [sink, S-2048) 是两级索引唯一服务对象")
pic(s, "fig7_tli_architecture.png", 0.8, 1.4, w=11.8)
bullet_box(s, 0.8, 5.6, 11.7, 1.6, [
    ( "分区语义（源码实证 + 本轮消融）：L1 块筛=全局 top-K1（near 带 34 块也参与竞争，但 far 池保底 ≥94 块，oracle far L1 存活 0.93–1.00）；B' 分区发生在 L2 终选级——far 池 [sink, t+1-2048) 独立 top-256 + 近端带独立 top-768；滑窗强制位不占配额", DARK),
    ( "D' 跳层 = 减少需要远端检索的层数；A = 减少 L1 粗筛读/算；B' = 名额分配从分数噪声驱动改为预算驱动（近端下限保障 + far 截断不损质量）——三个创新点正交于一条审稿故事线", BLUE),
], fs=11.5)

# ============ S8 系统贡献（分层 claim） ============
s = slide()
title_bar(s, "系统贡献：sglang 全链路集成（M3–M7）——分层表述 claim", "可 claim：原型→生产级工程路线（每步对拍零漂移）；不可 claim：10K 档胜过 triton 全量（收益位=长 S HBM）", GREEN)
mono_table(s, 0.7, 1.5, 7.6, 2.5, [
    ("decode 阶梯", "bs=1", "bs=8", "bs=16", "bs=32"),
    ("M3 原型", "51.4", "369.7", "666.4", "1236.8"),
    ("M5 graph", "17.7-36.8", "70.3-75.5", "87.5-115.2", "188.6"),
    ("triton graph", "7.8", "12.8", "18.7", "31.7"),
], fs=12.5)
stat_chip(s, 8.7, 1.6, 4.2, 1.15, "6.6×", "原型→生产级（bs=32，tok/s 25.9→169.7）", RED)
stat_chip(s, 8.7, 2.9, 4.2, 1.15, "2.08×", "prefill（select 快路径 5.2×@10K）", BLUE)
bullet_box(s, 0.7, 4.3, 12.2, 2.9, [
    ("M4 批量化：共享 index pool + 批量 select/增量（launch 数与 bs 无关）；线性项 38→5.2 ms/req", DARK),
    ("M5 CUDA graph：图内统一增量/稀疏路径零 host 同步；replay vs eager 逐位 0.00e+00；短行 veto 回退", DARK),
    ("M6 kq 真 4bit：uint8+scale 三张量，128→40 B/token-head，重建≡量化零漂移——131K 显存前提解锁", DARK),
    ("M7 prefill：归因修正（瓶颈是 select 而非 build）→ 池==因果区快路径；prefill 延迟随 S 亚线性（9.5→62.1 ms/层 @5K→130K）", DARK),
    ("诚实边界：S=10K 档仍慢于 triton 全量；稀疏收益=长 S 的 HBM 流量（bs=32×131K 时 triton ~620GB/步纯 HBM）——主表待 H100/131K", RED),
], fs=11.5)

# ============ S9 同机 kernel 三方对比（本轮新增，核心页） ============
s = slide()
title_bar(s, "同机 kernel 对比：Quest / DSA / TLI（NEW，本轮实测）", "官方 kernel 原样接入统一 harness，同机同口径诚实报告——H20-3e，decode 单 token per-layer-call，ms", RED)
mono_table(s, 0.55, 1.45, 8.3, 3.0, [
    ("indexer @131K", "S=10K", "S=40K", "S=131K", "存储/tok", "训练?", "LB-13"),
    ("Quest 官方", "0.066", "0.077", "0.107", "~1KB", "免训", "47.72"),
    ("DSA 官方", "0.476", "0.490", "0.503", "~132B", "需训", "—"),
    ("TLI eager", "0.736", "0.780", "0.853", "~336B", "免训", "49.92"),
    ("TLI fusedL1", "0.604", "0.658", "0.787", "~336B", "免训", "49.92"),
], fs=11.5, first_col=16)
bullet_box(s, 0.7, 4.6, 12.2, 2.7, [
    ("口径：Quest = page GEMV(q·kmin/kmax, fp16 32head) + 官方 raft decode_select_k(radix, k=64pages=1024tok)；DSA = 官方 tilelang fp8_index(64head×128d FP8 GEMV) + topk(2048)；TLI = 两级 select 全链路（真实 trace 两层均值，Hkv=8 GQA vs DSA 64head×128d 生产配置，架构差异如实报告）", GRAY),
    ("Quest 索引最快（µs 级 GEMV+radix）但代价是 1KB/token 索引存储（3× TLI）且 LongBench −2.2 分；DSA 0.5ms 与 S 无关（launch 主导）但需训练 indexer（1000 步/2.1B token warm-up）", DARK),
    ("诚实结论：TLI 当前原始延迟不占优（0.6–0.83 vs DSA 0.5ms）——但每 token 索引 MAC 32×↓（8kv-head×32d vs 64×128）、D' 跳 13/36 层摊销，且质量 +2.2 分 vs Quest；差距来源=eager topk/gather 的 launch 开销，M8 kernel 化是兑现路径（算力/存储余量已备好）", RED),
], fs=11)

# ============ S10 索引开销与 fused kernel（E8） ============
s = slide()
title_bar(s, "索引开销与 fused kernel（E8）", "正确解释：省算集中在 skip layers 与 L1 侧；attention 主计算量不变（K2 固定）——不包装成主算子胜利")
pic(s, "fig5_e8_speedup.png", 1.0, 1.6, w=11.2)
bullet_box(s, 1.0, 5.9, 11.5, 1.4, [
    ("E8-1: 跳层 4.88× / 非跳层 1.27×（A 子空间贡献）/ 全层平均 2.57×；index-side efficiency，非 attention 主算子加速", DARK),
    "E8-2: L1 fused 3.6×/1.6×（块 id 对拍一致）+ L2 级联 fused 1.63×（分区 topk 对拍 4096/4096）——完整两级仅 2 launches",
], fs=13)

# ============ S11 TLI vs TIA 逐层质量 ============
s = slide()
title_bar(s, "TLI vs TIA 逐层质量（E5b debug，hotpotqa）", "全 36 层 diff < 0.001；far-heavy 层反超（分区防挤出）——质量守恒的直接证据")
pic(s, "fig6_tli_vs_tia_layers.png", 1.5, 1.6, w=10.2)
stat_chip(s, 1.5, 6.2, 3.6, 1.0, "36/36 层", "diff < 0.001 vs TIA", GREEN)
stat_chip(s, 5.5, 6.2, 3.6, 1.0, "+0.0068", "L05 反超 TIA（0.9997 vs 0.9929）", BLUE)

# ============ S12 E5b 主表 + 负向任务诊断 ============
s = slide()
title_bar(s, "E5b：LongBench 13 子集主表（+ 负向任务诊断）", "质量 claim 收紧：近似守恒（−0.14 vs TIA / −0.44 vs FullKV），不是全面胜出")
box = s.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.7), Inches(3.9))
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
    r.font.size = Pt(11.5)
    r.font.bold = is_head or is_avg
    r.font.name = "Consolas"
    if is_avg:
        r2 = p.add_run(); r2.text = "  ← TLI"
        r2.font.size = Pt(11); r2.font.color.rgb = GREEN; r2.font.bold = True
    p.space_after = Pt(1)
bullet_box(s, 0.8, 5.35, 11.7, 1.9, [
    ("负向任务逐项诊断（GPT 复盘采纳）：musique −0.93 / gov_report −1.02 / qmsum −1.13 / repobench-p −0.81 vs TIA——musique/qasper 已定位为 D' gate 错位（关 D' 后精确恢复）；长摘要类疑似层跳过干扰全局聚合；正向任务 2wikimqa/hotpotqa/passage 支持分区预算价值", RED),
    ("D'-gate：far 总量 <0.3 的任务启用层掩码（4.88× 索引省算），多跳任务不跳层；补：多 seed/置信区间 + 远端挤出比例图（TIA vs TLI-B'）列入待办", DARK),
], fs=11)

# ============ S13 Negative results（→ design decision 格式） ============
s = slide()
title_bar(s, "Negative Results（论文护城河）", "每一项映射一个 final design decision——不是实验流水账", RED)
bullet_box(s, 0.7, 1.5, 12.0, 5.6, [
    ("E4c：严格 token 预算下，kmeans 聚类代表无一致优势（块 0.09–0.39 / token 0.45–0.79）；E4b 的 4–10× 占优含整簇超选 bug → 【decision】B' = 分区预算而非聚类", DARK),
    "E5：跨层 mask 复用 IoU 0.33；decode churn 22%；在线 L1 信号与 far 质量 corr≈0 → 【decision】D' 必须离线校准，无在线捷径",
    "H1/M4：dense top-1024 mass 覆盖恒 1.0000；竞争区仅占总 mass 0.515 → 【decision】主指标改剩余 mass coverage + 远端 recall",
    ("M6+扩展：L2 细筛维度选择不可压（δ→8 far rec 0.557→0.378，病因=维度非量化）；投影表示可压（PCA16≈选32）→ 【decision】非对称压缩定律精化：L1 上界可粗，L2 瓶颈在表示方式", DARK),
    ("分区消融（本轮）：hierarchical（L1 far/near 分区保底）No-Go——near 带仅 34 块 vs K1=128，far 池天然 ≥94 块（oracle far L1 存活 0.93–1.00）；L1 强制分区反把 far 池砍到 16 块 → 【decision】分区只需在 L2 终选级（当前实现即正确形态）；B' 增益机制=近端名额保障（far mass cov 同值 0.999）", DARK),
    ("投影对照（本轮）：随机投影崩溃（JL 保范数不保 GQA 点积排序）→ 【decision】降维必须用 K 协方差结构（PCA/校准基），不能用免校准随机基", DARK),
    ("M3-b：L2 级联纯 kernel 化 No-Go（4bit 并列过选无人兜底 + 二分串行反慢 4.5×）→ 【decision】级联分工：L1 容忍多选、topk 留给 torch", DARK),
    ("E5b：D' far 总量 gate 判据跨模型重叠（8B/32B 均不成立）；per-task 重校准长度失配更差 → 【decision】gate = per-task 层轮廓校准 + held-out 验证（待办）", DARK),
], fs=12.5)

# ============ S14 批判性复盘响应 ============
s = slide()
title_bar(s, "批判性复盘响应（外部 review 采纳清单）", "对 v4 的 19 页批判性复盘逐条分析：7 项采纳、2 项修正、其余已满足", GRAY)
bullet_box(s, 0.6, 1.4, 12.2, 5.7, [
    ( "采纳① claim 收紧：质量=守恒（≈TIA）而非胜出 FullKV；卖点二选一已定调=性能（同精度下索引降本+加速）", GREEN),
    ("采纳② 系统贡献分层：6.6× 明确标注为「原型→生产级」工程进步；10K 档 vs triton 差距继续如实报告", GREEN),
    ("采纳③ 负向任务逐项诊断 + 三张诊断图建议（任务×D'启用×delta 二维表 / 远端挤出比例图 / 任务级 residual coverage 相关性）——前两张列入待办", GREEN),
    ("采纳④ A 表述升级「非对称压缩定律」/ positional-stable subspace bound（S4 已改）", GREEN),
    ("采纳⑤ negative result → design decision 映射写法（S13 已改）", GREEN),
    ("采纳⑥ venue 定位 MLSys/ASPLOS/SC + Go/No-Go 硬阈值管理（S15）", GREEN),
    ("采纳⑦ RULER/NIAH/passkey + held-out gate + 多 seed 置信区间列入待办", GREEN),
    ( "修正① 「kernel 打不打得过」不用猜——本轮已补同机实测（S9）：Quest/DSA 官方 kernel 原样接入，TLI 延迟当前不占优但 MAC 32×↓ + 质量优势，兑现路径明确", RED),
    ( "修正② 「E8 与 E2E 关系」补分离口径：index/select/sparse-attn/decode-E2E 分开报（S9 表 + S8 归因行），2.57× 不冒充 E2E", RED),
], fs=11.5)

# ============ S15 待办（Go/No-Go 硬阈值版） ============
s = slide()
title_bar(s, "待办与计划（含 Go/No-Go 硬阈值）", "算法侧收敛 → 剩系统侧收尾 + 主表 + held-out 验证 + 写作")
bullet_box(s, 0.7, 1.35, 12.0, 5.9, [
    ("算法侧 ✅：A（双规模 + 维度边界）/ B'（重定位 + 敏感性）/ D'（跨规模 + gate 失败史）——结论可写", GREEN),
    ("系统侧 ✅：M3–M7（decode 原型→生产级 6.6× + prefill 2.1× + 4bit 3.2×），每步对拍零漂移", GREEN),
    ("kernel 对比 ✅（本轮）：Quest/DSA 官方 kernel 同机接入——TLI 0.6–0.83ms vs DSA 0.5ms，MAC 32×↓；⑤ M8 kernel 化是延迟兑现路径（TMA gather + fused select）", BLUE),
    ("PCA 投影集成（新方向，本轮验证）：kq 存 PCA 投影 4bit（r=16 → 16B/token-head vs 40B）+ q 侧 [D×r] GEMV；跨任务校准基已验证迁移——L2 打分 FLOP 再砍半 + 存储 2.5×↓", BLUE),
    ("⑥ MoBA 对比（flash-attn 依赖，降级预案=论文数字+算术账）+ Quest e2e 口径对比", DARK),
    ("⑦ 主表：H100 吞吐 + S=131K + RULER/NIAH/passkey（质量 Go/No-Go：AVG ≥ TIA−0.3，关键 far 任务退化 <1.5 分）", RED),
    ("⑧ gate held-out 验证（precision ≥0.98，误跳损失 <0.3 分；否则 D' 降级为 oracle/ablation）+ far_tokens 曲线图 + 多 seed 置信区间", RED),
    ("⑨ 论文写作（MLSys/ASPLOS/SC 骨架：算法边界 + 系统集成 + 负结果护城河）", DARK),
], fs=12.5)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
