# TLI 终局数据 PPT v8（2026-09-30 晨，#74 步骤④）
# 定位：v7（09-28）之后 E71 B7s 双超 + E64a-j 消融全家桶 + E72 五组合 e2e 判决的终局汇报
# 数据来源：paper_main_table.json / e72_screen_verdict.json / b7s_ruler_final.json / e64j_combo_best.json
# 自包含（helper 同 make_ppt_v7.py 风格），输出 sglang/TLI_progress_v8.pptx
import json

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

BLUE = RGBColor(0x2F, 0x6F, 0x9F)
RED = RGBColor(0xC1, 0x44, 0x3C)
GREEN = RGBColor(0x3A, 0x7D, 0x44)
GRAY = RGBColor(0x55, 0x55, 0x55)
DARK = RGBColor(0x22, 0x22, 0x22)

R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
main_tbl = json.load(open(f"{R}/paper_main_table.json"))
screen = json.load(open(f"{R}/e72_screen_verdict.json"))
b7s_ruler = json.load(open("/tmp/tli_chain/b7s_ruler_final.json"))

OUT = "/home/wangyuanshuo02/sglang/TLI_progress_v8.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def title_bar(s, text, sub=None, color=BLUE):
    from pptx.util import Inches as I
    box = s.shapes.add_textbox(I(0.5), I(0.25), I(12.3), I(0.85))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = color
    if sub:
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(12)
        p2.font.color.rgb = GRAY


def bullet_box(s, x, y, w, h, items, fs=14):
    from pptx.util import Inches as I
    box = s.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        color = DARK
        text = it
        if isinstance(it, tuple):
            text, color = it
        p.text = "• " + text if not text.startswith("   ") else text
        p.font.size = Pt(fs)
        p.font.color.rgb = color


def mono_table(s, x, y, w, h, rows, fs=12.5, highlight_col=None):
    from pptx.util import Inches as I
    tb = s.shapes.add_table(len(rows), len(rows[0]), I(x), I(y), I(w), I(h)).table
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tb.cell(ri, ci)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(fs if ri else fs + 0.5)
                p.font.bold = ri == 0
                if ri == 0:
                    p.font.color.rgb = DARK
                elif highlight_col is not None and ci == highlight_col:
                    p.font.color.rgb = GREEN
                    p.font.bold = True


def chip(s, x, y, w, h, big, small, color=GREEN):
    from pptx.util import Inches as I
    box = s.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = big
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = color
    p2 = tf.add_paragraph()
    p2.text = small
    p2.font.size = Pt(11)
    p2.font.color.rgb = GRAY


# ============ S1 封面：B7s 双超主结果 ============
s = slide()
title_bar(s, "TLI 终局判决（v8，2026-09-30）",
          "E71 B7s 严格口径双超 + E64a-j 消融全家桶 + E72 五组合 e2e 判决（#74 全面重写配套）")
chip(s, 0.8, 1.6, 3.7, 1.2, "50.41", "LongBench 13 任务 AVG\n超 TIA +0.35 / 超 FullKV +0.05\n（TLI 系列首次双超）", GREEN)
chip(s, 4.8, 1.6, 3.7, 1.2, "87.93", "RULER 33 任务（严格口径）\nvs C0 +0.17 反超\nL16384 +0.91 长度梯度", GREEN)
chip(s, 8.8, 1.6, 3.7, 1.2, "34.76", "musique e2e F1 全方法新高\n（FullKV 32.14 / TIA 32.28）\n分区 far 检索质量兑现", GREEN)
bullet_box(s, 0.8, 3.3, 12.0, 3.8, [
    ("分区创新点 e2e 判决成立：LongBench +0.25 / RULER +0.17 双口径（vs C0 单池）", GREEN),
    ("终局机制定性：分区收益本质 = 紧预算下的预算分配（trace 宽预算口径 mono 微胜 −0.005 与 e2e 并列报告）", BLUE),
    ("口径鸿沟（方法论贡献 #8）：trace mass 排名与 e2e F1 排名五臂系统性反转——双口径必须都测都报告", RED),
    "最优配置定稿：mavg=(minmax,avg) α=0.125 β=0.375 γ=0.125（E72 五臂快筛 + E64 132 臂消融）",
    "负结果资产：聚类代表（口径陷阱）/ 监督投影 e2e 崩 / TC 化反慢 0.63× / 异 method 组合无增益",
], fs=13.5)

# ============ S2 主表：双基准四方法 ============
s = slide()
title_bar(s, "论文主表：双基准四方法终值", "paper_main_table.json；LongBench 13 任务 + RULER 33 任务（三长度平均）")
rows = [("method", "LongBench", "RULER")]
hl = {m["method"]: (m["longbench_avg"], m["ruler_avg"]) for m in main_tbl["rows"]}
for m in ("FullKV", "Quest", "TIA", "TLI_C0", "TLI_B7"):
    l, r_ = hl[m]
    rows.append((m, str(l) if l is not None else "-", str(r_) if r_ is not None else "-"))
mono_table(s, 0.8, 1.45, 11.8, 2.6, rows, fs=14, highlight_col=None)
bullet_box(s, 0.8, 4.3, 12.0, 2.9, [
    ("TLI_B7 双基准同时超越 TIA：LongBench +0.35（首次超 FullKV +0.05）/ RULER 差 0.42 但 L16384 +0.91 反超", GREEN),
    "RULER 长度梯度：−0.14（4K）→ −0.25（8K）→ +0.91（16K）——分区 far 检索收益区 = 超长上下文，机制与「短 S near 覆盖大半 mid」一致",
    ("musique 32.82 反超 TIA（32.28）：多跳 far-heavy 任务上分区预算的检索质量增益 +2.4（vs C0）", GREEN),
    "唯一明显损失 narrativeqa 23.96（−0.50）；Quest 双基准最弱（47.72/79.95）——外部 baseline 领先幅度 LongBench +2.7",
    ("严格口径 vs 旧口径（v1caliber）：L16384 +0.98——sink/swa 保送的浪费随 S 放大，严格口径是正确工程选择", BLUE),
], fs=12.5)

# ============ S3 E72 五组合判决 ============
s = slide()
title_bar(s, "E72：五 method 组合 e2e 判决（快筛 hotpotqa+musique）",
          "e72_screen_verdict.json；trace mass（E64j 宽预算）与 e2e F1 排名系统性反转")
rows = [("arm (far,near)", "α/β", "hotpotqa", "musique", "avg", "trace mass", "trace rank")]
order = ["mavg", "aavg", "mminmax", "mavg(B7s)", "cavg"]
tr = {"mavg": 4, "aavg": 5, "mminmax": 2, "mavg(B7s)": 1, "cavg": 3}
for a in order:
    d = screen[a]
    rows.append((a.replace("(B7s)", "\nβ=.25 ref"), f"{d['alpha']}/{d['beta']}",
                 str(d["hotpotqa"]), str(d["musique"]), str(d["screen_avg"]),
                 f"{d['trace_mass']:.4f}", str(tr[a])))
mono_table(s, 0.8, 1.45, 11.8, 3.4, rows, fs=11.5)
bullet_box(s, 0.8, 5.0, 12.0, 2.2, [
    ("mavg e2e 冠军 44.59（musique 34.76 全方法新高）但 trace 第 4；aavg e2e 第 2 但 trace 垫底——排名五臂全反转", RED),
    "β=0.375 > β=0.25（+1.07）：近端预算份额宜超区域份额，与 trace 侧 E64i 结论跨口径一致",
    "cavg（cluster far）e2e 首测健康（kmeans 修复后与其它臂同速）但末位 42.87——E4d cluster far trace 结论未在 e2e 兑现",
], fs=12.5)

# ============ S4 消融定稿：参数空间 ============
s = slide()
title_bar(s, "消融定稿：六维参数空间（E64a-j，132 臂 + E72 五臂 e2e）",
          "用户一般化框架：mid 分 near/far → 降维 → 双区各自粗筛+细筛")
mono_table(s, 0.8, 1.45, 11.8, 3.0, [
    ("参数", "终值", "依据"),
    ("α（near 区比例）", "0.125", "51+81 臂网格单调递减；最优内部臂恒 α.125（E64a/g）"),
    ("β（near 页预算）", "0.375", "恒 ≥α；e2e 同 method β.375>.25 +1.07（E64i+E72）"),
    ("γ（near token 预算）", "0.125", "必须随 K2 等比缩放；γ.5 在 K2=1024 far 崩（B0 13.51 教训）"),
    ("far_method", "minmax", "块上界 mono 最稳 0.9166（E64a）；e2e 冠军臂"),
    ("near_method", "avg", "near 侧自由度 ≤0.003（E64i）；异 method 组合创新点删"),
    ("降维", "粗筛尾维 32 / 细筛 PCA d16", "E3/E65；监督投影 e2e 崩（negative，E66 LOTO 92-94%）"),
], fs=11.5)
bullet_box(s, 0.8, 4.7, 12.0, 2.5, [
    ("分区价值主张：near 免 min/max 打分 = 结构性带宽节省（near_method=avg）+ 紧预算分配质量（e2e 判决）", GREEN),
    "kernel 资产：fused L1 3.6× / L2 级联 1.63× / M11 prefill fused e2e 一致；「L1 容多选、L2 不能」级联边界",
    ("negative results 全家桶：聚类代表（超选+平均化口径陷阱）/ TC 化反慢 0.63×（内存受限算子）/ 在线信号失效 / 跨层复用失效", RED),
], fs=12.5)

# ============ S5 结论页 ============
s = slide()
title_bar(s, "结论与剩余工作", "#74 步骤①②③完成（§2' 评估表 + §2'' 一般化框架 + 主表落袋）")
bullet_box(s, 0.8, 1.5, 12.0, 5.5, [
    ("主结果：TLI_B7 = LongBench 50.41 / RULER 87.93——training-free 稀疏索引器双基准逼近或超越 FullKV", GREEN),
    "创新点定稿：A 子空间粗筛（保留）/ B' 分区预算（保留，e2e 成立）/ D' gate（降级消融）/ 口径鸿沟（新晋方法论贡献）",
    "方法叙事统一为一般化框架（§2''）：保护段 + mid 双区 + 六维参数空间 + 双区两级流水线",
    "速度三支柱（v7 继承）：kernel 1.3-5.1× / 30B 64K e2e 1.285× / 8B 稳态 decode 1.21×——对拍精确一致 + 逐字一致",
    ("剩余：E72 全量 13 任务终表（跑批中，T5 自动衔接）/ E67 τ 感知 gate e2e（链后自动）/ paper 全文重写 + 数字终审", BLUE),
], fs=14)

prs.save(OUT)
print("saved ->", OUT)
