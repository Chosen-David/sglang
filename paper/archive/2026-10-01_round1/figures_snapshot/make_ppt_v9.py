# TLI 终局数据 PPT v9（2026-09-30 05:05，用户指令：晨 5 点基于最新结论做一版）
# v8 增量：E72 全量终判（mavg 50.54 / 配置定稿 β.375）+ E67 tau01 gate 真判决
# 数据来源：paper_main_table.json / e72_screen_verdict.json / full_scores.json /
#           b7s_ruler_final.json / pred_e67_tau01_score.json
# 自包含（helper 同 v8 风格），输出 sglang/TLI_progress_v9.pptx
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
full = json.load(open("/tmp/tli_chain/full_scores.json"))
b7s_ruler = json.load(open("/tmp/tli_chain/b7s_ruler_final.json"))
tau01 = json.load(open("/home/wangyuanshuo02/sglang/pred_e67_tau01_score.json"))

OUT = "/home/wangyuanshuo02/sglang/TLI_progress_v9.pptx"

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


# ============ S1 封面：E72 全量终判主结果 ============
s = slide()
title_bar(s, "TLI 终局判决（v9，2026-09-30）",
          "E72 全量终判 mavg 50.54 + B7s 双基准双超 + E64a-j 消融全家桶 + E67 gate 真判决")
chip(s, 0.8, 1.6, 3.7, 1.2, "50.54",
     "E72 mavg LongBench 13 任务 AVG\nvs FullKV +0.18 / TIA +0.48 / Quest +2.82\n（TLI 系列新高，快筛幅度全量兑现）", GREEN)
chip(s, 4.8, 1.6, 3.7, 1.2, "87.93",
     "RULER 33 任务（B7s 严格口径）\nvs C0 +0.17 反超；L16384 +0.91\n（双基准主叙事行，mavg 臂未跑 RULER）", GREEN)
chip(s, 8.8, 1.6, 3.7, 1.2, "34.76",
     "musique e2e F1 全方法新高\n（FullKV 32.14 +2.62 / TIA 32.28 +2.48）\n多跳 far-heavy 独享大增益 = 机制证据", GREEN)
bullet_box(s, 0.8, 3.3, 12.0, 3.8, [
    ("分区创新点 e2e 判决成立：LongBench +0.25 / RULER +0.17 双口径（vs C0 单池）", GREEN),
    ("最优配置定稿：mavg=(minmax,avg) α=0.125 β=0.375 γ=0.125（E72 快筛 β 升级 +1.07 → 全量 +0.13）", BLUE),
    ("口径鸿沟（方法论贡献 #8）：trace mass 排名与 e2e F1 排名五臂系统性反转——双口径必须都测都报告", RED),
    "E67 gate 真判决（τ0.005 非 no-op，触发率实质非零）：多跳微损不崩 → gate 定位 = 安全省算力保守开关",
    "负结果资产：聚类代表（口径陷阱）/ 监督投影 e2e 崩 / TC 化反慢 0.63× / 异 method 组合无增益",
], fs=13.5)

# ============ S2 主表：双基准四方法 ============
s = slide()
title_bar(s, "论文主表：双基准四方法终值",
          "paper_main_table.json；LongBench 13 任务 + RULER 33 任务（三长度平均）")
rows = [("method", "LongBench", "RULER")]
hl = {m["method"]: (m["longbench_avg"], m["ruler_avg"]) for m in main_tbl["rows"]}
for m in ("FullKV", "Quest", "TIA", "TLI_C0", "TLI_B7", "TLI_E72"):
    l, r_ = hl[m]
    rows.append((m, str(l) if l is not None else "-",
                 str(r_) if r_ is not None else "- (not run)"))
mono_table(s, 0.8, 1.45, 11.8, 3.0, rows, fs=14)
bullet_box(s, 0.8, 4.6, 12.0, 2.6, [
    ("TLI_E72 = LongBench 列最优 50.54：training-free 稀疏索引器首次超 FullKV（+0.18）", GREEN),
    "TLI_B7 保持双基准叙事：LongBench 50.41 双超 + RULER 87.93（+0.17），长度梯度 −0.14→−0.25→+0.91",
    ("musique 34.76（FullKV +2.62）：分区 far 检索的预算分配增益在多跳 far-heavy 任务最纯粹兑现", GREEN),
    "Quest 双基准最弱（47.72/79.95）——外部 baseline 领先幅度 LongBench +2.8；唯一明显损失 narrativeqa 23.96（−0.50）",
], fs=12.5)

# ============ S3 E72 五组合判决 + 全量终判 ============
s = slide()
title_bar(s, "E72：五 method 组合 e2e 判决（快筛→全量兑现）",
          "e72_screen_verdict.json + full_scores.json；trace mass 与 e2e F1 排名系统性反转")
rows = [("arm (far,near)", "α/β", "hotpotqa", "musique", "screen avg", "trace mass", "trace rank")]
order = ["mavg", "aavg", "mminmax", "mavg(B7s)", "cavg"]
tr = {"mavg": 4, "aavg": 5, "mminmax": 2, "mavg(B7s)": 1, "cavg": 3}
for a in order:
    d = screen[a]
    rows.append((a.replace("(B7s)", "\nβ=.25 ref"), f"{d['alpha']}/{d['beta']}",
                 str(d["hotpotqa"]), str(d["musique"]), str(d["screen_avg"]),
                 f"{d['trace_mass']:.4f}", str(tr[a])))
mono_table(s, 0.8, 1.45, 11.8, 3.2, rows, fs=11.5)
bullet_box(s, 0.8, 4.9, 12.0, 2.4, [
    ("全量终判：mavg 13 任务 AVG 50.54 = 系列新高（vs B7s +0.13；musique +1.94 幅度与快筛精确一致）", GREEN),
    ("trace 第 4 的 mavg → e2e 全量冠军；trace 垫底 aavg → e2e 第 2——「trace mass 排名不可替代 e2e 判决」获全量确认", RED),
    "β=0.375 > β=0.25（快筛 +1.07 → 全量 +0.13）：近端预算份额宜超区域份额，跨口径一致",
    "cavg（cluster far）e2e 健康但末位 42.87——E4d trace 结论未在 e2e 兑现；其余 8 任务 mavg vs B7s 全部 0~−0.44 噪声级",
], fs=12.5)

# ============ S4 消融定稿：参数空间 ============
s = slide()
title_bar(s, "消融定稿：六维参数空间（E64a-j，132 臂 + E72 五臂 e2e）",
          "用户一般化框架：mid 分 near/far → 降维 → 双区各自粗筛+细筛")
mono_table(s, 0.8, 1.45, 11.8, 3.0, [
    ("参数", "终值", "依据"),
    ("α（near 区比例）", "0.125", "51+81 臂网格单调递减；最优内部臂恒 α.125（E64a/g）"),
    ("β（near 页预算）", "0.375", "恒 ≥α；e2e 同 method β.375>.25（E64i +0.001~+0.014 + E72 快筛 +1.07→全量 +0.13）"),
    ("γ（near token 预算）", "0.125", "必须随 K2 等比缩放；γ.5 在 K2=1024 far 崩（B0 13.51 教训）"),
    ("far_method", "minmax", "块上界 mono 最稳 0.9166（E64a）；e2e 冠军臂"),
    ("near_method", "avg", "near 侧自由度 ≤0.003（E64i 紧/宽预算双口径）；异 method 组合创新点删"),
    ("降维", "粗筛尾维 32 / 细筛 PCA d16", "E3/E65；监督投影 e2e 崩（negative，E66 LOTO 92-94%）"),
], fs=11.5)
bullet_box(s, 0.8, 4.7, 12.0, 2.5, [
    ("分区价值主张：near 免 min/max 打分 = 结构性带宽节省（near_method=avg）+ 紧预算分配质量（e2e 判决）", GREEN),
    "kernel 资产：fused L1 3.6× / L2 级联 1.63× / M11 prefill e2e 4.30×；「L1 容多选、L2 不能」级联边界教训",
    ("negative results 全家桶：聚类代表（超选+平均化口径陷阱）/ TC 化反慢 0.63× / 在线信号失效 / 跨层复用失效", RED),
], fs=12.5)

# ============ S5 E67 gate 真判决 ============
s = slide()
title_bar(s, "E67：感知型 per-request gate 真判决（升 τ）",
          "pred_e67_tau01_score.json；对照 = gate_off（musique 27.57 / qasper 40.37）")
mono_table(s, 0.8, 1.45, 11.8, 2.4, [
    ("τ", "触发程度", "musique", "qasper", "判决"),
    ("0.01（E5b 默认）", "近 no-op（1.1% 输出分歧）", "—", "—", "「无损」证据强度不足"),
    ("0.005（tau01）", "实质非零（11.8% 实体级分歧）", "27.80 (+0.23)", "39.96 (−0.41)", "微损不崩"),
    ("0.015（tau02）", "安全/多跳边界", "进行中（GPU1）", "进行中", "待出"),
], fs=12.5)
bullet_box(s, 0.8, 4.2, 12.0, 2.9, [
    ("τ0.005 真判决：触发率实质非零下多跳微损（合计 −0.18/任务对，噪声级）——对照静态版掉 4.8-5.9 分", GREEN),
    "per-request 动态信号（prefill last1 corr 0.924）确实防住反向错误（far 重要的层不跳）",
    ("gate 终定位 = 「安全省算力的保守开关」而非精度增益点：正收益（省算力换精度）未兑现", BLUE),
    "速度侧资产（trace 重放）：跳 near 空间 31% 层/2.3% mass@τ0.1 > 跳 far 12%/1.7%；τ≥0.2 才有可观跳层空间",
], fs=12.5)

# ============ S6 结论页 ============
s = slide()
title_bar(s, "结论与剩余工作",
          "#74 步骤①-⑤完成（创新点终评 + 一般化框架 + 主表 + PPT + 数字终审）")
bullet_box(s, 0.8, 1.5, 12.0, 5.5, [
    ("主结果：LongBench 50.54（E72 mavg）/ RULER 87.93（B7s）——training-free 稀疏索引器双基准逼近或超越 FullKV", GREEN),
    "创新点定稿：A 子空间粗筛（保留）/ B' 分区预算（保留，e2e 成立）/ D' gate（降级「安全开关」消融）/ 口径鸿沟（方法论贡献）",
    "方法叙事统一为一般化框架（§2''）：保护段 + mid 双区 + 六维参数空间 + 双区两级流水线",
    "速度三支柱：kernel fused 1.63-3.6× / 30B 64K e2e 1.285× / 8B 稳态 decode 1.21×——对拍精确一致 + 逐字一致",
    ("剩余：E67 tau02（τ0.015，GPU1 进行中）/ paper 全文重写（#67 blocked→待启动）", BLUE),
], fs=14)

prs.save(OUT)
print("saved ->", OUT)
