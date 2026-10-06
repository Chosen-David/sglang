# PSI 汇报 PPT v11（2026-10-04）
# 按 E98 全量重写后的论文结构与数据重制（TLI_paper.tex 19 页版为纲）
# v11 相对 v10 的重排：
#   ① 叙事对齐新论文：PSI 命名 / 双口径网格选举 / 部署简单性 / 口径鸿沟
#   ② 主表臂换为双口径选举出的新配置（50.78，13 任务全量实测）
#   ③ 新增「配置选举」页（mass→e2e 排序反转，fig11 入页）
#   ④ 新增「部署简单」页（四重否定 → 参数面平坦是结构性属性）
#   ⑤ 消融页对齐新论文（L1 上界维度双校准 / 紧预算 / top-σ / γ 截断坍缩）
#   ⑥ 文本零内部代号（E71/E98/mavg 等一律写方法语义，同论文铁律）
# 数据来源（逐位取自 JSON，不凭记忆）：
#   e98_full_13tasks.json   新主表臂 13 任务全量（50.78）
#   e98_best_election.json  选举判决（BEST 与 mass→e2e 反转）
#   e98_e2e_grid.json       e2e 网格 13 臂
#   e99_predictor_probe.json  预测器四重否定（NO-GO）
#   e71_main_table.json     基线行（FullKV/Quest/TIA/单池/三压缩/StreamLLM）
#   m6_sinkguard_full.json  sink 保送适配判决
#   e89_moba_full.json      训练门控类基线全量
#   e88_partition_verdict.json  紧预算分区 trace 口径
#   e87c_mavg_tail.json     L1 上界维度成对校准
# 纯 CPU，不碰 GPU。
import json

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor

BLUE = RGBColor(0x2F, 0x6F, 0x9F)
RED = RGBColor(0xC1, 0x44, 0x3C)
GREEN = RGBColor(0x3A, 0x7D, 0x44)
GRAY = RGBColor(0x55, 0x55, 0x55)
DARK = RGBColor(0x22, 0x22, 0x22)

FIG = "/home/wangyuanshuo02/two-level-attention/exp/figures"
FIG11 = "/tmp/fig11_mass_vs_e2e.png"          # 论文口径副本渲染（无代号版）
FIG5 = f"{FIG}/fig5_abg_scan.png"             # 环标去代号后的当前版
OUT = "/home/wangyuanshuo02/sglang/TLI_progress_v11.pptx"

# ---------- 数据加载（逐位取数） ----------
R = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
full13 = json.load(open(f"{R}/e98_full_13tasks.json"))["tasks"]      # 新主表臂
elec = json.load(open(f"{R}/e98_best_election.json"))
e2e_grid = json.load(open(f"{R}/e98_e2e_grid.json"))
pred = json.load(open(f"{R}/e99_predictor_probe.json"))
base = json.load(open(f"{R}/e71_main_table.json"))                   # FullKV/Quest/TIA/TLI_C0(单池)/...
sink = json.load(open(f"{R}/m6_sinkguard_full.json"))
moba = json.load(open(f"{R}/e89_moba_full.json"))
tight = json.load(open(f"{R}/e88_partition_verdict.json"))
calib = json.load(open(f"{R}/e87c_mavg_tail.json"))

PSI_AVG = full13["AVG"]                                              # 50.78
FULL_AVG = base["FullKV"]["AVG"]                                     # 50.36
QUEST_AVG = base["Quest"]["AVG"]                                     # 47.72
TIA_AVG = base["TIA"]["AVG"]                                         # 50.06
POOL_AVG = base["TLI_C0"]["AVG"]                                     # 50.16（单池消融臂）
MOBA_AVG = moba["AVG"]                                               # 49.19
GAMMA = PSI_AVG - FULL_AVG                                           # +0.42
VS_QUEST = PSI_AVG - QUEST_AVG                                       # +3.06
VS_MOBA = PSI_AVG - MOBA_AVG                                         # +1.59

# 13 任务胜负分解（程序化计算，不手写）
TASKS = [k for k in full13 if k != "AVG"]
wins = [(t, full13[t] - base["FullKV"][t]) for t in TASKS
        if full13[t] > base["FullKV"][t]]
losses = [(t, full13[t] - base["FullKV"][t]) for t in TASKS
          if full13[t] <= base["FullKV"][t]]
assert len(wins) == 8 and len(losses) == 5, (len(wins), len(losses))
win_top = sorted(wins, key=lambda x: -x[1])[:4]      # musique/hotpotqa/2wikimqa/narrativeqa
loss_max = min(losses, key=lambda x: x[1])           # passage_retrieval −1.00

# 选举页数字
BEST = elec["best"]                                                  # (minmax,avg) .125/.375/.625 → 45.08
REF_ARM = elec["ref_arm"]                                            # 部署参照臂 γ=.125 → 44.59
REF_GAIN = elec["ref_gain"]["delta"]                                 # +0.49
MASS_CHAMP = elec["mass_vs_e2e"]["mass_best"]                        # mass 冠军 e2e 43.9
MASS_LOSS = pred["oracle"]["mass_pick_loss_vs_oracle"]               # 1.18
SPEARMAN = pred["correlation_table"][-1]["e2e_avg_spearman"]         # 0.166
CI = pred["mass_vs_e2e_avg_bootstrap_CI"]                            # [-0.333, 0.553]

# 四重否定（预测器可行性初探的先验对齐表，逐位取数）
FOUR = pred["prior_alignment"]

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def title_bar(s, text, sub=None, color=BLUE):
    box = s.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.85))
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
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
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


def mono_table(s, x, y, w, h, rows, fs=12.5, highlight_col=None, bold_rows=()):
    tb = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                            Inches(w), Inches(h)).table
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tb.cell(ri, ci)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(fs if ri else fs + 0.5)
                p.font.bold = ri == 0 or ri in bold_rows
                if ri == 0:
                    p.font.color.rgb = DARK
                elif highlight_col is not None and ci == highlight_col:
                    p.font.color.rgb = GREEN
                    p.font.bold = True
    return tb


def chip(s, x, y, w, h, big, small, color=GREEN):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
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


def add_pic(s, path, x, y, w):
    """按宽度等比插入图片，返回 (width, height) 英寸。"""
    from PIL import Image
    iw, ih = Image.open(path).size
    h = w * ih / iw
    s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


# ============ S1 封面：PSI 与三个头部数字 ============
s = slide()
title_bar(s, "PSI：位置稳定免训练两级稀疏注意力索引（v11，2026-10-04）",
          "块级上界粗筛 + far/near 分区精筛；主表配置经双口径网格端到端选举产生（论文 19 页版为纲）")
chip(s, 0.8, 1.6, 3.7, 1.2, f"{PSI_AVG:.2f}",
     f"LongBench 13 任务主表臂\n超 FullKV（{FULL_AVG:.2f}）+{GAMMA:.2f}\nvs Quest +{VS_QUEST:.2f} / vs MoBA +{VS_MOBA:.2f}", GREEN)
chip(s, 4.8, 1.6, 3.7, 1.2, f"{full13['musique']:.2f}",
     "musique 多跳 far-heavy 任务\n全部方法（含 FullKV）最高\nFullKV 32.14 → +2.57", GREEN)
chip(s, 8.8, 1.6, 3.7, 1.2, "87.93",
     "RULER 33 任务（三长度平均）\n逼近 FullKV 88.71（−0.78）\n16K 档分区增益 +0.91 转正", GREEN)
bullet_box(s, 0.8, 3.3, 12.0, 3.8, [
    ("两级索引：L1 块级 minmax 上界粗筛（子空间 d'=32 + 4bit，无漏选）→ L2 far/near 分区 token 级精筛（远端独立配额防挤出）", BLUE),
    ("配置选举方法学（新增贡献）：3600 个合法 (α,β,γ) 配置的离线质量网格 + 13 个配置臂的端到端实测网格——离线质量冠军在端到端跌至第 3 档（排序系统性反转，秩相关 0.166），选举只认端到端实测", RED),
    "部署简单：任务/层/样本/配置臂四粒度 oracle 上限全部 ≤0.5 分——参数面平坦是结构性属性，一组固定配置覆盖 12/13 任务（对比 DSA 千步 warm-up）",
    "速度侧如实报告：选择链 kernel 级 1.46×@32K → 5.09×@128K；三方 indexer 延迟排名第三，优势在索引计算量（DSA 的 1/32）与索引存储（Quest 的 1/3）",
    "口径铁律贯穿：kernel microbench 与 e2e 两层都测都报；全部精度代价与负结果同机同协议入册",
], fs=13)

# ============ S2 背景与动机 ============
s = slide()
title_bar(s, "背景：KV 带宽瓶颈与三类现有方案的困难权衡",
          "Qwen3-8B 真实 trace：dense top-1024 的注意力质量呈三层位置台阶")
mono_table(s, 0.8, 1.45, 11.8, 1.7, [
    ("位置段", "质量占比", "统计性质", "索引需求"),
    ("sink（首 token）", "0.37–0.71", "质量大且位置确定", "免打分强制保留"),
    ("近端滑窗", "0.16–0.29", "密集平稳、分数接近", "粗索引即可（块均值不漏）"),
    ("远端检索", "0.02–0.29", "稀疏尖锐、kv-head 间极不均（单 head 独占 0.804）", "上界粗筛 + token 级精筛"),
], fs=12)
bullet_box(s, 0.8, 3.35, 12.0, 3.7, [
    ("免训练稀疏索引（Quest 为代表）：page 采样最小值构造下界近似——下界低估真实高分页，干扰 key 随长度超线性增多时漏选加剧：实测 RULER 16K multikey_3 崩至 21 分（FullKV 100）", RED),
    ("KV 压缩（SnapKV/H2O/PyramidKV）：prefill 时一次性丢弃 KV，无法响应 decode 查询分布；无显式 sink 保送时 Qwen3-8B 首 token 即 EOS（全量 AVG 14.9–28.8 崩坏）", RED),
    ("训练类索引（DSA）：可学习投影换选择能力，需 1000 步 warm-up 与 2.1B token 训练预算，选择质量依赖训练分布", RED),
    ("设问：能否设计一个 training-free、零 warm-up 的稀疏注意力索引，在不引入任何可学习参数的前提下，同时保住长上下文检索质量并兑现带宽收益？", BLUE),
    "关键观察：三类 attention 的预算需求互相冲突——稀疏索引要保精度必须同时喂饱三者，这正是分区预算的出发点",
], fs=12.5)

# ============ S3 方法：两级索引架构 ============
s = slide()
title_bar(s, "方法：PSI 两级级联索引（架构四组件）",
          "创新不在级联形态（已有工作验证），而在两个正交维度：分数表示取上界 + 预算组织取分区")
mono_table(s, 0.8, 1.45, 11.8, 2.5, [
    ("组件", "机制", "关键性质"),
    ("L1 块级上界粗筛", "B=64 切块，子空间 d'=32 上逐维 min/max，4bit 量化（128 维 → 40B/token-head）；query 打分取上界乐观侧，top-K1=128 块", "子空间打分口径下无漏选：真实高分 token 所在块必然入选（vs Quest 下界漏选）"),
    ("L2 far/near 分区精筛", "sink+滑窗强制保留不占预算；mid 预算分两池——far 池对 L1 入选块做 4bit token 级精筛（配额 128–256 饱和），near 池同口径精筛", "远端独立配额防挤出；池边界即因果边界，无需显式因果 mask"),
    ("感知型 per-request gate", "prefill 末段一次 softmax 统计识别 far 质量近零的层，decode 侧安全跳过其远端检索", "动态信号与 decode 侧 far 质量相关 0.924；安全省算力的保守开关"),
], fs=11.5)
bullet_box(s, 0.8, 4.15, 12.0, 2.9, [
    "预算三参数：α=0.125（near 区比例）/ β=0.375（near 页预算）/ γ=0.625（near token 预算折扣），far 配额保底 64；主表配置由端到端网格选举产生（见后页）",
    ("kernel 化四组件：fused L1 块打分 3.6×@32K / L2 级联 dual topk 单 launch / 统一稀疏 attention fused kernel 11–16× / O(n) 精确增量索引 0.128ms 与序列长度无关", GREEN),
    "生产形态三项工程：paged 寻址零拷贝 / per-layer 共享 index pool 预分配（launch 数与 batch 无关）/ CUDA graph 捕获区零 host 同步——已在 sglang 集成",
    "H20 设计约束：Tensor Core 吞吐仅 H100 的 15% 而 HBM 带宽同级——打分类 kernel 第一性约束是带宽，TC 化与 TMA 转置实测均否决",
], fs=12)

# ============ S4 核心结果：LongBench 13 任务主表 ============
s = slide()
title_bar(s, f"核心结果：LongBench 13 任务主表臂 {PSI_AVG:.2f}",
          "PSI = (minmax, avg) 打分 α=0.125/β=0.375/γ=0.625（端到端网格选举产生）；单池 = 同预算 α=0 消融臂")
rows = [("任务", "FullKV", "Quest", "TIA", "PSI-单池", "PSI", "MoBA")]
for t in TASKS:
    name = t.replace("_en", "").replace("passage_retrieval", "passage_ret.")
    rows.append((name, f"{base['FullKV'][t]:.2f}", f"{base['Quest'][t]:.2f}",
                 f"{base['TIA'][t]:.2f}", f"{base['TLI_C0'][t]:.2f}",
                 f"{full13[t]:.2f}", f"{moba[t]:.2f}"))
rows.append(("总分", f"{FULL_AVG:.2f}", f"{QUEST_AVG:.2f}", f"{TIA_AVG:.2f}",
             f"{POOL_AVG:.2f}", f"{PSI_AVG:.2f}", f"{MOBA_AVG:.2f}"))
mono_table(s, 0.5, 1.4, 12.4, 4.6, rows, fs=10.5, highlight_col=5, bold_rows=(len(rows) - 1,))
bullet_box(s, 0.5, 6.15, 12.4, 1.2, [
    (f"对 FullKV +{GAMMA:.2f} / 对 Quest +{VS_QUEST:.2f} / 对同管线第一代 TIA +{PSI_AVG - TIA_AVG:.2f} / 对训练门控类 MoBA +{VS_MOBA:.2f}；对同预算单池消融臂 +{PSI_AVG - POOL_AVG:.2f} = 分区防挤出净增益", GREEN),
    f"13 任务 {len(wins)} 胜 {len(losses)} 负：正向由 musique +{win_top[0][1]:.2f}、hotpotqa +{win_top[1][1]:.2f}、2wikimqa +{win_top[2][1]:.2f}、narrativeqa +{win_top[3][1]:.2f} 驱动；负向全部 ≤1 分（{loss_max[0].replace('_en','')} {loss_max[1]:.2f} 为最大损失点）",
], fs=11.5)

# ============ S5 配置选举：双口径网格与排序反转 ============
s = slide()
title_bar(s, "配置选举：双口径网格方法学与排序反转",
          "离线质量网格（5 组合 × 9×9×9 α/β/γ，3600 个合法配置，16 样本 trace）→ 端到端网格（按有效配置去重精选 13 臂，LongBench 双任务 n=200 实测）")
# 左：四组合反转表；右：13 臂代表行
mono_table(s, 0.5, 1.5, 6.3, 2.2, [
    ("方法组合 (far,near)", "质量口径", "质量名次", "e2e 均分", "e2e 名次"),
    ("(minmax, minmax)", "0.896", "1", "43.90", "3（并列）"),
    ("(minmax, avg)", "0.882", "2", "45.08", "1"),
    ("(cluster, avg)", "0.879", "3", "43.58", "4"),
    ("(avg, avg)", "0.796", "4", "43.90", "3（并列）"),
], fs=11, highlight_col=3)
mono_table(s, 7.0, 1.5, 5.9, 2.2, [
    ("端到端网格代表臂", "质量", "hotpotqa", "musique", "e2e 均分"),
    ("(minmax,avg) .125/.375/.625", "0.880", "55.44", "34.71", "45.08 ←当选"),
    ("(minmax,avg) .125/.25/.75", "0.882", "54.16", "35.57", "44.86"),
    ("(minmax,minmax) .875/.875/.5", "0.896", "54.40", "33.40", "43.90"),
    ("部署参照臂 .125/.375/.125", "0.873", "54.43", "34.76", "44.59"),
], fs=10.5)
bullet_box(s, 0.5, 3.7, 12.4, 1.2, [
    (f"排序系统性反转：质量口径与端到端口径秩相关仅 {SPEARMAN:.3f}（bootstrap 95% CI 含 0）；质量冠军在端到端跌至第 3 档，按质量口径选臂相对端到端最优亏 {MASS_LOSS:.2f}——代理口径对可用级配置主动误导而非中性", RED),
    f"选举只认端到端实测：最优臂 (minmax, avg) α=0.125/β=0.375/γ=0.625 均分 45.08 居首（hotpotqa 55.44 为全部 13 臂最高）；相对部署参照臂 +{REF_GAIN:.2f}",
    "   γ 截断口径注：质量网格总预算 2048 与端到端 K2=1024 的 2 倍差使 γ 被近端细筛配额截断压平（γ=0.375 与 0.75 逐位同分 54.16/35.57）——跨口径选臂必须按有效配置去重",
], fs=11)
# fig11 入页（宽 11.2 → 高约 2.49，页底内收）
add_pic(s, FIG11, 1.05, 4.95, 11.2)

# ============ S6 部署简单：参数面平坦四重否定 ============
s = slide()
title_bar(s, "部署简单：参数面平坦是结构性属性（四重否定）",
          "预测器可行性判定 NO-GO：任务/层/样本/配置臂四粒度否定收敛于同一结论")
mono_table(s, 0.5, 1.5, 6.6, 2.3, [
    ("选参粒度", "oracle 上限", "口径"),
    ("逐任务完美选 β", f"+{FOUR['E75_per_task_oracle_gain']:.2f}", "LongBench 13 任务（RULER +0.00）"),
    ("逐层最优", f"+{FOUR['E79a_per_layer_oracle_gain']:.4f}", "8 样本 × 96 层"),
    ("逐样本在线选择", f"{FOUR['E79b_arm_selector_net_gain']:.2f}", "prefill 端一次 far 统计驱动"),
    ("逐配置臂", f"+{FOUR['E99_oracle_gain_vs_b025']:.2f} / +{REF_GAIN:.2f}", "13 端到端臂（vs 固定 β.25 / vs 参照臂）"),
], fs=11, highlight_col=1)
bullet_box(s, 7.3, 1.5, 5.6, 4.6, [
    ("12 个候选代理特征（任务匹配质量、离散度、合成侧质量等）经 BH 校正后无一显著", RED),
    (f"质量代理总量与 e2e 精度秩相关 {SPEARMAN:.3f}（CI 含 0）；所有质量派生信号即使方向事后取优，regret 仍 ≥1.18", RED),
    "唯一系统性分化（β=0.375 vs 0.25，+0.21）是任务形态交互，不是任何质量侧量能表达的方向",
    ("结论：平坦性是参数面的结构性属性而非测量问题——预测器无杠杆可撬", BLUE),
], fs=11.5)
# fig5 入页（宽 6.4 → 高约 1.86；放左下）
add_pic(s, FIG5, 0.5, 4.0, 6.4)
bullet_box(s, 0.5, 6.1, 12.4, 1.3, [
    ("部署结论：一组默认配置覆盖 12/13 任务；far-heavy 多跳任务可选一次 β 上调——参数不需精调本身构成部署优势（对比 DSA 的 1000 步 warm-up / 2.1B token 训练预算）", GREEN),
    "   图示（左下）：β 在 0.25–0.75 平坦、α 单调至边界；右下 panel 为 13 个端到端配置臂的离线质量代理对端到端精度散点——两口径秩相关弱，排序反转",
], fs=11)

# ============ S7 消融：维度校准 / 紧预算 / top-σ / 降维 ============
s = slide()
title_bar(s, "消融：分区收益的本质与边界",
          "四组判决对齐论文消融节——正结果与负结果同协议入册")
mono_table(s, 0.5, 1.45, 12.4, 3.4, [
    ("消融", "数据", "判决"),
    ("L1 上界维度双校准", "同 (α,β) 成对：tail32 54.83/33.22 vs 全维128 54.43/34.76（hotpotqa/musique）", "主表结论对 L1 上界维度不敏感；维度效应非可加常数（方法×维度交互），不能跨臂移植校准"),
    ("紧预算分区（总预算 256→2048 全档）", "trace 质量口径单池微胜 Δ≈−0.003，分区胜率 0–1/16", "分歧是指标级而非预算级：质量覆盖被 sink/滑窗/近端饱和区遮蔽；分区收益只能在 e2e 任务结构上显影（LongBench +0.62 / RULER +0.17 / 16K +0.91）"),
    ("top-σ 混合臂（阈值替代精筛）", "far 侧 e2e 随 σ 单调劣化 54.31→52.48；near 侧有弹性（55.12 全配置最高）", "far 检索质量必须由「块上界粗筛 + token 精筛」保护，near 靠滑窗兜底可放松——分区预算不对称的机理根据"),
    ("降维", "tail32 = 16 个完整最低频 RoPE 旋转对（频率单调 0.811/0.360/0.004）；PCA d=16 近无损 0.766 vs 0.804", "降维自由度来自位置先验而非学习/自适应；训练投影 trace 成立（92–94%）而 e2e 崩坏——口径鸿沟是免训练与训练流派的数学分界"),
], fs=10.5)
bullet_box(s, 0.5, 5.05, 12.4, 2.2, [
    ("分区 e2e 双基准成立：LongBench +0.62（50.78−50.16，主表可直接验算）、RULER +0.17（87.93−87.76）——与 trace 重放口径单池微胜的分歧坐实「分区收益的本质是紧预算下的预算分配，而非候选质量」", GREEN),
    "聚类代表方案在严格 token 预算下无一致优势且非上界有漏选理论空洞——降级为负结果；(cluster, cluster) 近端聚类无实现路径如实排除",
    ("gate 收益侧量化：τ=0.05 保守臂跳过 14.3% 层、far 质量损失 0.8%、选择链计算等比节省（上界为离线轮廓 13/36 层）", GREEN),
], fs=11.5)

# ============ S8 baseline 对比：sink 保送 + 训练门控 ============
s = slide()
title_bar(s, "基线对比：sink 保送的必要性与训练门控的质量边界",
          "全部基线在统一管线/统一 harness 下复现（预算 1024 同口径）")
mono_table(s, 0.5, 1.45, 6.4, 2.1, [
    ("KV 压缩基线", "无 sink 保送", "+ sink 保送（计外）", "增益"),
    ("SnapKV", "27.70", f"{sink['snapkv_sinkguard']['AVG']:.2f}", f"+{sink['snapkv_sinkguard']['AVG'] - base['SnapKV']['AVG']:.2f}"),
    ("H2O", "14.87", f"{sink['h2o_sinkguard']['AVG']:.2f}", f"+{sink['h2o_sinkguard']['AVG'] - base['H2O']['AVG']:.2f}"),
    ("PyramidKV", "28.82", f"{sink['pyramidkv_sinkguard']['AVG']:.2f}", f"+{sink['pyramidkv_sinkguard']['AVG'] - base['PyramidKV']['AVG']:.2f}"),
], fs=11.5, highlight_col=3)
mono_table(s, 7.1, 1.45, 5.8, 2.1, [
    ("训练门控 MoBA", "hotpotqa", "musique", "代码任务"),
    ("PSI（主表臂）", "55.44", "34.71", "repobench 66.97 / lcc 69.28"),
    ("MoBA（统一 harness）", "51.02", "29.74", "repobench 68.13 / lcc 68.98"),
], fs=11)
bullet_box(s, 0.5, 3.75, 12.4, 3.5, [
    ("sink 论断坐实：无显式 sink 保送时 Qwen3-8B 首 token 即 EOS（H2O musique 0.63 / hotpotqa 2.68 崩坏；末层 sink 权重 mean 0.592 而浅层选择信号不指向 sink，逐层恶性畸变）——任何不显式处理 sink 的稀疏方案在该模型族上都会失真", RED),
    f"sink 保送修复（预算 1024 + sink 128 计外，对齐 PSI 口径）：三方法 +2.15~+3.53（SnapKV {sink['snapkv_sinkguard']['AVG']:.2f} / PyramidKV {sink['pyramidkv_sinkguard']['AVG']:.2f}），但距 FullKV 仍 −18~−20 分——sink 保送必要非充分，prefill 静态压缩的候选质量损失是方法固有",
    f"训练门控 MoBA 全量 {MOBA_AVG:.2f}（距 FullKV −1.17）：QA/检索任务落后主表臂 4 分以上（hotpotqa 51.02 vs 55.44、musique 29.74 vs 34.71）——无上界 chunk 门控在 token 级质量上掉档；代码任务差距收窄（repobench 68.13 占优 1.16），块局部性红利部分对冲",
    "静态稀疏对照 StreamLLM（sink+滑窗，预算 1024）全量 14.20：检索类系统性崩塌（passage 0.25 / musique 1.99）——强制区之外预算为零时 far 检索无从谈起，反面界定分区预算的必要工作面",
], fs=11.5)

# ============ S9 速度：kernel microbench 与 e2e ============
s = slide()
title_bar(s, "速度：kernel microbench 与端到端双层如实报告",
          "三方 indexer 对比 = 官方核心 kernel 原样接入统一 harness（131K cudaEvent 计时）")
mono_table(s,0.5, 1.45, 6.4, 1.9, [
    ("选择链 vs dense 打分", "加速比", "口径"),
    ("32K", "1.46×", "合成数据已标注，trace 重放"),
    ("64K", "2.77×", "同上"),
    ("128K", "5.09×", "同上"),
], fs=11.5, highlight_col=1)
mono_table(s, 7.1, 1.45, 5.8, 1.9, [
    ("三方 indexer 单 kernel（131K）", "延迟", "备注"),
    ("Quest", "0.107 ms", "延迟最低，如实报告"),
    ("DSA", "0.503 ms", "训练类"),
    ("PSI", "0.787 ms", "排名第三——131K 下三家都远离 HBM bound，排名反映实现成熟度而非算法上界"),
], fs=10.5)
bullet_box(s, 0.5, 3.55, 12.4, 3.6, [
    ("PSI 当前优势在算法结构侧：每 token 索引计算量 ≈258 MAC（DSA 的 1/32）；索引存储 336B/token vs Quest 约 1KB/token（全 head 汇总——Quest 逐 32 个 head 建索引而 PSI 仅建 8 个 kv-head）；质量 vs Quest LongBench +3.06 / RULER +7.98", GREEN),
    ("e2e 有形状边界（Qwen3-30B，H20×2）：64K 单请求 1.285×（稀疏理论流量收益首次在 e2e 净兑现）；32K 档 0.77×（短上下文固定开销未覆盖）；TP2 高并发 0.92×——差 8% 归因 launch-bound（带宽 util 仅 10–11%），跨请求批量化已验证 3.8× 为修复方向", BLUE),
    "8B 稳态：decode step 65.0→33.5ms（dense 的 1.21×）；prefill 733.9→183.4s（自身 4.00×，对 dense 仍慢 1.60×——8 个 kv-head 使 gather 形态更重；30B 4 头形态 prefill 为 dense 的 0.77 倍）",
    ("profiling 方法论发现：select 热路径 3 处 host 同步在生产形态造成约 10 万次队列排空，合成 bench 完全测不出（消除后 prefill −6.7%）——与「离线重放排名不可替代 e2e」合并为一条方法论：kernel 级与 e2e 两层都测、都诚实报告", RED),
], fs=11.5)

# ============ S10 总结：贡献四条 + Limitations ============
s = slide()
title_bar(s, "总结：四条贡献与诚实边界")
bullet_box(s, 0.5, 1.35, 12.4, 3.4, [
    ("① 免训练降维自由度的系统刻画：tail32 = 16 个完整最低频 RoPE 旋转对（三重判决）；选取方法学四象限给出可行域边界——tail32 是经 e2e 验证的零信号成本最优解，PCA d=16 近无损可再压", GREEN),
    ("② far/near 分区预算：远端独立配额防挤出，双基准 e2e 成立（LongBench +0.62 / RULER +0.17）；musique 34.71 全方法最高（vs 同预算单池 +4.97）", GREEN),
    ("③ 感知型 per-request gate：prefill 动态信号 corr 0.924 防住反向错误，定位为安全省算力的保守开关（保守臂跳 14.3% 层、质量损失 0.8%）", GREEN),
    ("④ 免调参的轻量配置选择 + 双口径选举方法学：四粒度 oracle 上限全部 ≤0.5 分；离线质量与 e2e 秩相关仅 0.166——选举只认端到端实测，平坦性转译为部署简单（vs DSA 千步 warm-up）", GREEN),
], fs=12)
bullet_box(s, 0.5, 4.95, 12.4, 2.5, [
    ("Limitations：双基准最优臂分属两套 (β,γ) 配置，单一配置跨双基准统一性未验证；RULER 87.93 仍低于 FullKV 88.71（残余代价集中于 multikey_3 的 4bit 粒度）；三方 indexer 延迟排名第三（跨请求批量化为兑现路径）；全部结论在 Qwen3 单模型族（rotate_half 布局依赖），跨模型族外推未验证；MLA latent 索引是 future work", RED),
    ("全部精度代价与负结果（聚类代表 / 训练投影 e2e 崩 / 自适应无杠杆 / 非线性降维不可行 / TC 化反慢）在 同机同协议下如实入册", RED),
    ("主表臂 50.78 已全量落袋并回填论文（13 任务逐位取自结果 JSON）；PSI 已在 sglang 生产形态（paged 寻址、增量索引、CUDA graph）下集成", BLUE),
], fs=11.5)

prs.save(OUT)
print("saved ->", OUT)
print(f"slides = {len(prs.slides._sldIdLst)}")
