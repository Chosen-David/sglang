"""E109 aavg (α,β,γ) 网格图：有效配置去重（γ 饱和/坍塌裁剪）。
数据快照：/tmp/e109_scan_v2/prescore.json + exp/trace/results/e109_screen4_selection.json
生成：ref/figs/e109_aavg_abg_grid.{png,pdf} + e109_aavg_abg_grid_data.json
"""
import json, math, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# CJK 字体（DejaVu Sans 无中文字形；字体文件存在但不在 matplotlib 缓存）
for _f in ('/home/wangyuanshuo02/.local/share/fonts/cjk/NotoSansCJKsc-Regular.otf',):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
_CJK = 'Noto Sans CJK SC'

# ---- 数据 ----
prescore = json.load(open('/tmp/e109_scan_v2/prescore.json'))
sel = json.load(open('/home/wangyuanshuo02/sglang/two-level-attention/exp/trace/results/e109_screen4_selection.json'))
sel_tasks = sel.get('tasks', sel)

K1, BS, K2_MID = 128, 64, 768

def parse(tag):
    m = re.match(r'E109_aavg(?:CHAMP)?_a([\d.]+)_b([\d.]+)_g([\d.]+)', tag)
    return tuple(float(x) for x in m.groups()) if m else None

def effective(a, b, g):
    """返回 (nb_near, nt_near, far_budget, collapsed, gamma_saturated)。"""
    if a == 0 or b == 0:
        return (0, 0, K2_MID, False, False)  # 单池
    nb_near = max(1, int(round(K1 * b)))
    raw = int(nb_near * BS * g)
    nt_near = min(raw, K2_MID)
    far_budget = max(0, K2_MID - nt_near)
    return (nb_near, nt_near, far_budget, far_budget == 0, raw >= K2_MID)

arms = []
for tag, v in prescore.items():
    p = parse(tag)
    if not p:
        continue
    a, b, g = p
    nb, nt, fb, col, sat = effective(a, b, g)
    avg5 = sel_tasks.get(tag, {}).get('AVG5') if isinstance(sel_tasks.get(tag), dict) else None
    arms.append(dict(a=a, b=b, g=g, avg2=v['avg2'], hq=v['hotpotqa'], mu=v['musique'],
                     avg5=avg5, nb=nb, nt=nt, fb=fb, collapsed=col, saturated=sat))
# 单池参考
arms.append(dict(a=0, b=0, g=0, avg2=None, hq=57.26, mu=30.52, avg5=None,
                 nb=0, nt=0, fb=K2_MID, collapsed=False, saturated=False))

# 等价类：(α, nb_near, nt_near) 相同 → 行为等价（γ 死参数），只保留 γ 最小者
classes = {}
for r in arms:
    key = (r['a'], r['nb'], r['nt'])
    classes.setdefault(key, []).append(r)
for key, group in classes.items():
    group.sort(key=lambda r: r['g'])
    for i, r in enumerate(group):
        r['dup'] = i > 0  # 第一个为代表，其余为重复

n_dup = sum(1 for r in arms if r.get('dup'))
n_col = sum(1 for r in arms if r['collapsed'] and not r.get('dup'))
json.dump({'arms': arms, 'n_arms': len(arms), 'n_dup_pruned': n_dup, 'n_collapsed': n_col},
          open('/home/wangyuanshuo02/sglang/ref/figs/e109_aavg_abg_grid_data.json', 'w'), indent=1)

# ---- 设计说明（brief）----
# 读者问题：aavg (α,β,γ) 网格中哪些配置真正不同、表现如何、哪里是坍塌区
# 布局：行=β，列=γ，x=α，y=prescore avg2；AVG5 臂星标叠加
# 裁剪规则：等价类只画代表（γ 最小），dup 不画；坍塌臂(far_budget=0)红色
# 单池 (0,0) 为水平参考线；evidence_type=measured

plt.rcParams.update({
    'font.family': _CJK, 'font.size': 8.5,
    'axes.unicode_minus': False,
    'axes.linewidth': 0.7, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
})
betas = sorted({r['b'] for r in arms if r['b'] > 0})
gammas = sorted({r['g'] for r in arms if r['b'] > 0})
fig, axes = plt.subplots(len(betas), len(gammas), figsize=(2.1 * len(gammas), 2.0 * len(betas) + 0.6),
                         sharex=True, sharey=True, squeeze=False)
fig.subplots_adjust(hspace=0.4, wspace=0.12, left=0.07, right=0.98, top=0.86, bottom=0.16)

C_OK, C_COL, C_AVG5 = '#2166ac', '#b2182b', '#f4a029'
for i, b in enumerate(betas):
    for j, g in enumerate(gammas):
        ax = axes[i][j]
        cell = [r for r in arms if r['b'] == b and r['g'] == g]
        # 单池参考线（每 panel）
        ax.axhline((57.26 + 30.52) / 2, color='#666666', lw=0.9, ls='--', zorder=1)
        if j == 0 and i == 0:
            ax.text(0.98, 0.06, 'single-pool (0,0)=43.9', transform=ax.transAxes,
                    fontsize=6.5, color='#666666', ha='right')
        for idx, r in enumerate(sorted(cell, key=lambda r: r['a'])):
            if r.get('dup'):
                continue  # 裁剪：等价配置不重复画
            y = r['avg2']
            if y is None:
                y = (r['hq'] + r['mu']) / 2
            color = C_COL if r['collapsed'] else C_OK
            ax.plot(r['a'], y, 'o', ms=5, color=color, mfc='white' if r['collapsed'] else color,
                    mec=color, mew=1.2, zorder=3)
            if r['avg5'] is not None:
                ax.plot(r['a'], y, marker='*', ms=11, color=C_AVG5, mec='#8a5a00', mew=0.7,
                        ls='none', zorder=4)
                dy = 3 if idx % 2 == 0 else -12
                ax.annotate(f"{r['avg5']:.2f}", (r['a'], y), textcoords='offset points',
                            xytext=(6, dy), fontsize=7, color='#8a5a00', zorder=5)
            if r['saturated'] and not r['collapsed']:
                ax.annotate('sat', (r['a'], y), textcoords='offset points', xytext=(6, -9),
                            fontsize=6, color='#999999', zorder=5)
        dups = [r for r in cell if r.get('dup')]
        if dups:
            ax.text(0.98, 0.04, f"≡×{len(dups)} pruned", transform=ax.transAxes,
                    fontsize=6, color='#bbbbbb', ha='right', va='bottom')
        if r['collapsed'] if cell else False:
            pass
        ax.set_title(f'β={b}  γ={g}', fontsize=8, pad=3)
        if i == len(betas) - 1:
            ax.set_xlabel('α')
        if j == 0:
            ax.set_ylabel('prescore avg2 (hq+mu)/2')
        ax.set_xlim(-0.06, 1.0)
        ax.set_ylim(25.5, 47.5)
        ax.set_xticks([0.125, 0.5, 0.875])

handles = [
    plt.Line2D([], [], marker='o', color=C_OK, ms=5, ls='none', label='有效配置（去重后）'),
    plt.Line2D([], [], marker='o', color=C_COL, mfc='white', ms=5, ls='none', label='坍塌臂 far_budget=0'),
    plt.Line2D([], [], marker='*', color=C_AVG5, mec='#8a5a00', ms=11, ls='none', label='已落袋 AVG5（标注值）'),
    plt.Line2D([], [], color='#666666', ls='--', lw=0.9, label='单池 (0,0) 参考'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, 0.005))
fig.suptitle('E109 aavg (α,β,γ) 网格：有效配置视图（γ 饱和等价类裁剪 '
             f'{n_dup} 点，坍塌 {n_col} 点标红）', fontsize=10, y=0.975)

for ext in ('png', 'pdf'):
    fig.savefig(f'/home/wangyuanshuo02/sglang/ref/figs/e109_aavg_abg_grid.{ext}', dpi=220)
print('saved. arms =', len(arms), 'dup_pruned =', n_dup, 'collapsed =', n_col)
