"""E109 aavg (α,β,γ) 3D 有效配置图（静态双视角）。
数据快照：e109_aavg_abg_grid_data.json（由 e109_aavg_abg_grid.py 生成）。
特性：score 彩色散点 + 坍塌分界面（far_budget=0 临界曲面）+ AVG5 星标 + 等价裁剪 ghost 点。
生成：ref/figs/e109_aavg_abg_3d.{png,pdf}
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from matplotlib.lines import Line2D

for _f in ('/home/wangyuanshuo02/.local/share/fonts/cjk/NotoSansCJKsc-Regular.otf',):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 9, 'axes.unicode_minus': False})

K1, BS, K2_MID = 128, 64, 768
d = json.load(open('/home/wangyuanshuo02/sglang/ref/figs/e109_aavg_abg_grid_data.json'))
arms = d['arms']

# ---- 坍塌临界曲面：round(128β)·64·γ = 768 → γ_crit(β)（与 α 无关，沿 α 挤出）----
al_edge = np.linspace(0.0, 1.0, 2)
betas_edge = np.linspace(0.04, 0.94, 60)
A, B = np.meshgrid(al_edge, betas_edge)
nb = np.maximum(1, np.round(K1 * B)).astype(int)
G_crit = K2_MID / (BS * nb)          # γ 高于此面 → far_budget=0（坍塌区）
G_surf = np.clip(G_crit, 0.10, 1.0)

fig = plt.figure(figsize=(12.4, 5.6))
fig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.06, wspace=0.02)

norm = plt.Normalize(28, 47)
cmap = cm.get_cmap('RdYlGn')

def draw(ax, elev, azim, title):
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 0.75))
    # 坍塌临界面（红色半透明阶梯面，沿 α 全幅挤出）
    ax.plot_surface(A, B, G_surf, alpha=0.20, color='#d6604d',
                    edgecolor='#b2182b', lw=0.25, shade=False)
    for r in arms:
        x, y, z = r['a'], r['b'], r['g']
        score = r['avg2'] if r['avg2'] is not None else (r['hq'] + r['mu']) / 2
        if r.get('dup'):
            ax.scatter([x], [y], [z], s=22, c='#cccccc', marker='o', alpha=0.55, depthshade=False)
            continue
        if r['a'] == 0 and r['b'] == 0:
            ax.scatter([0], [0], [0], s=110, c='#666666', marker='D', depthshade=True)
            ax.text(0.02, 0.02, 0.06, '单池(0,0)', fontsize=7, color='#555555')
            continue
        c = '#b2182b' if r['collapsed'] else cmap(norm(score))
        face = 'none' if r['collapsed'] else c
        ax.plot([x, x], [y, y], [0.1, z], color='#bbbbbb', lw=0.6, zorder=1)  # 落影柱线
        ax.scatter([x], [y], [z], s=70 if r['avg5'] is None else 130,
                   c=[c] if not r['collapsed'] else ['#b2182b'],
                   marker='X' if r['collapsed'] else 'o',
                   edgecolors='#8a5a00' if r['avg5'] is not None else c,
                   linewidths=1.4 if r['avg5'] is not None else 0.6,
                   depthshade=True, zorder=3)
        if r['avg5'] is not None:
            ax.text(x, y, z + 0.035, f"{r['avg5']:.2f}", fontsize=7.5, color='#8a5a00',
                    ha='center', zorder=4)
    ax.set_xlabel('α（near 时域边界）', fontsize=8.5, labelpad=6)
    ax.set_ylabel('β（L1 near 块额）', fontsize=8.5, labelpad=6)
    ax.set_zlabel('γ（L2 near 配额）', fontsize=8.5, labelpad=4)
    ax.set_xlim(-0.05, 1.0); ax.set_ylim(0.0, 0.95); ax.set_zlim(0.1, 1.02)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.tick_params(labelsize=7.5)

ax1 = fig.add_subplot(121, projection='3d')
draw(ax1, 16, -58, '视角 1：γ 轴展开（坍塌面之上 = far_budget=0）')
ax2 = fig.add_subplot(122, projection='3d')
draw(ax2, 52, -72, '视角 2：俯视 β-γ 平面（β 大且 γ 高 → 坠入红区）')

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.62, pad=0.10, aspect=24)
cb.set_label('prescore avg2（hq+mu）/2', fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)

handles = [
    Line2D([], [], marker='o', color='#3a8f3a', ms=7, ls='none', label='有效配置（颜色=avg2）'),
    Line2D([], [], marker='X', color='#b2182b', ms=8, ls='none', label='坍塌臂（红面之上）'),
    Line2D([], [], marker='o', color='#3a8f3a', mec='#8a5a00', mew=1.5, ms=10, ls='none',
           label='已落袋 AVG5（金圈+标注）'),
    Line2D([], [], marker='D', color='#666666', ms=7, ls='none', label='单池 (0,0,0)'),
    Line2D([], [], marker='o', color='#cccccc', ms=5, ls='none', label='γ 饱和等价点（已裁剪）'),
]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle('E109 aavg (α,β,γ) 三维有效配置空间：红色阶梯面为 far_budget=0 坍塌临界',
             fontsize=11.5, y=0.97)

for ext in ('png', 'pdf'):
    fig.savefig(f'/home/wangyuanshuo02/sglang/ref/figs/e109_aavg_abg_3d.{ext}', dpi=220)
print('saved 3d figure')
