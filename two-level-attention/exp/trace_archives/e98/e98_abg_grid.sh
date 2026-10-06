#!/bin/bash
# E98 α/β/γ 三维全网格 mass 采集：16 CPU 进程分样本（GPU 双卡留给 M6 e2e）
cd /home/wangyuanshuo02/two-level-attention/exp/trace
export PYTHONPATH=/home/wangyuanshuo02/two-level-attention/exp/trace:/home/wangyuanshuo02/two-level-attention:/home/wangyuanshuo02/.local/pylibs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for p in $(seq 0 15); do
  E98_PROC=$p E98_NPROC=16 E98_DEVICE=cpu \
    python3 analyze_e98_abg_full_grid.py > /tmp/e98_proc$p.log 2>&1 &
done
wait
# 合并 shard → 单一 JSON（样本 → 臂 → 均值，附全体均值）
python3 - << 'EOF'
import glob, json
out = {}
for f in sorted(glob.glob('/tmp/e98_shard_*.json')):
    out.update(json.load(open(f)))
arms = {}
for s in out:
    for k, v in out[s].items():
        arms.setdefault(k, []).append(v)
mean = {k: round(sum(v) / len(v), 4) for k, v in arms.items()}
json.dump({"per_sample": out, "mean": mean,
           "n_samples": len(out), "n_arms": len(mean),
           "note": "E98 alpha/beta/gamma 3D full grid (0.125 step), 5 method combos, "
                   "B_TOK=2048 BP=64; constraints per 论文indexer.md: near_L>=near_bp*BS, "
                   "far_budget<=far_L, corner guards"},
          open('/home/wangyuanshuo02/two-level-attention/exp/trace/results/e98_abg_full_grid.json', 'w'))
print('merged', len(out), 'samples', len(mean), 'arms')
EOF
echo "E98_GRID_DONE"
