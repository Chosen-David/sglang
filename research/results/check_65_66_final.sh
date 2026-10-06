#!/bin/bash
# #65/#66 晨班一键收尾检查（2026-09-28）
# 用法：bash check_65_66_final.sh
echo "===== 1) TIA RULER L16384 / 终表 ====="
N=$(ls /home/wangyuanshuo02/two-level-attention/exp/results_ruler/Qwen3-8B/L16384/pred_1024/ 2>/dev/null | grep -c tia)
echo "L16384 TIA 完成: $N/11"
ps aux | grep pred_ruler | grep -v grep | grep -oE "\-\-task [a-z0-9_]+ --context_length [0-9]+ --method [a-z]+" || echo "（无跑批进程）"
if [ -f /home/wangyuanshuo02/two-level-attention/exp/results_ruler/ruler_table_final.json ]; then
    echo "--- 四方法终表已生成："
    python3 - <<'EOF'
import json
res = json.load(open('/home/wangyuanshuo02/two-level-attention/exp/results_ruler/ruler_table_final.json'))
methods = ["none", "quest_64_16", "tia_64_128_1024_c4", "tli_64_128_1024_c4_ABD"]
names = {"none": "FullKV", "quest_64_16": "Quest", "tia_64_128_1024_c4": "TIA",
         "tli_64_128_1024_c4_ABD": "TLI"}
for L in ("L4096", "L8192", "L16384"):
    row = []
    for m in methods:
        d = res.get(f"{L}/{m}")
        if d:
            avg = sum(d.values()) / len(d)
            row.append(f"{names[m]}={avg:.2f}")
        else:
            row.append(f"{names[m]}=缺")
    print(f"{L}: " + "  ".join(row))
EOF
else
    echo "终表未生成（watcher 未跑到或 TIA 未收官）"
fi

echo ""
echo "===== 2) 30B TP2 复测（#65）====="
if [ -f /tmp/tli_64k_tp2_v2.log ]; then
    tail -6 /tmp/tli_64k_tp2_v2.log
    echo "--- 结果对比："
    python3 - <<'EOF'
import json, os
p = '/home/wangyuanshuo02/sglang/tli_64k_tp2_tli.json'
if os.path.exists(p):
    d = json.load(open(p))
    t = d.get('total_s')
    print(f"tli total_s = {t}  vs triton 基线 106.24  → {'✅ 翻正(≥1.0×)' if t and t <= 106.24 else '未翻正 ' + f'({106.24/t:.3f}×)' if t else '无数据'}")
    print(f"vs 旧 tli 167.12 → 提升 {167.12/t:.2f}×" if t else "")
else:
    print("结果 json 未生成（复测未跑或进行中）")
EOF
else
    echo "复测未启动（TP2 log 不存在）"
fi

echo ""
echo "===== 3) watcher / GPU ====="
tail -3 /tmp/night_relay.log 2>/dev/null
ps -p 204147 -o pid,etime --no-headers 2>/dev/null && echo "watcher 存活" || echo "watcher 已结束（或死亡——若 TIA 未收官且 GPU1 空闲则需人工重启复测）"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
