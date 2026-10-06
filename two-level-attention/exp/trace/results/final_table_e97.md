| method | config | trace mass | e2e | speed | note |
|---|---|---|---|---|---|
| PSI (minmax,avg) 分区 | α.125/β.375/γ.125 | 0.883 | LB 50.54 / hq 54.43 / mu 34.76 | 选择链 1.46×@32K→5.09×@128K；MACs/token≈258；336B/token | L1 上界全维 128 口径（L2 细筛两口径数学等价 tail32，E87c 配对校准：L1-tail 臂 hq 54.83/mu 33.22——主表结论对 L1 维度不敏感） |
| PSI 分区 (β.25 对照) | α.125/β.25/γ.125 | 0.8819 | LB 50.41 / hq 54.23 / mu 32.82 | 同上 |  |
| PSI 单池 (minmax) | bp128/γ.125 | 0.8979 | LB 50.16 | 同上（无分区开销略低） |  |
| mavg (minmax,avg) | E72 最优 α/β/γ | 0.883 | hq 54.23 / mu 32.82 | - | E72 冠军 β.375 |
| mminmax (minmax,minmax) | E72 最优 α/β/γ | 0.8424 | hq 53.27 / mu 34.14 | - |  |
| aavg (avg,avg) | E72 最优 α/β/γ | 0.7918 | hq 54.72 / mu 32.82 | - |  |
| cavg (cluster,avg) | E72 最优 α/β/γ | 0.8801 | hq 53.43 / mu 32.31 | — | ClusterKV-style 代表 |
| 两级 baseline (minmax,avg) | α.125/β.25/γ1 | 0.8819 | hq 54.23 / mu 32.82 | macs=76892.0, sel=2048.0 | trace 重放参照臂 |
| top-σ near_sigma_8.0 | σ=8.0 | 0.8596 | hq 55.12 / mu 31.83 | macs=130255.0, sel=1300.9 | 预算可变质量-预算旋钮 |
| top-σ far_sigma_8.0 | σ=8.0 | 0.7352 | hq 54.31 / mu 30.09 | macs=635809.0, sel=1972.5 | 预算可变质量-预算旋钮 |
| top-σ mid_sigma_8.0 | σ=8.0 | 0.7125 | hq 53.5 / mu 31.09 | macs=689172.0, sel=1225.4 | 预算可变质量-预算旋钮 |
| top-σ far_sigma_32.0 | σ=32.0 | 0.7522 | hq 52.48 / mu ? | macs=635809.0, sel=2287.4 | 预算可变质量-预算旋钮 |
| top-σ mid_sigma_32.0 | σ=32.0 | 0.7328 | hq 53.48 / mu ? | macs=689172.0, sel=1573.5 | 预算可变质量-预算旋钮 |
| mono vs 分区 @b256 | mavg 同预算 | -0.0027 | - | - | mono=0.8233 part=0.8206 分区胜 1/16 |
| mono vs 分区 @b512 | mavg 同预算 | -0.0029 | - | - | mono=0.8549 part=0.852 分区胜 0/16 |
| mono vs 分区 @b768 | mavg 同预算 | -0.0038 | - | - | mono=0.8754 part=0.8716 分区胜 1/16 |
| mono vs 分区 @b1024 | mavg 同预算 | -0.0037 | - | - | mono=0.8902 part=0.8866 分区胜 1/16 |
| mono vs 分区 @b2048 | mavg 同预算 | -0.0035 | - | - | mono=0.9271 part=0.9236 分区胜 0/16 |
| FullKV | 官方配置 | - | LB 50.36 | - |  |
| Quest | 官方配置 | - | LB 47.72 | - |  |
| TIA | 官方配置 | - | LB 50.06 | - |  |
| SnapKV | 官方配置 | - | LB 27.7 | - |  |
| H2O | 官方配置 | - | LB 14.87 | - |  |
| PyramidKV | 官方配置 | - | LB 28.82 | - |  |
| StreamingLLM | budget=1024 | - | LB 14.2 / hq 2.9 / mu 1.99 | - | E89 全量 13 任务：静态稀疏崩塌证据 |
| MoBA (统一harness) | chunk gate top-16 块 | - | LB 49.19 / hq 51.02 / mu 29.74 | - | E89 复现 13 任务全量：QA/检索落后 PSI 3+/代码任务反超（repobench 68.13 vs 66.67）——无上界 chunk gate 掉档 |
| E90 子空间 full | twolvl α.125/β.25/γ.125 | - | hq 54.72 / mu 32.82 | - | 五臂排序（hq）：rope64 55.98 > full128 54.72 > tail32 54.23 > nope64 52.4 >> random32 46.01 >> highfreq 33.8；崩塌级双口径同向 |
| E90 子空间 rope | twolvl α.125/β.25/γ.125 | - | hq 55.98 / mu 32.96 | - | 五臂排序（hq）：rope64 55.98 > full128 54.72 > tail32 54.23 > nope64 52.4 >> random32 46.01 >> highfreq 33.8；崩塌级双口径同向 |
| E90 子空间 nope | twolvl α.125/β.25/γ.125 | - | hq 52.4 / mu 30.8 | - | 五臂排序（hq）：rope64 55.98 > full128 54.72 > tail32 54.23 > nope64 52.4 >> random32 46.01 >> highfreq 33.8；崩塌级双口径同向 |
| E90 子空间 random | twolvl α.125/β.25/γ.125 | - | hq 46.01 / mu 24.08 | - | 五臂排序（hq）：rope64 55.98 > full128 54.72 > tail32 54.23 > nope64 52.4 >> random32 46.01 >> highfreq 33.8；崩塌级双口径同向 |
| E90 子空间 highfreq | twolvl α.125/β.25/γ.125 | - | hq 33.8 / mu 14.26 | - | 五臂排序（hq）：rope64 55.98 > full128 54.72 > tail32 54.23 > nope64 52.4 >> random32 46.01 >> highfreq 33.8；崩塌级双口径同向 |
| top-σ near σ8 (L1-tail 校准臂) | σ=8, L1 上界 tail32 | - | hq 55.08 / mu 31.19 | - | E87c：top-σ 判决不受 L1 维度混杂影响（hq 维度差 −0.04 噪声级） |
| mavg β.375 (L1-tail 校准臂) | α.125/β.375, L1 上界 tail32 | - | hq 54.83 / mu 33.22 | - | E87c：L1 维度效应在 mavg 臂方向反转（tail +0.40）——非可加常数、method×dimension 交互 |
