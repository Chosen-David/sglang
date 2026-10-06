# E67：感知型 per-request gate（用户 2026-09-29 设想：测完一条数据的 prefill 阶段
#   就得到哪些层可以跳 far / 跳 near，应用到 decode 上提速——非静态 D'，非逐步在线）
#
# 动机链：
#   E5b 教训 = 静态层掩码跨任务不泛化（层轮廓跨任务 corr 0.05-0.89）、
#   per-task 离线校准也失败（长度/GQA 口径失配）；但 E5 发现同任务内 corr 0.93-1.00
#   ——用户方案 = 校准信号来自「该请求自己的 prefill」，同请求内预测，正中数据支持方向。
#   dyngate（在线逐步）已证 e2e 无损（−0.17 噪声级），本实验测的是「prefill 一次决策
#   + decode 全程生效」的更省形态（决策成本 ≈ 0，无逐步开销）。
#
# 设计（trace 级，CPU）：
#   信号 = 每层 prefill q（chunk 末 q，prefill 本来就要算的注意力）的 far mass 占比
#   决策 = far_mass_prefill < τ 的层 → decode 跳 far（只算 near+sink+swa）
#          near_mass_prefill < τ_n 的层 → decode 跳 near（只算 far+sink+swa）
#   验证 = decode q（步长 1 的 255 个）真实 mass 损失：
#          被跳层在 decode 分布下贡献的 far/near mass
#   校准变体 = last1（仅最后 prefill chunk q，最接近 decode 分布）
#              avg16（16 个 prefill q 平均，更稳）
#   基线 = 静态 D'（全局层掩码，跨样本共享）——同 τ 下 precision 对比
#   阈值 τ 扫描 = 0.01/0.02/0.05/0.10/0.20 → 跳层率 × mass 损失帕累托
#   区域定义（严格口径）= sink 128 / swa 1024 / mid 其余 / near = mid 靠 q 4096 / far = mid 其余
import json
import os

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e67_perrequest_gate.json"
SINK, SWA, NEAR_BAND = 128, 1024, 4096
TAUS = [0.01, 0.02, 0.05, 0.10, 0.20]
SAMPLES = ["lb_gov_report_0", "lb_hotpotqa_0", "lb_musique_0", "lb_narrativeqa_0",
            "lb_passage_retrieval_en_0", "lb_qasper_0", "needle32k", "natural32k",
            "lb_gov_report_1", "lb_hotpotqa_1", "lb_musique_1", "lb_narrativeqa_1",
            "lb_passage_retrieval_en_1", "lb_qasper_1"]


def load(lf):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].float(), d["q"].float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    diffs = qpos[1:] - qpos[:-1]
    dec_i = int(torch.where(diffs == 1)[0][0]) + 1        # decode 段起点
    pre_i = list(range(0, dec_i))                          # prefill chunk 末 q
    pos = torch.arange(S)
    return dict(k=k, q=q, qpos=qpos, S=S, Hkv=Hkv, G=G,
                pre_i=pre_i, dec_i=dec_i, pos=pos)


def mass_profile(td, qi):
    """单个 q 的区域 mass 占比 → (far_frac, near_frac)。"""
    qg = td["q"][qi].reshape(td["Hkv"], td["G"], 128)
    k4 = td["k"][:, :, None, :].expand(td["S"], td["Hkv"], td["G"], 128)
    t = int(td["qpos"][qi])
    s = torch.einsum("hgd,shgd->hgs", qg, k4) * (128 ** -0.5)
    s = s.masked_fill((td["pos"] > t).view(1, 1, -1), float("-inf"))
    p = torch.softmax(s, dim=-1)                            # [Hkv,G,S]
    tot = p.sum()
    far = p[..., :td["S"] - SWA - NEAR_BAND].sum()
    near = p[..., td["S"] - SWA - NEAR_BAND:td["S"] - SWA].sum()
    return float(far / tot), float(near / tot)


def main():
    torch.set_num_threads(10)
    results = {}
    for name in SAMPLES:
        if not os.path.isfile(f"{TRACE}/{name}/meta.json"):
            continue
        n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
        layers = list(range(0, n_layers, max(1, n_layers // 12)))
        rec = {"skip_far": {}, "skip_near": {}}
        for li in layers:
            td = load(f"{TRACE}/{name}/layer{li:02d}.pt")
            if td["dec_i"] >= len(td["qpos"]) - 8:
                continue
            # ---- prefill 信号（avg16 + last1 两变体）----
            pre_far, pre_near = [], []
            for qi in td["pre_i"]:
                f, n = mass_profile(td, qi)
                pre_far.append(f); pre_near.append(n)
            sig_avg16 = (sum(pre_far) / len(pre_far), sum(pre_near) / len(pre_near))
            sig_last1 = (pre_far[-1], pre_near[-1])
            # ---- decode 真实分布（抽样 16 个 decode q 均值，速度考虑）----
            dec_qs = list(range(td["dec_i"], len(td["qpos"]), max(1, (len(td["qpos"]) - td["dec_i"]) // 16)))[:16]
            dec_far, dec_near = [], []
            for qi in dec_qs:
                f, n = mass_profile(td, qi)
                dec_far.append(f); dec_near.append(n)
            dec_far_m = sum(dec_far) / len(dec_far)
            dec_near_m = sum(dec_near) / len(dec_near)
            for tau in TAUS:
                for sig_name, (sf, sn) in (("avg16", sig_avg16), ("last1", sig_last1)):
                    # 跳 far 决策：prefill far 占比 < τ → decode 跳 far
                    if sf < tau:
                        rec["skip_far"].setdefault(f"tau{tau}_{sig_name}", []).append(
                            dict(layer=li, pre_far=sf, dec_far=dec_far_m))
                    # 跳 near 决策：prefill near 占比 < τ → decode 跳 near
                    if sn < tau:
                        rec["skip_near"].setdefault(f"tau{tau}_{sig_name}", []).append(
                            dict(layer=li, pre_near=sn, dec_near=dec_near_m))
            rec.setdefault("dec_profiles", []).append(
                dict(layer=li, dec_far=dec_far_m, dec_near=dec_near_m,
                     pre_far_avg16=sig_avg16[0], pre_near_avg16=sig_avg16[1],
                     pre_far_last1=sig_last1[0], pre_near_last1=sig_last1[1]))
        results[name] = rec
        n_sk = {k: len(v) for k, v in rec["skip_far"].items()}
        print(f"[{name}] layers={len(rec.get('dec_profiles', []))} "
              f"skip_far@tau0.05_avg16={n_sk.get('tau0.05_avg16', 0)}", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    # ---- 汇总：跳层率 × decode mass 损失帕累托 ----
    summary = {}
    for mode in ("skip_far", "skip_near"):
        for tau in TAUS:
            for sig in ("avg16", "last1"):
                key = f"tau{tau}_{sig}"
                tot_layers = sum(len(r.get("dec_profiles", [])) for r in results.values())
                skips = sum(len(r[mode].get(key, [])) for r in results.values())
                if mode == "skip_far":
                    lost = sum(e["dec_far"] for r in results.values() for e in r[mode].get(key, []))
                else:
                    lost = sum(e["dec_near"] for r in results.values() for e in r[mode].get(key, []))
                summary[f"{mode}/{key}"] = dict(
                    skip_ratio=round(skips / max(tot_layers, 1), 4),
                    mass_lost_total=round(lost / max(tot_layers, 1), 6),
                    n_skipped=skips, n_layers=tot_layers)
    json.dump(dict(per_sample=results, summary=summary),
              open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
