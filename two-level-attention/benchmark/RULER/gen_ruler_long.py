# E109 三件套 #3：RULER 64K/128K 数据生成器（离线零外网）
#
# 背景：KVCache-Factory 预生成数据只有 4096/8192/16384/32768 四档，
# 65536/131072 档需自建。设计文档（research/docs/逐层参数求解器_数学原理
# 与实验设计.md §10.4）要求：64K/128K 单独作为 YaRN 扩展 profile、
# 最大 prompt 由总容量减输出及包装开销确定（不机械复制 127.5K 常数）、
# 任务语义保持（needle 数量/位置分布/expected answer 不变）。
#
# 生成方法（循环体扩展，token 线性度实测 1.9991/3.9974 ≈ 精确线性）：
#   input = prefix(指令行) + body(噪声+needle 混合体) + suffix(问题行)
#   新 input = prefix + body×k(整倍) + body 截断段 + suffix
#   - body 整体循环：needle 随 body 一起均匀重复（位置分布保持均匀、
#     密度不变、key-value 对逐字相同 → answer 不变，multikey 干扰 needle
#     同样重复不引入歧义）。与官方 RULER 的 essay 循环填充同构，最机械保真。
#   - 截断点对齐到行/句/词边界（绝不落在 needle 内部）。
#   - cwe 特殊处理：编号列表循环后重新连续编号（1..N），高频词频率比例
#     与 top-10 不变；fwe/vt/niah 均可直接循环。
#   - token 数按 Qwen3 tokenizer 实测校准（每样本 tokenize 源一次，
#     n_target = 档位 - 64(max gen) - 8(包装余量)），生成后抽样复验 ±10%。
#
# 用法（在 two-level-attention 仓库根执行）：
#   python -u -m benchmark.RULER.gen_ruler_long \
#     --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
#     --model-path /mnt/dolphinfs/.../Qwen3-8B \
#     --lengths 65536 131072 [--tasks niah_single_1 ...] [--max-num 100]
import argparse
import json
import os
import re
import sys

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
MAX_GEN = 64        # 官方 RULER 统一 max_new_tokens
WRAP_MARGIN = 8     # chat/prompt 包装开销余量
# needle 正则（cwe/fwe 无 needle，走专用路径；vt 的 VAR 行无句点结尾）
NEEDLE_RE = {
    "niah": re.compile(r"One of the special magic (?:numbers|uuids|words) "
                       r"for \S+ is: [^.\n]+\."),
    "vt": re.compile(r"VAR [A-Z]+ = \d+"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True,
                    help="KVCache-Factory data/RULER 目录（32768 源在此，"
                         "65536/131072 生成到同根）")
    ap.add_argument("--model-path", type=str, required=True,
                    help="Qwen3-8B 本地权重（仅用 tokenizer）")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[65536, 131072], choices=[65536, 131072])
    ap.add_argument("--tasks", type=str, nargs="*", default=RULER_TASKS)
    ap.add_argument("--max-num", type=int, default=100,
                    help="每任务生成样本数（源文件为 100）")
    ap.add_argument("--source-length", type=int, default=32768,
                    help="源数据档位")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="token 校验容忍度（相对目标档）")
    return ap.parse_args()


def split_input(task, s):
    """把 input 切成 prefix / body / suffix 三段。
    - suffix = 最后一行（question 行：niah 系是 "\\nWhat is the special
      magic ..."，cwe/fwe/vt 是 "\\nQuestion: ..."，均为 input 最后一行）
    - niah 系/vt/cwe：prefix = 第一行指令（含换行）
    - fwe：指令与编码文本同行（无换行分隔），用指令尾锚点切分
    """
    i = s.rfind("\n")
    if i == -1:
        raise ValueError(f"[{task}] 找不到 question 行")
    suffix = s[i:]
    if task == "fwe":
        m = re.match(r"^(.*?coded words\.\s)", s)
        prefix = m.group(1)
        body = s[len(prefix):i]
    else:
        j = s.find("\n")
        prefix = s[:j + 1]
        body = s[j + 1:i]
    return prefix, body, suffix


def align_cut(task, body, cut):
    """截断点对齐：优先行边界（\\n），其次空格；绝不落在 needle 中间。
    needle 内部不含 \\n，行对齐天然保证 needle 完整；空格对齐时再做
    needle 完整性回退。"""
    if cut >= len(body):
        return len(body)
    j = body.rfind("\n", 0, cut)
    if j > cut - 2000:      # 行边界距截断点 2000 字符内 → 用行边界
        return j + 1
    j = body.rfind(" ", 0, cut)
    if j <= 0:
        return cut
    seg = body[:j]
    # 空格边界仍可能落在 needle 内部（essay 内联 needle）→ 回退到 needle 前
    pat = NEEDLE_RE["vt"] if task == "vt" else NEEDLE_RE["niah"]
    for m in list(pat.finditer(seg))[-4:]:
        if m.start() < j < m.end():
            return m.start()
    return j + 1


def expand_body(task, body, n_body_tok, n_add_tok, chars_per_tok):
    """把 body 循环扩展到约 (n_body_tok + n_add_tok) 个 token。
    返回 (新 body, 新 body 的 token 估算值)。整倍循环 + 对齐截断段，
    截断点绝不落在 needle 内部（align_cut 保证）。"""
    total_tok = n_body_tok + n_add_tok
    m = max(1, int(total_tok // max(n_body_tok, 1)))
    frac_tok = total_tok - m * n_body_tok
    out = body * m
    est = m * n_body_tok
    if frac_tok > 0:
        cut_chars = align_cut(task, body, int(frac_tok * chars_per_tok))
        if cut_chars > 0:
            out = out + body[:cut_chars]
            est += int(cut_chars / chars_per_tok)
    return out, est


def expand_cwe(body, n_body_tok, n_add_tok, chars_per_tok, corr=1.0):
    """cwe 专用：编号词列表循环 + 重新连续编号（保持高频词频率比例与
    top-10 集合不变，outputs 无需改动）。返回 (新 body, token 估算)。
    corr：编号位数增长的 token 校正系数（gen_task 用样本 0 实测校准，
    否则 131072 档编号从 4 位变 6 位会把 token 顶高 ~10%）。"""
    words = re.findall(r"\d+\. (\S+)", body)
    if not words:
        raise ValueError("cwe 解析失败：无编号词")
    tok_per_word = n_body_tok / len(words) * corr
    total_words = int((n_body_tok + n_add_tok) / tok_per_word)
    total_words = max(len(words), total_words)
    new_words = words * (total_words // len(words))
    new_words += words[:total_words % len(words)]
    out = " ".join(f"{i+1}. {w}" for i, w in enumerate(new_words))
    return out, len(new_words) * tok_per_word


def gen_task(tok, args, task, target_len, src_dir, dst_dir):
    src_file = os.path.join(src_dir, f"{task}.jsonl")
    rows = [json.loads(l) for l in open(src_file)]
    rows = rows[:args.max_num]

    # 每样本结构切分（同一任务 prefix/suffix 内容相同）
    parts = [split_input(task, r["input"]) for r in rows]
    prefix, _, suffix = parts[0]

    # 批量 tokenize：源 input 全体 + prefix/suffix 固定段（精确固定开销）
    enc = tok([r["input"] for r in rows] + [prefix + suffix],
              add_special_tokens=False).input_ids
    n_src = [len(e) for e in enc[:-1]]
    n_fixed = len(enc[-1])

    n_target = target_len - MAX_GEN - WRAP_MARGIN
    # cwe 编号位数增长的 token 校正：用样本 0 试生成一次实测校准
    cwe_corr = 1.0
    if task == "cwe":
        r0, (pre0, body0, suf0), n00 = rows[0], parts[0], n_src[0]
        nb0 = n00 - n_fixed
        n_add0 = max(0, n_target - n_fixed - nb0)
        b_est, est0 = expand_cwe(body0, nb0, n_add0, len(body0) / max(nb0, 1))
        n_real0 = len(tok(pre0 + b_est + suf0,
                          add_special_tokens=False).input_ids)
        if est0 > 0 and abs(n_real0 - n_target) > 0.02 * n_target:
            cwe_corr = n_real0 / n_target
            print(f"[gen] cwe corr: est={est0:.0f} real={n_real0} "
                  f"-> corr={cwe_corr:.4f}", flush=True)
    out_rows, stats = [], []
    for r, (pre, body, suf), n0 in zip(rows, parts, n_src):
        n_body = n0 - n_fixed
        chars_per_tok = len(body) / max(n_body, 1)
        n_add = max(0, n_target - n_fixed - n_body)
        if task == "cwe":
            new_body, est = expand_cwe(body, n_body, n_add, chars_per_tok,
                                       corr=cwe_corr)
        else:
            new_body, est = expand_body(task, body, n_body, n_add,
                                        chars_per_tok)
        new_input = pre + new_body + suf
        out_rows.append({
            "index": r["index"],
            "input": new_input,
            "outputs": r["outputs"],
            "length": n_fixed + int(est),
        })
        stats.append((n0, out_rows[-1]["length"]))

    # 写盘
    os.makedirs(dst_dir, exist_ok=True)
    dst_file = os.path.join(dst_dir, f"{task}.jsonl")
    with open(dst_file, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 抽样复验（5 样本真实 tokenize）+ 语义完整性断言
    import random as _rd
    _rd.seed(0)
    idx = _rd.sample(range(len(out_rows)), min(5, len(out_rows)))
    bad = []
    for i in idx:
        n_real = len(tok(out_rows[i]["input"], add_special_tokens=False
                         ).input_ids)
        if abs(n_real - n_target) > args.tol * n_target:
            bad.append((i, n_real))
        # expected answer 须仍在（且出现次数不少于源——循环只会增加不会减少）
        for ans in rows[i]["outputs"]:
            if (out_rows[i]["input"].count(ans) < rows[i]["input"].count(ans)
                    or ans not in out_rows[i]["input"]):
                bad.append((i, f"answer 缺失 {ans}"))
        # needle 数量不少于源（截断段可能带出部分 needle，非整倍属正常）
        if task != "cwe" and task != "fwe":
            pat = NEEDLE_RE["vt"] if task == "vt" else NEEDLE_RE["niah"]
            if len(pat.findall(out_rows[i]["input"])) < \
                    len(pat.findall(rows[i]["input"])):
                bad.append((i, "needle 丢失"))
    avg0 = sum(s[0] for s in stats) / len(stats)
    avg1 = sum(s[1] for s in stats) / len(stats)
    print(f"[gen] {task} L={target_len}: n={len(out_rows)} "
          f"src_avg_tok={avg0:.0f} new_avg_tok={avg1:.0f} "
          f"target={n_target} sample_check={'FAIL ' + str(bad) if bad else 'OK'}",
          flush=True)
    return not bad


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path,
                                        trust_remote_code=True)
    src_dir = os.path.join(args.data_root, str(args.source_length))
    all_ok = True
    for L in args.lengths:
        dst_dir = os.path.join(args.data_root, str(L))
        for task in args.tasks:
            ok = gen_task(tok, args, task, L, src_dir, dst_dir)
            all_ok = all_ok and ok
    print("ALL OK" if all_ok else "CHECK FAILED", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
