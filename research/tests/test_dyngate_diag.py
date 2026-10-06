# #60 口径 bug 诊断：一条 musique 样本三变体对照（GPU 空闲窗口）
# A. pred.py 完整流程（q_pos 切分 + question 逐 token 喂入 + mind 模板）
# B. 全串一次 generate（mind 模板）
# C. 全串一次 generate（干净模板 assistant\n）
# 输出对比定位 sglang 版 0.27 vs 主表 32.28 的差异来源
import json
import sys

import torch

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention")

MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B"
_TPL = json.load(open(
    "/home/wangyuanshuo02/two-level-attention/benchmark/LongBench/config/dataset2prompt.json"))
MIND_TPL = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\nmind\n\n\n\n"
CLEAN_TPL = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"


def run():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    row = json.loads(open("/home/wangyuanshuo02/datasets/LongBench/data/musique.jsonl").readline())
    prompt = _TPL["musique"].format(**{k: row.get(k, "") for k in ("context", "input")})
    MAXGEN = 32

    def gen_full(p, tag):
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        out = model.generate(ids, max_new_tokens=MAXGEN, do_sample=False,
                             temperature=1.0, num_beams=1)
        pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"[{tag}] {pred[:120]!r}", flush=True)
        return pred

    # B. mind 模板全串
    gen_full(MIND_TPL.format(p=prompt), "B-mind-full")
    # C. 干净模板全串
    gen_full(CLEAN_TPL.format(p=prompt), "C-clean-full")

    # A. pred.py 完整流程（q_pos 切分 + question 逐 token）
    p = MIND_TPL.format(p=prompt)
    q_pos = max(len(p) - 100, p.rfind("Question:"))
    question = p[q_pos:]
    p_main = p[:q_pos]
    input_ids = tok(p_main, return_tensors="pt").input_ids.cuda()
    q_ids = tok(question, return_tensors="pt").input_ids[:, 1:].cuda()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        for tid in q_ids[0]:
            out = model(input_ids=tid.view(1, 1), past_key_values=past, use_cache=True)
            past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1).view(1, 1)
        gen = [nxt.item()]
        for _ in range(MAXGEN - 1):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1).view(1, 1)
            gen.append(nxt.item())
            if nxt.item() == tok.eos_token_id:
                break
    print(f"[A-predpy-full] {tok.decode(gen, skip_special_tokens=True)[:120]!r}", flush=True)
    print("answers:", row["answers"])


if __name__ == "__main__":
    run()
