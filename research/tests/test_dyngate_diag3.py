# #60 诊断 3：sglang triton backend（全量注意力）+ mind 模板 vs transformers
import json
import warnings

warnings.filterwarnings("ignore")
from test_dyngate_diag2 import MODEL, _TPL, MIND_TPL


def main():
    row = json.loads(open(
        "/home/wangyuanshuo02/datasets/LongBench/data/musique.jsonl").readline())
    prompt = MIND_TPL.format(p=_TPL["musique"].format(
        **{k: row.get(k, "") for k in ("context", "input")}))
    from sglang import Engine
    eng = Engine(model_path=MODEL, attention_backend="triton", dtype="bfloat16",
                 device="cuda", tp_size=1, mem_fraction_static=0.7,
                 trust_remote_code=True, disable_radix_cache=True,
                 disable_cuda_graph=True, watchdog_timeout=1800)
    eng.generate(["warmup"], sampling_params={"temperature": 0.0, "max_new_tokens": 1})
    o = eng.generate([prompt], sampling_params={"temperature": 0.0, "max_new_tokens": 32})
    print("[triton-backend]", repr(o[0]["text"][:120]), flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
