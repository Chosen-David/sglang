import os
import re
import json
import argparse
import numpy as np

from .metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
    SCORER_BACKEND_ID,
)

def lbv2_choice_score(prediction, ground_truth, all_classes=None):
    """LongBench-v2（lbv2）四选一 accuracy：按官方口径解析 pred 字母（'The correct
    answer is (X)' → 'The correct answer is X' → 首个独立 A/B/C/D 兜底，大小写兼容，
    无匹配记 0 分），与真值字母比对。

    注意：解析逻辑与 pred.py 的 extract_choice_letter 保持同步（两处不互相 import，
    因 metrics.py 顶层依赖 jieba 等打分库，GPU 推理机上未必安装）。
    """
    pred_choice = None
    if prediction:
        t = prediction.replace("*", "")
        m = re.search(r"The correct answer is \(([A-D])\)", t, flags=re.IGNORECASE)
        if m:
            pred_choice = m.group(1).upper()
        else:
            m = re.search(r"The correct answer is ([A-D])", t, flags=re.IGNORECASE)
            if m:
                pred_choice = m.group(1).upper()
            else:
                m = re.search(r"\b([ABCD])\b", t, flags=re.IGNORECASE)
                pred_choice = m.group(1).upper() if m else None
    gt = str(ground_truth).strip().upper()
    return 1.0 if (pred_choice is not None and pred_choice == gt) else 0.0


dataset2metric = {
    "lbv2": lbv2_choice_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench": code_sim_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--e", type=int, default=0, help="Evaluate on LongBench-E")
    parser.add_argument("--t", type=str, default="benchmark launch time")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--pred_postfix", type=str, default="")
    # E116a（GPT TL-LBV1-SAMPLE-GATE-024）fail-closed 完整性门禁：
    #   --manifest <json>：{task: {"ids": [...], "answers_sha": {id: sha}}} 预期清单，
    #     实际 _id 集合必须与预期完全相等（无缺行/重复/混行），answers hash 一致
    #   --expect-count <int>：每文件行数下限（不满足即非零退出）
    #   门禁失败 → SystemExit(2)，不写 result.json
    parser.add_argument("--manifest", type=str, default="",
                        help="样本身份 manifest（JSON），给定时做集合闭包校验")
    parser.add_argument("--expect-count", type=int, default=0,
                        help="每个 jsonl 的最小行数（>0 时启用）")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == "__main__":
    args = parse_args()
    scores = dict()
    if len(args.output_path) > 0:
        dir = f"{args.output_path}"
        path = f"{dir}/"
    elif args.e:
        dir = f"{args.output_dir}/pred_e{args.pred_postfix}"
        path = f"{dir}/"
    else:
        dir = f"{args.output_dir}/pred{args.pred_postfix}"
        path = f"{dir}/"

    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    # E116a：fail-closed 完整性门禁（GPT TL-LBV1-SAMPLE-GATE-024）
    manifest = json.load(open(args.manifest)) if args.manifest else None
    import hashlib as _hashlib

    def _answers_sha(ans):
        return _hashlib.sha256(
            json.dumps(ans, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    for filename in sorted(all_files):
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths, budgets = [], [], [], []
        ids = []
        dataset = filename.split("-")[0]
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "_id" in data:
                    ids.append(data["_id"])
                if "length" in data:
                    lengths.append(data["length"])
                if "budget" in data:
                    budgets.append(data["budget"])
        # ---- 门禁 1：行数下限（--expect-count）----
        if args.expect_count > 0 and len(predictions) < args.expect_count:
            raise SystemExit(
                f"[GATE-FAIL] {filename}: {len(predictions)} 行 < expect-count "
                f"{args.expect_count}——fail closed，不写 result.json"
            )
        # ---- 门禁 2：_id 无重复（新数据全带 _id；旧 v1 文件无 _id 时跳过）----
        if ids:
            if len(ids) != len(set(ids)):
                dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
                raise SystemExit(
                    f"[GATE-FAIL] {filename}: 检测到重复 _id（示例 {dup}）——"
                    f"fail closed，不写 result.json"
                )
            if len(ids) != len(predictions):
                raise SystemExit(
                    f"[GATE-FAIL] {filename}: 部分 行缺失 _id（{len(ids)}/"
                    f"{len(predictions)}）——fail closed，不写 result.json"
                )
        # ---- 门禁 3：manifest 集合闭包 + answers hash 一致 ----
        if manifest is not None:
            m_task = manifest.get(dataset)
            if m_task is None:
                print(f"[GATE-WARN] {filename}: 任务 {dataset} 不在 manifest——跳过该文件")
                continue
            exp_ids = set(m_task["ids"])
            got_ids = set(ids) if ids else None
            if got_ids is None:
                raise SystemExit(
                    f"[GATE-FAIL] {filename}: 预测缺 _id 字段，无法对 manifest 校验——"
                    f"fail closed，不写 result.json"
                )
            if got_ids != exp_ids:
                missing = sorted(exp_ids - got_ids)[:5]
                extra = sorted(got_ids - exp_ids)[:5]
                raise SystemExit(
                    f"[GATE-FAIL] {filename}: _id 集合与 manifest 不等"
                    f"（缺 {len(exp_ids-got_ids)} 例 {missing}；多 "
                    f"{len(got_ids-exp_ids)} 例 {extra}）——fail closed，不写 result.json"
                )
            exp_sha = m_task.get("answers_sha", {})
            for i, a in zip(ids, answers):
                if i in exp_sha and _answers_sha(a) != exp_sha[i]:
                    raise SystemExit(
                        f"[GATE-FAIL] {filename}: _id={i} 的 answers hash 与 manifest "
                        f"不一致（答案错配/混行）——fail closed，不写 result.json"
                    )
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        budgets = list(filter(lambda x: x is not None, budgets))
        budget = sum(budgets) / len(budgets) if len(budgets) > 0 else -1
        scores[filename] = {"score": score, "budget": budget}
    if args.e:
        out_path = f"{dir}/result.json"
    else:
        out_path = f"{dir}/result.json"
    # E116a：scorer 后端身份随 result.json 落盘（跨环境复现性锚点）
    scores["_meta"] = {"scorer_backend": SCORER_BACKEND_ID}
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
