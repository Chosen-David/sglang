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
    for filename in sorted(all_files):
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths, budgets = [], [], [], []
        dataset = filename.split("-")[0]
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
                if "budget" in data:
                    budgets.append(data["budget"])
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
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
