import re
import string
import os as _os

import jieba
import difflib

from typing import List
from collections import Counter
from rouge import Rouge

# ---- E116a（GPT TL-LBV1-SCORER-BACKEND-025）：代码相似度后端显式固定 ----
# 历史 bug：fuzzywuzzy.fuzz.ratio 在装了 python-Levenshtein 时走 StringMatcher
# （indel ratio），没装时走 difflib.SequenceMatcher——同一份预测两任务漂 ~1.5pt，
# 13 任务宏平均漂 ~0.234。此处显式固定单一后端，禁止可选依赖静默改变语义：
#   默认 difflib：纯标准库，零依赖，跨环境逐位可复现（canonical 口径）
#   TLI_SCORER_BACKEND=levenshtein：显式要求 Levenshtein，缺包即启动失败（fail closed）
# 语义与 fuzzywuzzy 两条历史路径逐一对应（均做 int(round(100*ratio))）。
_SCORER_BACKEND = _os.environ.get("TLI_SCORER_BACKEND", "difflib").strip().lower()
if _SCORER_BACKEND not in ("difflib", "levenshtein"):
    raise ImportError(
        f"TLI_SCORER_BACKEND={_SCORER_BACKEND!r} 非法（合法值: difflib | levenshtein）"
    )
if _SCORER_BACKEND == "levenshtein":
    try:
        import Levenshtein as _Lev
    except ImportError as _e:
        raise ImportError(
            "TLI_SCORER_BACKEND=levenshtein 但 python-Levenshtein 未安装——"
            "拒绝静默回退 difflib（fail closed），请安装或改用默认 difflib"
        ) from _e
    def _fuzz_ratio(s1: str, s2: str) -> int:
        return int(round(100 * _Lev.ratio(s1, s2)))
    SCORER_BACKEND_ID = "levenshtein:" + getattr(_Lev, "__version__", "?")
else:
    def _fuzz_ratio(s1: str, s2: str) -> int:
        # 等价 fuzzywuzzy 无 Levenshtein 时的 stdlib 路径
        return int(round(100 * difflib.SequenceMatcher(None, s1, s2).ratio()))
    SCORER_BACKEND_ID = "difflib:stdlib"

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (_fuzz_ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score
    
def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)
