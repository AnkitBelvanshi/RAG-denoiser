import re
import string
from collections import Counter
from typing import List, Tuple


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def lower(text):
        return text.lower()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_em_f1(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    em = max(exact_match_score(prediction, gt) for gt in ground_truths)
    f1 = max(f1_score(prediction, gt) for gt in ground_truths)
    return float(em), float(f1)


def _answer_in_text(answer: str, text: str) -> bool:
    """
    Light normalization match:
    - normalize_answer removes punctuation/articles + collapses whitespace
    - substring check in normalized space
    Fallback to lowercase raw substring if normalization becomes empty.
    """
    a = normalize_answer(answer)
    t = normalize_answer(text)

    if a:
        return a in t
    # fallback for rare cases where normalized answer becomes empty
    answer_raw = answer.strip().lower()
    return bool(answer_raw) and (answer_raw in text.lower())


def answer_span_hit(retrieved_texts: List[str], gold_answers: List[str]) -> int:
    """
    Returns 1 if ANY gold answer appears in ANY retrieved chunk (after light normalization).
    """
    for ans in gold_answers:
        for txt in retrieved_texts:
            if _answer_in_text(ans, txt):
                return 1
    return 0