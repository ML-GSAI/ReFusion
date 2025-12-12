from itertools import zip_longest
import transformers.data.metrics.squad_metrics as squad_metrics
import re, string
import math
from latex2sympy2 import latex2sympy
from math import sqrt, sin, cos, log, pi, factorial, exp, e
import os
import numpy as np
import json
import argparse
from collections import Counter

def answer_clean(direct_answer_trigger_for_fewshot: tuple, pred: str):
    pred = pred.strip('\n')

    # Split the trigger to find the answer.
    preds = re.split('|'.join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ')

    return pred

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
  
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def f1_score(prediction, ground_truth):
    ZERO_METRIC = (0, 0, 0)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def get_metrics(pred, truth):
    pred = normalize_answer(pred)
    gt = normalize_answer(truth)
    em = (pred == gt)
    f1, p, r = f1_score(pred, gt)
    return {'em': em, 'f1': f1} #, 'recall': r, 'precision': p}


def compute_scores(truth, pred):
    pred = answer_clean(['The answer is:', 'The answer is', 'the answer is', 'final answer is:', 'final answer is', 'Therefore, the final answer is'], pred)
    pred = pred.strip().split("\n")[0]
    metric = get_metrics(pred=pred, truth=truth)
    return metric


def process_results(doc, results):
    truth = doc["answer"]
    pred = results[0].strip()
    scores = compute_scores(truth, pred)
    return scores