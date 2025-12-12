from itertools import zip_longest

import transformers.data.metrics.squad_metrics as squad_metrics

def doc_to_target(doc):
    if doc["answer"]:
        return 'A'
    else:
        return 'B'
def doc_to_target_gen(doc):
    if doc["answer"]:
        return ['A', 'True']
    else:
        return ['B', 'False']


def compute_scores(gold_list, pred):
    em_sum = 0.0
    if len(gold_list) > 1:
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1 :]
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_answers)
    else:
        em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)

    return {
        "em": em_sum / max(1, len(gold_list)),
    }


def process_results(doc, results):
    gold_list = doc_to_target(doc)
    pred = results[0].strip().split("\n")[0]

    scores = compute_scores(gold_list, pred)
    return scores
