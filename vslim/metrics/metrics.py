import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score as seq_f1

def getEntities(elems):
    """
    Compress the entities
    For example: ['O', 'Good', 'Good', 'O', 'O']
                 will be compressed as [('O', 0, 1), ('Good', 1, 3), ('O', 3, 5)]
    """
    if isinstance(elems[0], list):
        elems = [item for sub in elems for item in sub + ['O']]

    current_char = elems[0]
    elem_len = len(elems)
    current_idx = 0
    ptr = current_idx + 1
    entities = []

    while ptr < elem_len:
        if ptr == elem_len - 1:
            if elems[ptr] == current_char:
                entities.append((current_char, current_idx, ptr + 1))
            else:
                entities.append((current_char, current_idx, ptr))
                entities.append((elems[ptr], ptr, ptr+1))
            break

        if elems[ptr] != current_char:
            entities.append((current_char, current_idx, ptr))
            current_idx = ptr
            current_char = elems[ptr]

        ptr += 1

    return entities


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": seq_f1(labels, preds)
    }

def get_multi_intent_acc(intent_preds, intent_labels):
    intent_preds = intent_preds.tolist() if torch.is_tensor(intent_preds) else intent_preds.tolist()
    intent_labels = intent_labels.tolist() if torch.is_tensor(intent_labels) else intent_labels.tolist()
    records = []
    for ip, il in zip(intent_preds, intent_labels):
        one_sent = True
        for ipn, iln in zip(ip, il):
            if ipn != iln:
                one_sent = False
                break
        records.append(int(one_sent))

    return {
        "intent_acc": np.mean(records)
    }

def get_intent_token_metrics(intent_token_preds, intent_tokens):
    pred_tokens = set([item for item in getEntities(intent_token_preds) if item[0] != 'O'])
    true_tokens = set([item for item in getEntities(intent_tokens) if item[0] != 'O'])

    nb_correct = len(pred_tokens & true_tokens)
    nb_pred = len(pred_tokens)
    nb_true = len(true_tokens)

    pre = nb_correct / nb_pred if nb_pred > 0 else 0
    recall = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * pre * recall / (pre + recall) if pre + recall > 0 else 0

    return {
        'intent_token_precision': pre,
        'intent_token_recall': recall,
        'intent_token_f1': score,
    }

def get_tag_intent_metrics_fixed(tag_intent_preds, tag_intent_ids):
    """
    So sánh tag-intent CHỈ Ở VỊ TRÍ NON-PAD trong ground truth
    """
    total_cnt = 0
    correct_cnt = 0

    for pred_list, gt_list in zip(tag_intent_preds, tag_intent_ids):
        for pred, gt in zip(pred_list, gt_list):
            # CHỈ tính những vị trí có ground truth label (không phải PAD)
            if gt != "PAD":
                if pred == gt:
                    correct_cnt += 1
                total_cnt += 1

    return {
        'tag_intent_acc': correct_cnt / total_cnt if total_cnt > 0 else 0
    }

def get_semantic_basic_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """
    Semantic Basic Accuracy - CHỈ so sánh intent + slot
    (Không tính token-intent và tag-intent)
    """
    intent_preds = intent_preds.tolist() if torch.is_tensor(intent_preds) else intent_preds.tolist()
    intent_labels = intent_labels.tolist() if torch.is_tensor(intent_labels) else intent_labels.tolist()

    correct = 0
    total = len(intent_labels)

    for intent_pred, intent_label, slot_pred, slot_label in zip(
        intent_preds, intent_labels, slot_preds, slot_labels
    ):
        # Intent match (multi-hot comparison)
        intent_match = all(ip == il for ip, il in zip(intent_pred, intent_label))

        # Slot match (sequence comparison)
        slot_match = (slot_pred == slot_label)

        if intent_match and slot_match:
            correct += 1

    return {
        "semantic_basic_acc": correct / total if total > 0 else 0
    }

def get_sentence_frame_acc_multi_intent_fixed(intent_preds,
                                        intent_labels,
                                        slot_preds,
                                        slot_labels,
                                        intent_token_preds=None,
                                        intent_tokens=None,
                                        tag_intent_preds=None,
                                        tag_intent_labels=None):
    """
    FIX: CHỈ so sánh tag-intent ở vị trí NON-PAD
    """
    intent_token_existence = (intent_token_preds is not None and intent_tokens is not None)
    tag_intent_existence = (tag_intent_preds is not None and tag_intent_labels is not None)

    # Intent comparison
    intent_result = []
    intent_preds = intent_preds.tolist() if torch.is_tensor(intent_preds) else intent_preds.tolist()
    intent_labels = intent_labels.tolist() if torch.is_tensor(intent_labels) else intent_labels.tolist()

    for ip, il in zip(intent_preds, intent_labels):
        one_sent = True
        for ipn, iln in zip(ip, il):
            if ipn != iln:
                one_sent = False
                break
        intent_result.append(int(one_sent))
    intent_result = np.array(intent_result)

    # Slot comparison
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)

    slot_result = np.array(slot_result)

    # Intent token comparison
    if intent_token_existence:
        intent_token_result = []
        for preds, labels in zip(intent_token_preds, intent_tokens):
            assert len(preds) == len(labels)
            one_sent_result = True
            for p, l in zip(preds, labels):
                if p != l:
                    one_sent_result = False
                    break
            intent_token_result.append(one_sent_result)

        intent_token_result = np.array(intent_token_result)

    # Tag-intent comparison - FIX: CHỈ so sánh NON-PAD
    if tag_intent_existence:
        tag_intent_result = []
        for preds, labels in zip(tag_intent_preds, tag_intent_labels):
            one_sent_result = True
            for p, l in zip(preds, labels):
                # CHỈ so sánh khi ground truth không phải PAD
                if l != "PAD":
                    if p != l:
                        one_sent_result = False
                        break
            tag_intent_result.append(one_sent_result)

        tag_intent_result = np.array(tag_intent_result)

    # Combine all results
    if tag_intent_existence and intent_token_existence:
        semantic_acc = np.multiply(np.multiply(np.multiply(intent_result, slot_result), intent_token_result), tag_intent_result).mean()
    elif tag_intent_existence:
        semantic_acc = np.multiply(np.multiply(intent_result, slot_result), tag_intent_result).mean()
    elif intent_token_existence:
        semantic_acc = np.multiply(np.multiply(intent_result, slot_result), intent_token_result).mean()
    else:
        semantic_acc = np.multiply(intent_result, slot_result).mean()

    intent_slot_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "semantic_exact_acc": semantic_acc,
        "intent_slot_acc": intent_slot_acc,
    }

def compute_metrics_multi_intent_fixed(intent_preds,
                                 intent_labels,
                                 slot_preds,
                                 slot_labels,
                                 intent_token_preds=None,
                                 intent_tokens=None,
                                 tag_intent_preds=None,
                                 tag_intent_ids=None):
    intent_seq_existence = (intent_token_preds is not None and intent_tokens is not None)
    tag_intent_existence = (tag_intent_preds is not None and tag_intent_ids is not None)

    results = {}
    intent_result = get_multi_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)

    if intent_seq_existence:
        intent_token_result = get_intent_token_metrics(intent_token_preds, intent_tokens)

    if tag_intent_existence:
        tag_intent_result = get_tag_intent_metrics_fixed(tag_intent_preds, tag_intent_ids)

    semantic_basic_result = get_semantic_basic_acc(
        intent_preds,
        intent_labels,
        slot_preds,
        slot_labels
    )

    semantic_exact_result = get_sentence_frame_acc_multi_intent_fixed(
        intent_preds,
        intent_labels,
        slot_preds,
        slot_labels,
        intent_token_preds,
        intent_tokens,
        tag_intent_preds,
        tag_intent_ids)

    results.update(intent_result)
    results.update(slot_result)
    if intent_seq_existence:
        results.update(intent_token_result)
    if tag_intent_existence:
        results.update(tag_intent_result)

    results.update(semantic_basic_result) 

    results.update(semantic_exact_result)

    return results

