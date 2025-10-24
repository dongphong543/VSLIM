import logging
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

MODEL_CLASSES = {
    'phobert': (AutoTokenizer,)
}

MODEL_PATH_MAP = {
    'phobert': 'vinai/phobert-base-v2'
}

def get_intent_labels(args):
    return [line.strip() for line in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]

def get_slot_labels(args):
    return [line.strip() for line in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][0].from_pretrained(args.model_name_or_path)

def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]

def init_logger(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

