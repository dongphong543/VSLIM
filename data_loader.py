import os
import logging
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from seqeval.metrics.sequence_labeling import get_entities

logger = logging.getLogger(__name__)


def read_lines(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def resolve_file(base_dir: Path, name: str) -> Path:
    name_mappings = {
        'seq_in': ['seq.in', 'seq_in.txt'],
        'seq_out': ['seq.out', 'seq_out.txt'],
        'seq_intent_out': ['seq_intent.out', 'seq_intent_out.txt'],
        'label': ['label.txt', 'label']
    }
    
    # Get possible names
    possible_names = name_mappings.get(name, [name, f"{name}.txt"])
    
    # Try each possible name
    for candidate_name in possible_names:
        path = base_dir / candidate_name
        if path.exists():
            return path
    
    raise FileNotFoundError(f"Cannot find any of {possible_names} in {base_dir}")

def load_data_with_masks(data_dir: str, mappings: dict, num_mask: int = 4) -> Tuple:
    """
    Load data with B/BI masks for tag-intent 
    """
    base = Path(data_dir)

    if (base / "seq.in").exists():
        seq_in_path = base / "seq.in"
    else:
        seq_in_path = resolve_file(base, "seq_in")
    
    if (base / "seq.out").exists():
        seq_out_path = base / "seq.out"
    else:
        seq_out_path = resolve_file(base, "seq_out")
    
    if (base / "seq_intent.out").exists():
        seq_intent_path = base / "seq_intent.out"
    else:
        seq_intent_path = resolve_file(base, "seq_intent_out")
    
    label_path = resolve_file(base, "label")

    seq_in_lines = read_lines(seq_in_path)
    label_lines = read_lines(label_path)
    seq_out_lines = read_lines(seq_out_path)
    seq_intent_lines = read_lines(seq_intent_path)

    n = len(seq_in_lines)
    assert len(label_lines) == n
    assert len(seq_out_lines) == n
    assert len(seq_intent_lines) == n

    sentences = []
    slot_tags = []
    token_intent_tags = []
    utterance_intents = []
    B_tag_masks = []
    BI_tag_masks = []
    tag_intent_labels = []

    TOKINT2ID = mappings['TOKINT2ID']
    TAGINT2ID = mappings['TAGINT2ID']

    for i in range(n):
        tokens = seq_in_lines[i].strip().split()
        sentences.append(tokens)

        label_line = label_lines[i].strip()
        if label_line == "":
            utterance_intents.append(["none"])
        else:
            utterance_intents.append(label_line.split("#"))

        slot_line = seq_out_lines[i].strip()
        if slot_line == "":
            slot_tags.append(["O"] * len(tokens))
        else:
            raw_slots = slot_line.split()
            if len(raw_slots) < len(tokens):
                raw_slots = raw_slots + ["O"] * (len(tokens) - len(raw_slots))
            elif len(raw_slots) > len(tokens):
                raw_slots = raw_slots[:len(tokens)]
            slot_tags.append(raw_slots)

        tokint_line = seq_intent_lines[i].strip()
        if tokint_line == "":
            token_intent_tags.append(["O"] * len(tokens))
        else:
            raw_tokints = tokint_line.split()
            if len(raw_tokints) < len(tokens):
                raw_tokints = raw_tokints + ["O"] * (len(tokens) - len(raw_tokints))
            elif len(raw_tokints) > len(tokens):
                raw_tokints = raw_tokints[:len(tokens)]
            token_intent_tags.append(raw_tokints)

        # Generate B/BI masks and tag-intent labels 
        entities = get_entities(slot_line.split())
        if len(entities) > num_mask:
            entities = entities[:num_mask]

        B_tag_mask = [[0 for _ in range(len(tokens))] for _ in range(num_mask)]
        BI_tag_mask = [[0 for _ in range(len(tokens))] for _ in range(num_mask)]
        tag_intent_label = [TAGINT2ID["PAD"] for _ in range(num_mask)]

        try:
            for idx, (tag, start, end) in enumerate(entities):
                B_tag_mask[idx][start] = 1
                # Weighted BI mask
                weight = 1.0 / (end - start + 1)
                BI_tag_mask[idx][start:end+1] = [weight] * (end - start + 1)
                # Tag intent label from token intent at start position
                if start < len(token_intent_tags[-1]):
                    tokint_tag = token_intent_tags[-1][start]
                    if tokint_tag in TOKINT2ID and tokint_tag != "O":
                        tag_intent_label[idx] = TAGINT2ID[tokint_tag]
        except:
            pass  # Keep default PAD labels

        B_tag_masks.append(B_tag_mask)
        BI_tag_masks.append(BI_tag_mask)
        tag_intent_labels.append(tag_intent_label)

    logger.info(f"Loaded {n} examples from {data_dir}")

    return (sentences, slot_tags, token_intent_tags, utterance_intents,
            B_tag_masks, BI_tag_masks, tag_intent_labels)


class VSLIMDataset(Dataset):
    def __init__(self, sentences, slot_tags, token_intent_tags, utterance_intents,
                 B_masks, BI_masks, tag_intent_labels, tokenizer, mappings, max_len=128, num_mask=4, ignore_index=-100):
        self.sentences = sentences
        self.slot_tags = slot_tags
        self.token_intent_tags = token_intent_tags
        self.utterance_intents = utterance_intents
        self.B_masks = B_masks
        self.BI_masks = BI_masks
        self.tag_intent_labels = tag_intent_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_mask = num_mask
        self.ignore_index = ignore_index
        
        self.INTENT_LABELS = mappings['INTENT_LABELS']
        self.INTENT2ID = mappings['INTENT2ID']
        self.SLOT2ID = mappings['SLOT2ID']
        self.TOKINT2ID = mappings['TOKINT2ID']
        
        self.num_intents = len(self.INTENT_LABELS)
        self.num_tag_intents = len(mappings['INTENT_LABELS_WITH_PAD'])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        slot_labels = self.slot_tags[idx]
        tokint_labels = self.token_intent_tags[idx]
        utt_intents = self.utterance_intents[idx]
        B_mask = self.B_masks[idx]
        BI_mask = self.BI_masks[idx]
        tag_intent_label = self.tag_intent_labels[idx]

        is_none_utterance = (len(utt_intents) == 1 and utt_intents[0] == "none")

        # Subword tokenization with alignment
        subword_tokens = []
        subword_slot_labels = []
        subword_tokint_labels = []
        subword_B_masks = []
        subword_BI_masks = []

        for token, slot_tag, tokint_tag, B_pos_mask, BI_pos_mask in zip(
            tokens, slot_labels, tokint_labels, zip(*B_mask), zip(*BI_mask)):

            pieces = self.tokenizer.tokenize(token) or [self.tokenizer.unk_token]
            subword_tokens.extend(pieces)

            # Label assignment (first subword only)
            if is_none_utterance:
                subword_slot_labels.extend([self.ignore_index] * len(pieces))
                subword_tokint_labels.extend([self.ignore_index] * len(pieces))
            else:
                slot_id = self.SLOT2ID.get(slot_tag, self.SLOT2ID["O"])
                subword_slot_labels.append(slot_id)
                subword_slot_labels.extend([self.ignore_index] * (len(pieces) - 1))

                if tokint_tag == "O":
                    subword_tokint_labels.extend([self.ignore_index] * len(pieces))
                else:
                    tokint_id = self.TOKINT2ID.get(tokint_tag, self.TOKINT2ID["O"])
                    subword_tokint_labels.append(tokint_id)
                    subword_tokint_labels.extend([self.ignore_index] * (len(pieces) - 1))

            # B/BI masks (first subword only)
            subword_B_masks.append(B_pos_mask)
            subword_B_masks.extend([(0,) * self.num_mask] * (len(pieces) - 1))
            subword_BI_masks.append(BI_pos_mask)
            subword_BI_masks.extend([(0.0,) * self.num_mask] * (len(pieces) - 1))

        # Convert to input IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(subword_tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        # Add IGNORE_INDEX for special tokens
        subword_slot_labels = [self.ignore_index] + subword_slot_labels + [self.ignore_index]
        subword_tokint_labels = [self.ignore_index] + subword_tokint_labels + [self.ignore_index]
        subword_B_masks = [(0,) * self.num_mask] + subword_B_masks + [(0,) * self.num_mask]
        subword_BI_masks = [(0.0,) * self.num_mask] + subword_BI_masks + [(0.0,) * self.num_mask]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # Truncate if necessary
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            token_type_ids = token_type_ids[:self.max_len]
            subword_slot_labels = subword_slot_labels[:self.max_len]
            subword_tokint_labels = subword_tokint_labels[:self.max_len]
            subword_B_masks = subword_B_masks[:self.max_len]
            subword_BI_masks = subword_BI_masks[:self.max_len]

        # Pad to max_len
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            input_ids.extend([pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            token_type_ids.extend([0] * pad_len)
            subword_slot_labels.extend([self.ignore_index] * pad_len)
            subword_tokint_labels.extend([self.ignore_index] * pad_len)
            subword_B_masks.extend([(0,) * self.num_mask] * pad_len)
            subword_BI_masks.extend([(0.0,) * self.num_mask] * pad_len)

        utt_intent_vector = torch.zeros(self.num_intents, dtype=torch.float)
        for intent in utt_intents:
            if intent in self.INTENT2ID:
                utt_intent_vector[self.INTENT2ID[intent]] = 1.0

        # Convert B/BI masks to proper format
        B_mask_tensor = torch.tensor(list(zip(*subword_B_masks)), dtype=torch.long).T
        BI_mask_tensor = torch.tensor(list(zip(*subword_BI_masks)), dtype=torch.float).T
        tag_intent_tensor = torch.tensor(tag_intent_label, dtype=torch.long)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "slot_labels": torch.tensor(subword_slot_labels, dtype=torch.long),
            "tokint_labels": torch.tensor(subword_tokint_labels, dtype=torch.long),
            "uttint_labels": utt_intent_vector,
            "B_tag_mask": B_mask_tensor,
            "BI_tag_mask": BI_mask_tensor,
            "tag_intent_label": tag_intent_tensor
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "token_type_ids": torch.stack([item["token_type_ids"] for item in batch]),
        "slot_labels": torch.stack([item["slot_labels"] for item in batch]),
        "tokint_labels": torch.stack([item["tokint_labels"] for item in batch]),
        "uttint_labels": torch.stack([item["uttint_labels"] for item in batch]),
        "B_tag_mask": torch.stack([item["B_tag_mask"] for item in batch]).transpose(1, 2),
        "BI_tag_mask": torch.stack([item["BI_tag_mask"] for item in batch]).transpose(1, 2),
        "tag_intent_label": torch.stack([item["tag_intent_label"] for item in batch])
    }


def load_and_cache_examples(args, tokenizer, mode):
    """
    Load and cache examples for training/dev/test
    """
    from vslim.processors import get_label_mappings
    
    # Get label mappings
    _, _, mappings = get_label_mappings(args)
    
    # Determine data directory
    if mode == "train":
        data_dir = os.path.join(args.data_dir, args.task, "train")
    elif mode == "dev":
        data_dir = os.path.join(args.data_dir, args.task, "dev")
    elif mode == "test":
        data_dir = os.path.join(args.data_dir, args.task, "test")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Load data
    sentences, slot_tags, token_intent_tags, utterance_intents, B_masks, BI_masks, tag_intent_labels = \
        load_data_with_masks(data_dir, mappings, args.num_mask)
    
    # Create dataset
    dataset = VSLIMDataset(
        sentences, slot_tags, token_intent_tags, utterance_intents,
        B_masks, BI_masks, tag_intent_labels, tokenizer, mappings,
        max_len=args.max_seq_len,
        num_mask=args.num_mask,
        ignore_index=args.ignore_index
    )
    
    return dataset

