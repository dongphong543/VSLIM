import os
from typing import List, Dict, Tuple


def load_labels_from_file(file_path: str) -> List[str]:
    """
    Load labels from text file (one label per line)
    """
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                labels.append(line)
    return labels


def get_label_mappings(args):
    """
    Load all label mappings from files
    Returns: tuple of (INTENT_LABELS, SLOT_LABELS, mappings_dict)
    """
    intent_label_path = os.path.join(args.data_dir, args.task, args.intent_label_file)
    slot_label_path = os.path.join(args.data_dir, args.task, args.slot_label_file)
    
    INTENT_LABELS = load_labels_from_file(intent_label_path)
    SLOT_LABELS = load_labels_from_file(slot_label_path)
    
    # Generate token-intent labels (O + intents)
    TOKEN_INTENT_LABELS = ["O"] + INTENT_LABELS
    
    # Add PAD for tag-intent
    INTENT_LABELS_WITH_PAD = ["PAD"] + INTENT_LABELS
    
    # Create mappings
    INTENT2ID = {intent: i for i, intent in enumerate(INTENT_LABELS)}
    ID2INTENT = {i: intent for intent, i in INTENT2ID.items()}
    
    SLOT2ID = {slot: i for i, slot in enumerate(SLOT_LABELS)}
    ID2SLOT = {i: slot for slot, i in SLOT2ID.items()}
    
    TOKINT2ID = {tokint: i for i, tokint in enumerate(TOKEN_INTENT_LABELS)}
    ID2TOKINT = {i: tokint for tokint, i in TOKINT2ID.items()}
    
    TAGINT2ID = {intent: i for i, intent in enumerate(INTENT_LABELS_WITH_PAD)}
    ID2TAGINT = {i: intent for intent, i in TAGINT2ID.items()}
    
    mappings = {
        'INTENT_LABELS': INTENT_LABELS,
        'SLOT_LABELS': SLOT_LABELS,
        'TOKEN_INTENT_LABELS': TOKEN_INTENT_LABELS,
        'INTENT_LABELS_WITH_PAD': INTENT_LABELS_WITH_PAD,
        'INTENT2ID': INTENT2ID,
        'ID2INTENT': ID2INTENT,
        'SLOT2ID': SLOT2ID,
        'ID2SLOT': ID2SLOT,
        'TOKINT2ID': TOKINT2ID,
        'ID2TOKINT': ID2TOKINT,
        'TAGINT2ID': TAGINT2ID,
        'ID2TAGINT': ID2TAGINT
    }
    
    return INTENT_LABELS, SLOT_LABELS, mappings

