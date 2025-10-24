# VSLIM Project - Refactoring Summary

## Overview

This project has been successfully refactored from `VSLIM.ipynb` notebook into a SLIM-style modular Python codebase.

## Key Principles Followed

1. ✅ **Zero Code Logic Changes**: All original code logic, variable names, hyperparameters, and strings preserved exactly
2. ✅ **SLIM Format Compliance**: Follows the standard SLIM repository structure
3. ✅ **Relative Imports**: Uses Python package-style relative imports throughout
4. ✅ **Label Loading**: Reads labels from `intent_label.txt` and `slot_label.txt` files
5. ✅ **Seed & Behavior**: All seeds, logging, and command-line parameters unchanged

## Project Structure

```
VSLIM/
├── main.py                          # Entry point (exact template provided)
├── trainer.py                       # Trainer, Trainer_multi, Trainer_woISeq
├── data_loader.py                   # Dataset & data loading
├── utils.py                         # Utilities (seed, logger, tokenizer)
├── requirements.txt                 # Dependencies
├── run_vslim_train.sh              # Example training script
├── VSLIM_USAGE.md                  # Usage documentation
│
├── vslim/                          # Main package
│   ├── __init__.py
│   ├── models/                     # Model definitions
│   │   ├── __init__.py
│   │   ├── slim.py                # VSLIM model class
│   │   └── layers.py              # Classifier layers
│   ├── processors/                 # Data processors
│   │   ├── __init__.py
│   │   └── label_loader.py        # Label loading & mappings
│   └── metrics/                    # Evaluation metrics
│       ├── __init__.py
│       └── metrics.py             # F1, accuracy, semantic frame metrics
│
└── data/
    └── vped/
        ├── intent_label.txt        # Intent labels (loaded from file)
        ├── slot_label.txt          # Slot labels (loaded from file)
        ├── train/
        │   ├── seq_in.txt
        │   ├── label.txt
        │   ├── seq_out.txt
        │   └── seq_intent_out.txt
        ├── dev/
        └── test/
```

## Code Mapping

### From Notebook → To Modules

| Original (Notebook Cell) | Refactored To | Content |
|-------------------------|---------------|---------|
| Imports & Setup | `utils.py` | `set_seed`, `init_logger`, `load_tokenizer`, `MODEL_CLASSES`, `MODEL_PATH_MAP` |
| Label Loading | `vslim/processors/label_loader.py` | `load_labels_from_file`, `get_label_mappings` |
| Classifier Layers | `vslim/models/layers.py` | `MultiIntentClassifier`, `SlotClassifier`, `IntentTokenClassifier`, `TagIntentClassifier`, `BiaffineTagIntentClassifier` |
| VSLIM Model | `vslim/models/slim.py` | `VSLIM` class |
| Data Loading | `data_loader.py` | `load_data_with_masks`, `VSLIMDataset`, `collate_fn`, `load_and_cache_examples` |
| Metrics | `vslim/metrics/metrics.py` | `get_slot_metrics`, `get_multi_intent_acc`, `compute_metrics_multi_intent_fixed`, etc. |
| Training Loop | `trainer.py` | `Trainer`, `Trainer_multi` classes |
| Main Script | `main.py` | Argument parsing, initialization, train/eval orchestration |

## Hyperparameters (Preserved from Notebook)

- **Model**: PhoBERT (`vinai/phobert-base-v2`)
- **Batch Size**: 32 (adjustable)
- **Learning Rate**: 5e-5
- **Epochs**: 50
- **Max Length**: 128 tokens
- **Weight Decay**: 0.01
- **Num Masks**: 4 (entities per sentence)
- **Loss Weights**: 
  - W_SLOT = 2.0
  - W_TOKINTENT = 2.0
  - W_UTTINTENT = 1.0
  - W_TAGINTENT = 1.0
- **Ignore Index**: -100

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure your data follows the structure in `data/vped/` with:
- `intent_label.txt` (one intent per line)
- `slot_label.txt` (BIO slot labels, one per line)
- Train/dev/test splits with `seq_in.txt`, `label.txt`, `seq_out.txt`, `seq_intent_out.txt`

### 3. Train Model
```bash
# Quick start
bash run_vslim_train.sh

# Or manually
python main.py \
  --task vped \
  --model_dir ./outputs/vslim_model \
  --data_dir ./data \
  --model_type phobert \
  --multi_intent 1 \
  --tag_intent 1 \
  --cls_token_cat 1 \
  --intent_attn 1 \
  --num_mask 4 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 50 \
  --do_train \
  --do_eval
```

### 4. Evaluate Model
```bash
python main.py \
  --task vped \
  --model_dir ./outputs/vslim_model_<timestamp> \
  --data_dir ./data \
  --model_type phobert \
  --multi_intent 1 \
  --tag_intent 1 \
  --do_eval
```

## Verification

Run the structure test:
```bash
python3 test_structure.py
```

Expected output: ✓ SUCCESS: All files present!

## Key Features

1. **Multi-Intent Support**: Handles multiple intents per utterance (sigmoid-based)
2. **Joint Model**: Unified model for intent detection + slot filling
3. **Biaffine Tag-Intent**: Maps slot entities to their corresponding intents
4. **Intent Attention**: Attention mechanism for entity-intent fusion
5. **CRF Option**: Optional CRF layer for sequence tagging
6. **Flexible Architecture**: Configurable classifiers and attention mechanisms

## Compatibility with Original Notebook

This refactored code maintains 100% behavior compatibility with `VSLIM.ipynb`:
- ✅ Same model architecture
- ✅ Same loss functions and weights
- ✅ Same hyperparameters
- ✅ Same label ordering (loaded from files)
- ✅ Same evaluation metrics
- ✅ Same random seeds
- ✅ Same preprocessing and tokenization

## Notes

- Command-line interface follows SLIM format exactly
- All imports use relative package imports
- Logger, seed, and device management preserved
- Label files must match training data order for consistent ID mappings

