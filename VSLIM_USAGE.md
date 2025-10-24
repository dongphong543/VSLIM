# VSLIM - Multi-Intent Slot Filling Model

This codebase is adapted from the VSLIM.ipynb notebook into a SLIM-style modular structure.

## Project Structure

```
.
├── main.py                    # Main entry point (follows SLIM format exactly)
├── trainer.py                 # Trainer, Trainer_multi, Trainer_woISeq classes
├── data_loader.py             # load_and_cache_examples, VSLIMDataset
├── utils.py                   # Utilities: set_seed, load_tokenizer, init_logger
├── requirements.txt           # Python dependencies
├── run_vslim_train.sh        # Example training script
├── vslim/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── slim.py           # VSLIM model
│   │   └── layers.py         # Classifier layers (MultiIntent, Slot, Biaffine, etc.)
│   ├── processors/
│   │   ├── __init__.py
│   │   └── label_loader.py   # Label loading and mapping functions
│   └── metrics/
│       ├── __init__.py
│       └── metrics.py        # Evaluation metrics
└── data/
    └── vped/
        ├── intent_label.txt
        ├── slot_label.txt
        ├── train/
        │   ├── seq_in.txt
        │   ├── label.txt
        │   ├── seq_out.txt
        │   └── seq_intent_out.txt
        ├── dev/
        └── test/
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

The model expects data in the following format:

1. **`intent_label.txt`**: One intent per line
   ```
   add_transaction
   update_transaction
   delete_transaction
   search_transaction
   stat_transaction
   ```

2. **`slot_label.txt`**: Slot labels in BIO format
   ```
   O
   B-target_date
   I-target_date
   B-target_price
   ...
   ```

3. **`seq_in.txt`**: Input tokens (space-separated)
4. **`label.txt`**: Intent labels (multi-intent separated by #)
5. **`seq_out.txt`**: Slot tags (BIO format, space-separated)
6. **`seq_intent_out.txt`**: Token-level intent tags

## Training

### Quick Start

```bash
bash run_vslim_train.sh
```

### Custom Training

```bash
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

## Key Arguments

### Model Arguments
- `--model_type`: Model type (`phobert` for PhoBERT)
- `--multi_intent`: Use multi-intent setting (1 for VSLIM)
- `--tag_intent`: Use biaffine tag-intent classifier
- `--cls_token_cat`: Concatenate CLS token
- `--intent_attn`: Use intent attention mechanism
- `--num_mask`: Number of slot entities per utterance (default: 4)
- `--use_crf`: Use CRF for sequence tagging

### Training Arguments
- `--train_batch_size`: Training batch size
- `--eval_batch_size`: Evaluation batch size
- `--learning_rate`: Learning rate
- `--num_train_epochs`: Number of training epochs
- `--max_seq_len`: Maximum sequence length
- `--dropout_rate`: Dropout rate

### Loss Coefficients
- `--slot_loss_coef`: Coefficient for slot loss (default: 2.0 from notebook)
- `--tag_intent_coef`: Coefficient for tag-intent loss (default: 1.0 from notebook)

## Model Architecture

VSLIM is a multi-intent joint model with:
1. **Multi-Intent Classification**: Sigmoid-based multi-label classification
2. **Slot Filling**: Token-level slot tagging
3. **Intent Token Classification**: Token-level intent prediction
4. **Tag-Intent Classification**: Biaffine attention for entity-intent mapping
5. **Intent Attention**: Attention mechanism combining utterance and entity intents

## Evaluation Metrics

- **Intent Accuracy**: Multi-intent exact match
- **Slot F1**: Sequence labeling F1 score
- **Semantic Frame Accuracy**: Complete frame (intent + slots) match
- **Intent-Slot Accuracy**: Intent and slot exact match

## Notes

- This code is adapted from VSLIM.ipynb research notebook
- Model follows SLIM repository structure
- Loss weights: W_SLOT=2.0, W_TOKINTENT=2.0, W_UTTINTENT=1.0, W_TAGINTENT=1.0 (from notebook)
- Default ignore_index=-100 for padding tokens

## Citation

If you use this code, please cite the original VSLIM paper.

