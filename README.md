# VSLIM: Vietnamese Explicit Slot-Intent Mapping for Joint Multi-Intent Detection and Slot Filling

VSLIM (Vietnamese SLot-Intent Mapping) is a unified neural architecture designed for Vietnamese Natural Language Understanding (NLU) tasks, specifically focusing on joint multi-intent detection and slot filling. VSLIM explicitly models the relationships between slots and intents, making it the first comprehensive effort to develop explicit slot-intent modeling in Vietnamese dialogue systems that supports multiple intents.

## Features

- **Multi-Intent Classification**: Handles multiple intents per utterance using sigmoid activation
- **Slot Filling**: Sequence labeling task using BIO tagging scheme for Named Entity Recognition
- **Intent Token Classification**: Token-level intent prediction for fine-grained understanding
- **Tag-Intent Classification**: B/BI tag-based intent mapping with biaffine attention mechanism
- **Vietnamese Language Support**: Built on PhoBERT for optimal Vietnamese language understanding

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dongphong543/VSLIM
cd VSLIM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Requirements

- Python 3.7+
- PyTorch >= 2.1.0
- Transformers == 4.41.0
- Additional dependencies listed in `requirements.txt`

### Training and Evaluation

Run the following command to train and evaluate the VSLIM model on the VPED dataset:

```bash
python main.py \
  --task vped \
  --model_dir ./outputs/vslim_model \
  --data_dir ./data \
  --model_type phobert \
  --num_train_epochs 50 \
  --logging_steps 10 \
  --warmup_steps 160 \
  --save_steps 200 \
  --do_train \
  --do_eval
```

### Key Parameters

- `--task`: Dataset name (vped or phoatis)
- `--model_dir`: Directory to save/load model checkpoints
- `--data_dir`: Path to data directory
- `--model_type`: Backbone model (phobert)
- `--num_train_epochs`: Number of training epochs
- `--do_train`: Enable training mode
- `--do_eval`: Enable evaluation mode


### Configuration Options

#### Model Architecture
- `--intent_seq` (default: 1): Whether to use intent sequence setting
- `--tag_intent` (default: 1): Whether to use tag to predict intent (biaffine classifier)
- `--BI_tag` (default: 1): Use BI sum or just B tags
- `--cls_token_cat` (default: 1): Whether to concatenate CLS token to slot output
- `--intent_attn` (default: 1): Whether to use attention mechanism on CLS intent output
- `--num_mask` (default: 4): Assumptive number of slots in one sentence

#### Training Parameters
- `--train_batch_size` (default: 32): Batch size for training
- `--eval_batch_size` (default: 32): Batch size for evaluation
- `--max_seq_len` (default: 128): Maximum input sequence length
- `--learning_rate` (default: 5e-5): Initial learning rate for Adam optimizer
- `--num_train_epochs` (default: 10.0): Total number of training epochs
- `--weight_decay` (default: 0.01): Weight decay for regularization
- `--gradient_accumulation_steps` (default: 1): Steps to accumulate before backward pass
- `--warmup_steps` (default: 0): Linear warmup steps (recommended: 10% of total steps)
- `--dropout_rate` (default: 0.1): Dropout rate for fully-connected layers
- `--max_grad_norm` (default: 1.0): Maximum gradient norm for clipping
- `--patience` (default: 0): Early stopping patience

#### Loss Coefficients
- `--intent_loss_coef` (default: 1.0): Weight for intent classification loss
- `--slot_loss_coef` (default: 2.0): Weight for slot filling loss
- `--token_intent_loss_coef` (default: 2.0): Weight for token-level intent loss
- `--tag_intent_coef` (default: 1.0): Weight for tag-intent mapping loss

#### Logging and Saving
- `--logging_steps` (default: 500): Log every X update steps
- `--save_steps` (default: 200): Save checkpoint every X update steps
- `--seed` (default: 36): Random seed for reproducibility

#### Device Options
- `--no_cuda`: Avoid using CUDA when available (force CPU usage)

#### Data Configuration
- `--intent_label_file` (default: "intent_label.txt"): Intent label file name
- `--slot_label_file` (default: "slot_label.txt"): Slot label file name
- `--slot_pad_label` (default: "PAD"): Padding token for slot labels

## Acknowledgement

Our code is based on the implementation of the SLIM paper from
https://github.com/TRUMANCFY/SLIM

## Citation

If you use this code in your research, please cite the original VSLIM paper.

## Authors

- Phong Chung, Kha Le-Minh, Xuan-Bach Le, and Tho Quan
