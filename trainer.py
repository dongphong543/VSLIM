import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from data_loader import collate_fn
from vslim.models import VSLIM
from vslim.metrics import compute_metrics_multi_intent_fixed
from vslim.processors import get_label_mappings

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        # Get label mappings
        _, _, self.mappings = get_label_mappings(args)
        
        self.intent_labels = self.mappings['INTENT_LABELS']
        self.slot_labels = self.mappings['SLOT_LABELS']
        self.intent2id = self.mappings['INTENT2ID']
        self.id2intent = self.mappings['ID2INTENT']
        self.slot2id = self.mappings['SLOT2ID']
        self.id2slot = self.mappings['ID2SLOT']

        # Model
        self.model = VSLIM(
            model_name=args.model_name_or_path,
            num_slots=len(self.slot_labels),
            num_intents=len(self.intent_labels),
            num_token_intents=len(self.mappings['TOKEN_INTENT_LABELS']),
            num_tag_intents=len(self.mappings['INTENT_LABELS_WITH_PAD']),
            dropout=args.dropout_rate,
            use_crf=args.use_crf,
            num_mask=args.num_mask,
            cls_token_cat=(args.cls_token_cat == 1),
            intent_attn=(args.intent_attn == 1),
            use_biaffine_tag_intent=(args.tag_intent == 1),
            args=args
        )
        
        # GPU or CPU
        # self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        if torch.cuda.is_available() and not args.no_cuda:
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"  # Mac Metal Performance Shaders
        else:
            self.device = "cpu"
        self.model.to(self.device)

    def train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, 
                                     shuffle=True, collate_fn=collate_fn)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Optimizer with no_decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        for epoch in range(int(self.args.num_train_epochs)):
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(train_dataloader):
                self.model.train()
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    intent_label_ids=batch["uttint_labels"],
                    slot_labels_ids=batch["slot_labels"],
                    intent_token_ids=batch["tokint_labels"],
                    B_tag_mask=batch["B_tag_mask"],
                    BI_tag_mask=batch["BI_tag_mask"],
                    tag_intent_label=batch["tag_intent_label"]
                )
                
                loss = outputs["total_loss"]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                epoch_loss += loss.item()
                epoch_steps += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        logger.info("  Step %d, Loss: %.4f", global_step, tr_loss / global_step)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    break

            logger.info("  Epoch %d finished", epoch + 1)
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            logger.info("=" * 60)
            logger.info("EPOCH %d RESULTS:", epoch + 1)
            logger.info("  Average Loss: %.4f", avg_epoch_loss)
            logger.info("  Steps: %d", epoch_steps)
            
            # Save model after each epoch
            self.save_model()
            logger.info("  Model saved after epoch %d", epoch + 1)
            
            # Evaluate on dev set after each epoch
            if self.dev_dataset is not None:
                logger.info("  Evaluating on dev set...")
                dev_results = self.evaluate("dev")
                logger.info("  Dev Results:")
                for key, value in dev_results.items():
                    logger.info("    %s: %.4f", key, value)
            
            logger.info("=" * 60)

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size, collate_fn=collate_fn)

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        self.model.eval()

        intent_preds = []
        slot_preds = []
        out_intent_label_ids = []
        out_slot_labels_ids = []

        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"]
                )

            # Intent prediction
            intent_logits = outputs["intent_logits"]
            intent_preds_batch = (intent_logits > 0.5).int()
            intent_preds.append(intent_preds_batch.cpu())
            out_intent_label_ids.append(batch["uttint_labels"].cpu())

            # Slot prediction
            slot_logits = outputs["slot_logits"]
            slot_preds_ids = torch.argmax(slot_logits, dim=2)
            slot_preds.append(slot_preds_ids.cpu())
            out_slot_labels_ids.append(batch["slot_labels"].cpu())

        # Concatenate all batches
        intent_preds = torch.cat(intent_preds, dim=0)
        out_intent_label_ids = torch.cat(out_intent_label_ids, dim=0)
        slot_preds = torch.cat(slot_preds, dim=0)
        out_slot_labels_ids = torch.cat(out_slot_labels_ids, dim=0)

        # Convert to lists
        slot_preds_list = []
        out_slot_label_list = []
        for i in range(slot_preds.size(0)):
            pred_seq = []
            label_seq = []
            for j in range(slot_preds.size(1)):
                if out_slot_labels_ids[i][j] != self.args.ignore_index:
                    pred_seq.append(self.id2slot[slot_preds[i][j].item()])
                    label_seq.append(self.id2slot[out_slot_labels_ids[i][j].item()])
            slot_preds_list.append(pred_seq)
            out_slot_label_list.append(label_seq)

        # Compute metrics
        total_result = compute_metrics_multi_intent_fixed(
            intent_preds,
            out_intent_label_ids,
            slot_preds_list,
            out_slot_label_list
        )

        logger.info("***** Eval results *****")
        for key in sorted(total_result.keys()):
            logger.info("  %s = %s", key, str(total_result[key]))

        return total_result

    def save_model(self):
        # Save model checkpoint
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), os.path.join(self.args.model_dir, 'model.pt'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Load a trained model
        if not os.path.exists(os.path.join(self.args.model_dir, 'model.pt')):
            raise Exception("Model doesn't exists! Train first!")

        self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, 'model.pt'), map_location=self.device))
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")


# class Trainer_multi(Trainer):
#     """
#     Trainer for multi-intent model (VSLIM)
#     """
#     pass


# class Trainer_woISeq(Trainer):
#     """
#     Trainer without intent sequence
#     """
#     pass

