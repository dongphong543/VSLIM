import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from seqeval.metrics.sequence_labeling import get_entities
from data_loader import collate_fn, load_data_with_masks
from vslim.models import VSLIM
from vslim.metrics import compute_metrics_multi_intent_fixed, get_semantic_basic_acc
from vslim.processors import get_label_mappings
from predict import VSLIMPredictor

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
        self.tokint2id = self.mappings['TOKINT2ID']
        self.id2tokint = self.mappings['ID2TOKINT']
        self.tagint2id = self.mappings['TAGINT2ID']
        self.id2tagint = self.mappings['ID2TAGINT']

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
        if torch.cuda.is_available() and not args.no_cuda:
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        
        # Tokenizer (needed for inference)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # Create predictor
        self.predictor = VSLIMPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            mappings=self.mappings,
            device=self.device,
            args=self.args
        )

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
                    if isinstance(value, (int, float)):
                        logger.info("    %s: %.4f", key, value)
            
            logger.info("=" * 60)

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        """2-pass evaluation matching notebook exactly"""
        if mode == 'test':
            dataset = self.test_dataset
            data_dir = os.path.join(self.args.data_dir, self.args.task, "test")
        elif mode == 'dev':
            dataset = self.dev_dataset
            data_dir = os.path.join(self.args.data_dir, self.args.task, "dev")
        else:
            raise Exception("Only dev and test dataset available")

        # Load raw data for 2-pass evaluation
        dataset_sentences, dataset_slots, dataset_tokints, dataset_intents, _, _, dataset_tag_intents = \
            load_data_with_masks(data_dir, self.mappings, self.args.num_mask)

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset_sentences))

        # ========== PASS 1: Predict intents and slots ==========
        intent_preds = []
        slot_preds = []
        intent_token_preds = []
        out_intent_label_ids = []
        out_slot_labels_ids = []
        out_intent_token_ids = []

        self.model.eval()

        for tokens, slot_tags, tokint_tags, utt_intents in zip(
            dataset_sentences, dataset_slots, dataset_tokints, dataset_intents):

            with torch.no_grad():
                result = self.predictor.predict_single(tokens, threshold=0.5)

                # Intent predictions (multi-hot)
                intent_multihot = [0] * len(self.intent_labels)
                for intent in result["utterance_intents"]:
                    if intent in self.intent2id:
                        intent_multihot[self.intent2id[intent]] = 1
                intent_preds.append(intent_multihot)

                # Gold intent (multi-hot)
                gold_multihot = [0] * len(self.intent_labels)
                for intent in utt_intents:
                    if intent in self.intent2id:
                        gold_multihot[self.intent2id[intent]] = 1
                out_intent_label_ids.append(gold_multihot)

                # Slot predictions
                slot_preds.append(result["slot_tags"])
                out_slot_labels_ids.append(slot_tags)

                # Token-intent predictions
                intent_token_preds.append(result["token_intents"])
                out_intent_token_ids.append(tokint_tags)

        # Convert to proper format
        intent_preds = torch.as_tensor(intent_preds, dtype=torch.int32)
        out_intent_label_ids = np.array(out_intent_label_ids)

        # Generate predicted masks
        slot_preds_list = slot_preds
        out_slot_label_list = out_slot_labels_ids

        B_tag_mask_pred_tensor, BI_tag_mask_pred_tensor = self.predictor.generate_predicted_masks_from_slots(
            slot_preds_list, max_seq_len=self.args.max_seq_len)

        # ========== PASS 2: Predict tag-intents with predicted masks ==========
        tag_intent_preds = []
        out_tag_intent_ids = []

        self.model.eval()

        for idx, tokens in enumerate(dataset_sentences):
            with torch.no_grad():
                # Prepare input
                input_ids, attention_mask, token_type_ids, word_positions = \
                    self.predictor.align_tokens_for_inference(tokens)

                batch_size, seq_len = input_ids.shape

                # Get predicted masks for this sample
                B_mask = B_tag_mask_pred_tensor[idx].numpy().tolist()
                BI_mask = BI_tag_mask_pred_tensor[idx].numpy().tolist()

                # Align to subwords
                B_mask_tensor = self.predictor.align_masks_to_subwords(B_mask, word_positions, seq_len)
                B_mask_tensor = B_mask_tensor.unsqueeze(0).to(self.device).long()

                BI_mask_tensor = self.predictor.align_masks_to_subwords(BI_mask, word_positions, seq_len)
                BI_mask_tensor = BI_mask_tensor.unsqueeze(0).to(self.device)

                tag_intent_label = torch.full((1, self.args.num_mask), self.args.ignore_index, 
                                             dtype=torch.long, device=self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    intent_label_ids=None,
                    slot_labels_ids=None,
                    intent_token_ids=None,
                    B_tag_mask=B_mask_tensor,
                    BI_tag_mask=BI_mask_tensor,
                    tag_intent_label=tag_intent_label
                )

                # Extract tag-intent predictions
                if outputs["tag_intent_logits"] is not None:
                    tag_intent_logits = outputs["tag_intent_logits"].view(self.args.num_mask, -1)
                    tag_intent_pred = torch.argmax(tag_intent_logits, dim=-1).cpu().numpy()

                    # Convert to labels
                    pred_labels = []
                    for pred_id in tag_intent_pred:
                        pred_label = self.id2tagint.get(pred_id, "PAD")
                        pred_labels.append(pred_label)

                    tag_intent_preds.append(pred_labels)
                else:
                    tag_intent_preds.append(["PAD"] * self.args.num_mask)

                # Gold tag intents
                gold_labels = []
                for label_id in dataset_tag_intents[idx]:
                    gold_label = self.id2tagint.get(label_id, "PAD")
                    gold_labels.append(gold_label)
                out_tag_intent_ids.append(gold_labels)

        # ========== Compute metrics ==========
        out_intent_token_list = out_intent_token_ids
        intent_token_preds_list = intent_token_preds
        out_tag_intent_list = out_tag_intent_ids
        tag_intent_preds_list = tag_intent_preds

        # Compute all metrics
        total_result = compute_metrics_multi_intent_fixed(
            intent_preds,
            out_intent_label_ids,
            slot_preds_list,
            out_slot_label_list,
            intent_token_preds_list,
            out_intent_token_list,
            tag_intent_preds_list,
            out_tag_intent_list
        )

        # Compute semantic basic accuracy
        semantic_basic_result = get_semantic_basic_acc(
            intent_preds,
            out_intent_label_ids,
            slot_preds_list,
            out_slot_label_list
        )

        # Merge results
        results = total_result
        results.update(semantic_basic_result)

        # Add predictions
        predictions = {
            "intents": intent_preds.tolist(),
            "slots": slot_preds_list,
            "tokints": intent_token_preds_list,
            "tag_intents": tag_intent_preds_list
        }
        results["predictions"] = predictions

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            if isinstance(results[key], (int, float)):
                logger.info("  %s = %.4f", key, results[key])

        return results

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