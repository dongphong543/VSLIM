import os
import logging
import torch
import torch.nn as nn
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

class VSLIMPredictor:
    def __init__(self, model, tokenizer, mappings, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.mappings = mappings
        self.args = args
        self.device = device
        
        # Extract mappings
        self.intent_labels = mappings['INTENT_LABELS']
        self.id2slot = mappings['ID2SLOT']
        self.id2tokint = mappings['ID2TOKINT']
        self.id2tagint = mappings['ID2TAGINT']

    def align_tokens_for_inference(self, tokens, max_len=None):
        """Align pre-tokenized tokens to subwords for inference"""
        if max_len is None:
            max_len = self.args.max_seq_len
            
        subword_tokens = []
        word_to_subword_map = []

        for token in tokens:
            word_to_subword_map.append(len(subword_tokens))
            pieces = self.tokenizer.tokenize(token) or [self.tokenizer.unk_token]
            subword_tokens.extend(pieces)

        # Convert to input IDs with special tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(subword_tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        # Adjust mapping for special tokens
        word_to_subword_map = [idx + 1 for idx in word_to_subword_map]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # Truncate if necessary
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            token_type_ids = token_type_ids[:max_len]
            word_to_subword_map = [idx for idx in word_to_subword_map if idx < max_len]

        # Pad to max_len
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            input_ids.extend([pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            token_type_ids.extend([0] * pad_len)

        return (
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(self.device),
            word_to_subword_map
        )

    def predict_single(self, tokens, threshold=0.5):
        """Full SLIM inference with all paper features"""
        self.model.eval()

        with torch.no_grad():
            input_ids, attention_mask, token_type_ids, word_positions = self.align_tokens_for_inference(tokens)

            # Create dummy B/BI masks for inference
            batch_size, seq_len = input_ids.shape
            B_tag_mask = torch.zeros(batch_size, self.args.num_mask, seq_len, dtype=torch.long, device=self.device)
            BI_tag_mask = torch.zeros(batch_size, self.args.num_mask, seq_len, dtype=torch.float, device=self.device)
            tag_intent_label = torch.full((batch_size, self.args.num_mask), self.args.ignore_index, dtype=torch.long, device=self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                intent_label_ids=None,
                slot_labels_ids=None,
                intent_token_ids=None,
                B_tag_mask=B_tag_mask,
                BI_tag_mask=BI_tag_mask,
                tag_intent_label=tag_intent_label
            )

            # Extract logits
            slot_logits = outputs["slot_logits"][0]
            tokint_logits = outputs["intent_token_logits"][0] if outputs["intent_token_logits"] is not None else None
            uttint_logits = outputs["intent_logits"][0]

            # # Sau dòng 250, thêm:
            # if tokint_logits is not None:
            #     logger.info("Token-intent logits shape: %s", tokint_logits.shape)
            #     logger.info("Sample tokint predictions: %s", tokint_logits[:5].argmax(dim=-1))
            # else:
            #     logger.info("WARNING: tokint_logits is None!")

            # Utterance intent prediction
            uttint_probs = uttint_logits.cpu().numpy()
            predicted_intents = []

            for i, prob in enumerate(uttint_probs):
                if prob >= threshold:
                    predicted_intents.append(self.intent_labels[i])

            if not predicted_intents:
                best_idx = np.argmax(uttint_probs)
                predicted_intents = [self.intent_labels[best_idx]]

            # Token-level predictions
            slot_predictions = []
            tokint_predictions = []

            for word_idx, subword_pos in enumerate(word_positions):
                if subword_pos >= slot_logits.size(0):
                    slot_predictions.append("O")
                    tokint_predictions.append("O")
                    continue

                # Slot prediction
                slot_id = torch.argmax(slot_logits[subword_pos]).item()
                slot_tag = self.id2slot[slot_id]
                slot_predictions.append(slot_tag)

                # Token-intent prediction
                if tokint_logits is not None:
                    if slot_tag == "O":
                        tokint_predictions.append("O")
                    else:
                        tokint_id = torch.argmax(tokint_logits[subword_pos]).item()
                        tokint_tag = self.id2tokint[tokint_id]
                        tokint_predictions.append(tokint_tag)
                else:
                    tokint_predictions.append("O")

            # Ensure output length matches input length
            num_tokens = len(tokens)
            if len(slot_predictions) < num_tokens:
                slot_predictions.extend(["O"] * (num_tokens - len(slot_predictions)))
                tokint_predictions.extend(["O"] * (num_tokens - len(tokint_predictions)))
            elif len(slot_predictions) > num_tokens:
                slot_predictions = slot_predictions[:num_tokens]
                tokint_predictions = tokint_predictions[:num_tokens]

        return {
            "utterance_intents": predicted_intents,
            "slot_tags": slot_predictions,
            "token_intents": tokint_predictions
        }

    def generate_predicted_masks_from_slots(self, slot_preds_list, max_seq_len):
        """Generate B AND BI masks from predicted slots"""
        B_tag_mask_pred = []
        BI_tag_mask_pred = []

        for i in range(len(slot_preds_list)):
            entities = get_entities(slot_preds_list[i])
            entities = [tag for tag in entities if slot_preds_list[i][tag[1]].startswith('B')]

            if len(entities) > self.args.num_mask:
                entities = entities[:self.args.num_mask]

            B_entity_masks = []
            BI_entity_masks = []

            for entity_idx, entity in enumerate(entities):
                # B mask: only mark beginning
                B_mask = [0 for _ in range(max_seq_len)]
                start_idx = entity[1]
                B_mask[start_idx] = 1
                B_entity_masks.append(B_mask)

                # BI mask: weighted span
                BI_mask = [0.0 for _ in range(max_seq_len)]
                end_idx = entity[2] + 1
                weight = 1.0 / (end_idx - start_idx)
                for pos in range(start_idx, end_idx):
                    if pos < len(slot_preds_list[i]):
                        BI_mask[pos] = weight
                BI_entity_masks.append(BI_mask)

            # Pad to NUM_MASK
            for extra_idx in range(self.args.num_mask - len(B_entity_masks)):
                B_entity_masks.append([0 for _ in range(max_seq_len)])
                BI_entity_masks.append([0.0 for _ in range(max_seq_len)])

            B_tag_mask_pred.append(B_entity_masks)
            BI_tag_mask_pred.append(BI_entity_masks)

        return torch.LongTensor(B_tag_mask_pred), torch.FloatTensor(BI_tag_mask_pred)

    def align_masks_to_subwords(self, masks, word_to_subword_map, max_len):
        """Align word-level masks to subword-level masks"""
        num_masks = len(masks)
        aligned_masks = torch.zeros(num_masks, max_len, dtype=torch.float)

        for mask_idx in range(num_masks):
            for word_idx, subword_idx in enumerate(word_to_subword_map):
                if word_idx < len(masks[mask_idx]) and subword_idx < max_len:
                    aligned_masks[mask_idx, subword_idx] = masks[mask_idx][word_idx]

        return aligned_masks
