import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF
from .layers import (
    MultiIntentClassifier,
    SlotClassifier,
    IntentTokenClassifier,
    TagIntentClassifier,
    BiaffineTagIntentClassifier
)

class VSLIM(nn.Module):
    """
    Features:
    - Multi-intent classification with sigmoid
    - Slot filling with optional CRF
    - Intent token classification
    - Tag-intent classification with B/BI masks
    - Intent attention for tag-intent
    """

    def __init__(self,
             model_name,
             num_slots,
             num_intents,
             num_token_intents,
             num_tag_intents,
             dropout=0.1,
             use_crf=False,
             num_mask=4,
             cls_token_cat=True,
             intent_attn=True,
             use_biaffine_tag_intent=True,
             args=None):
      super().__init__()

      # PhoBERT encoder
      self.encoder = AutoModel.from_pretrained(model_name)
      hidden_size = self.encoder.config.hidden_size

      # Classification heads 
      self.multi_intent_classifier = MultiIntentClassifier(hidden_size, num_intents, dropout)
      self.slot_classifier = SlotClassifier(hidden_size, num_slots, dropout)
      self.intent_token_classifier = IntentTokenClassifier(hidden_size, num_token_intents, dropout)

      # Tag-intent classifier: Biaffine or Linear
      self.use_biaffine_tag_intent = use_biaffine_tag_intent

      if use_biaffine_tag_intent:
          # Biaffine classifier: both h_cls và r have dim = hidden_size
          self.biaffine_tag_intent_classifier = BiaffineTagIntentClassifier(
              hidden_size, num_tag_intents, dropout
          )
      else:
          # Linear classifier (concat [CLS; r])
          tag_input_dim = 2 * hidden_size if cls_token_cat else hidden_size
          self.tag_intent_classifier = TagIntentClassifier(tag_input_dim, num_tag_intents, dropout)

      if use_crf:
          self.crf = CRF(num_tags=num_slots, batch_first=True)

      self.use_crf = use_crf
      self.num_mask = num_mask
      self.cls_token_cat = cls_token_cat
      self.intent_attn = intent_attn
      self.num_intents = num_intents
      self.args = args

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                intent_label_ids=None, slot_labels_ids=None,
                intent_token_ids=None, B_tag_mask=None, BI_tag_mask=None,
                tag_intent_label=None):
        # Encode with PhoBERT
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output        # [batch, hidden]

        total_loss = 0

        # Get loss weights from args or use defaults
        W_UTTINTENT = self.args.intent_loss_coef if self.args else 1.0
        W_SLOT = self.args.slot_loss_coef if self.args else 2.0
        # W_TOKINTENT = 2.0  # from notebook
        # W_UTTINTENT = 1.0  # from notebook
        W_TOKINTENT = self.args.token_intent_loss_coef if self.args else 2.0  
        W_TAGINTENT = self.args.tag_intent_coef if self.args else 1.0
        IGNORE_INDEX = self.args.ignore_index if self.args else -100

        # ==================================== 1. Multi-Intent Classification ========================================
        intent_logits = self.multi_intent_classifier(pooled_output)  # [batch, num_intents]

        if intent_label_ids is not None:
            intent_loss_fct = nn.BCELoss()
            intent_loss = intent_loss_fct(intent_logits + 1e-10, intent_label_ids)
            total_loss += W_UTTINTENT * intent_loss

        # ==================================== 2. Slot Classification ========================================
        slot_logits = self.slot_classifier(sequence_output)  # [batch, seq_len, num_slots]

        if slot_labels_ids is not None:
            if self.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, slot_logits.size(-1))[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, slot_logits.size(-1)), slot_labels_ids.view(-1))

            total_loss += W_SLOT * slot_loss

        # ==================================== 3. Intent Token Classification ========================================
        intent_token_logits = self.intent_token_classifier(sequence_output)  

        intent_token_loss = 0.0
        if intent_token_ids is not None:  
            if self.use_crf:
                intent_token_loss = self.crf(intent_token_logits, intent_token_ids, mask=attention_mask.byte(), reduction='mean')
                intent_token_loss = -1 * intent_token_loss
            else:
                intent_token_loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                if attention_mask is not None:
                    active_intent_loss = attention_mask.view(-1) == 1
                    active_intent_logits = intent_token_logits.view(-1, intent_token_logits.size(-1))[active_intent_loss]
                    active_intent_tokens = intent_token_ids.view(-1)[active_intent_loss]
                    intent_token_loss = intent_token_loss_fct(active_intent_logits, active_intent_tokens)
                else:
                    intent_token_loss = intent_token_loss_fct(intent_token_logits.view(-1, intent_token_logits.size(-1)), intent_token_ids.view(-1))

            total_loss += W_TOKINTENT * intent_token_loss

        # ==================================== 4. Tag-Intent Classification ========================================
        tag_intent_loss = 0.0
        tag_intent_logits = None

        if B_tag_mask is not None and BI_tag_mask is not None and tag_intent_label is not None:
            # if BI_tag_mask.type() != torch.cuda.FloatTensor:
            #     BI_tag_mask = BI_tag_mask.type(torch.cuda.FloatTensor)
            # if B_tag_mask.type() != torch.cuda.FloatTensor:
            #     B_tag_mask = B_tag_mask.type(torch.cuda.FloatTensor)
            if BI_tag_mask.type() != torch.float32:
                BI_tag_mask = BI_tag_mask.type(torch.float32)
            if B_tag_mask.type() != torch.float32:
                B_tag_mask = B_tag_mask.type(torch.float32)

            # Use BI_tag_mask for weighted pooling
            tag_intent_vec = torch.einsum('bml,bld->bmd', BI_tag_mask, sequence_output)  # [batch, num_mask, hidden]

            # BIAFFINE TAG-INTENT CLASSIFICATION
            if self.use_biaffine_tag_intent:
                # h_cls: pooled_output [batch, hidden]
                h_cls = pooled_output.unsqueeze(1)  # [batch, 1, hidden]
                h_cls = h_cls.repeat(1, self.num_mask, 1)  # [batch, num_mask, hidden]

                # r: tag_intent_vec [batch, num_mask, hidden]
                batch_size = h_cls.size(0)
                h_cls_flat = h_cls.view(batch_size * self.num_mask, -1)  # [batch*num_mask, hidden]
                r_flat = tag_intent_vec.view(batch_size * self.num_mask, -1)  # [batch*num_mask, hidden]

                # Biaffine classification
                tag_intent_logits = self.biaffine_tag_intent_classifier(h_cls_flat, r_flat)  # [batch*num_mask, num_tag_intents]

            else:
                if self.cls_token_cat:
                    cls_token = pooled_output.unsqueeze(1)  # [batch, 1, hidden]
                    cls_token = cls_token.repeat(1, self.num_mask, 1)  # [batch, num_mask, hidden]
                    tag_intent_vec = torch.cat((cls_token, tag_intent_vec), dim=2)  # [batch, num_mask, 2*hidden]

                tag_intent_vec = tag_intent_vec.view(tag_intent_vec.size(0) * tag_intent_vec.size(1), -1)
                tag_intent_logits = self.tag_intent_classifier(tag_intent_vec)  # [batch*num_mask, num_tag_intents]

            if self.intent_attn:
                intent_probs = intent_logits.unsqueeze(1)  # [batch, 1, num_intents]
                intent_probs = intent_probs.repeat(1, self.num_mask, 1)  # [batch, num_mask, num_intents]
                intent_probs = intent_probs.view(intent_probs.size(0) * intent_probs.size(1), -1)  # [batch*num_mask, num_intents]

                # Add PAD dimension to intent_probs
                pad_probs = torch.zeros(intent_probs.size(0), 1, device=intent_probs.device)  # [batch*num_mask, 1] for PAD
                intent_probs_expanded = torch.cat([pad_probs, intent_probs], dim=1)  # [batch*num_mask, 6]

                # Apply attention weighting
                tag_intent_logits = tag_intent_logits * intent_probs_expanded
                tag_intent_logits = tag_intent_logits.div(tag_intent_logits.sum(dim=1, keepdim=True) + 1e-10)

            nll_fct = nn.NLLLoss(ignore_index=IGNORE_INDEX)
            tag_intent_loss = nll_fct(torch.log(tag_intent_logits + 1e-10), tag_intent_label.view(-1))
            total_loss += W_TAGINTENT * tag_intent_loss

        return {
            "total_loss": total_loss,
            "intent_loss": intent_loss if intent_label_ids is not None else 0,
            "slot_loss": slot_loss if slot_labels_ids is not None else 0,
            "intent_token_loss": intent_token_loss,
            "tag_intent_loss": tag_intent_loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            "intent_token_logits": intent_token_logits,
            "tag_intent_logits": tag_intent_logits if B_tag_mask is not None else None
        }

