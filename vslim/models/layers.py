import torch
import torch.nn as nn


class MultiIntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(MultiIntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)
        self.sigmoid = nn.Sigmoid()
        self.reset_params()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return self.sigmoid(x)

    def reset_params(self):
        nn.init.uniform_(self.linear.weight)
        nn.init.uniform_(self.linear.bias)

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.2):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class IntentTokenClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentTokenClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class TagIntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(TagIntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(x)
        return self.softmax(self.linear(x))


class BiaffineTagIntentClassifier(nn.Module):
    """
    Biaffine Tag-Intent Classifier
    score = h_cls^T U r + W [h_cls; r] + b
    """
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(BiaffineTagIntentClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_intent_labels = num_intent_labels

        self.dropout = nn.Dropout(dropout_rate)

        # Bilinear term: U (input_dim x input_dim x num_intent_labels)
        self.U = nn.Parameter(torch.Tensor(num_intent_labels, input_dim, input_dim))

        # Linear term: W (2*input_dim x num_intent_labels)
        self.W = nn.Linear(2 * input_dim, num_intent_labels)

        # Softmax
        self.softmax = nn.Softmax(dim=1)

        self.reset_params()

    def forward(self, h_cls, r):
        """
        Args:
            h_cls: [batch*num_mask, hidden_dim] - CLS representations
            r: [batch*num_mask, hidden_dim] - tag intent vectors

        Returns:
            [batch*num_mask, num_intent_labels] - probabilities
        """
        h_cls = self.dropout(h_cls)
        r = self.dropout(r)

        # Bilinear term: h_cls^T U r
        # h_cls: [B, H], U: [C, H, H], r: [B, H]
        # Result: [B, C]
        bilinear_scores = torch.einsum('bh,chd,bd->bc', h_cls, self.U, r)

        # Linear term: W [h_cls; r]
        concat = torch.cat([h_cls, r], dim=1)  # [B, 2H]
        linear_scores = self.W(concat)  # [B, C]

        # Total score
        scores = bilinear_scores + linear_scores

        return self.softmax(scores)

    def reset_params(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

