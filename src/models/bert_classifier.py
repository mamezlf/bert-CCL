import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-uncased",
        num_labels: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        logits = self.linear(self.dropout(pooled_output))
        return logits