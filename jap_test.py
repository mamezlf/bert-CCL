import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm  

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 3)  
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertClassifier()

best_model_path = "./bert_checkpoint/best_model_state.bin"  
model.load_state_dict(torch.load(best_model_path, map_location=device))  
model.to(device)
model.eval()

test_df = pd.read_csv('livedoor_sentence_test.csv') 

encoding = tokenizer(test_df['sentence'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
labels = torch.tensor(test_df['label'].tolist())

test_dataset = TensorDataset(input_ids, attention_mask, labels)
test_loader = DataLoader(test_dataset, batch_size=16)

correct_predictions = 0
total_predictions = len(test_dataset)

print("保存されたモデルでテストを開始します...")
print(f"テストデータセットの総サンプル数: {len(test_dataset)}")

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        predicted_labels = outputs.argmax(dim=1)
        
        correct_predictions += (predicted_labels == labels).sum().item()

accuracy = correct_predictions / total_predictions
print(f"保存されたモデルでのテストデータセットの正解率: {accuracy:.3f}")

