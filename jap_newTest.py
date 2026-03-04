import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  

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

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        predicted_labels = outputs.logits.argmax(dim=1)
        
        correct_predictions += (predicted_labels == labels).sum().item()

accuracy = correct_predictions / total_predictions
print(f"未学習モデルでのテストデータセットの正解率: {accuracy:.3f}")
