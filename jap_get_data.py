import os
import torch
import random
import pandas as pd
from transformers import BertTokenizer, AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader
from transformers import BertModel
import torch.nn as nn

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(0)

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
model = BertClassifier()
model = model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

df = pd.read_csv('livedoor_sentence_train.csv')  

encoding = tokenizer(df['sentence'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
labels = torch.tensor(df['label'].tolist())

dataset = TensorDataset(input_ids, attention_mask, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

optimizer = AdamW(model.parameters(), lr=1e-5)

save_path = './bert_checkpoint'
os.makedirs(save_path, exist_ok=True)

best_val_accuracy = 0.0

for epoch in range(5):  
    model.train()
    epoch_loss = 0
    total_acc_train = 0
    for batch in DataLoader(train_dataset, batch_size=16, shuffle=True):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids, attention_mask)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        acc = (outputs.argmax(dim=1) == labels).sum().item()
        total_acc_train += acc

    avg_epoch_loss = epoch_loss / len(train_dataset)
    train_acc = total_acc_train / len(train_dataset)
    print(f"エポック {epoch + 1} - 損失: {avg_epoch_loss:.4f}, 訓練精度: {train_acc:.3f}")

    model.eval()
    total_acc_val = 0
    with torch.no_grad():
        for batch in DataLoader(val_dataset, batch_size=16):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask)
            acc = (outputs.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc

    val_accuracy = total_acc_val / len(val_dataset)
    print(f"エポック {epoch + 1} - 検証精度: {val_accuracy:.3f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), os.path.join(save_path, 'best_model_state.bin'))
        print(f"エポック {epoch + 1}: モデルを保存しました（検証精度: {val_accuracy:.3f}）")



