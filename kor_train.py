import os
import torch
import random
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

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
best_model_path = './bert_checkpoint/best_model_state.bin'  # 日语模型的保存路径
model.load_state_dict(torch.load(best_model_path, map_location=device)) 
model = model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

dataset = load_dataset("klue", "ynat")


selected_labels = [0, 3, 5]

def filter_labels(example):
    return example['label'] in selected_labels

filtered_train_data = dataset['train'].filter(filter_labels)
filtered_test_data = dataset['validation'].filter(filter_labels)

label_mapping = {
    0: {"new_label": 0, "カテゴリ名": "IT"},
    3: {"new_label": 1, "カテゴリ名": "生活文化"},
    5: {"new_label": 2, "カテゴリ名": "スポーツ"},
}

filtered_train_data = filtered_train_data.map(lambda x: {"new_label": label_mapping[x['label']]["new_label"]})
filtered_test_data = filtered_test_data.map(lambda x: {"new_label": label_mapping[x['label']]["new_label"]})

filtered_train_data = filtered_train_data.shuffle(seed=42).select(range(min(len(filtered_train_data), 3200)))
filtered_test_data = filtered_test_data.shuffle(seed=42).select(range(min(len(filtered_test_data), 800)))

print(f"トレーニングデータの件数: {len(filtered_train_data)}")
print(f"テストデータの件数: {len(filtered_test_data)}")

train_label_counts = Counter(filtered_train_data['new_label'])
test_label_counts = Counter(filtered_test_data['new_label'])

print("\nトレーニングデータ内の各カテゴリの分布:")
for label, count in train_label_counts.items():
    category_name = [v["カテゴリ名"] for k, v in label_mapping.items() if v["new_label"] == label][0]
    print(f"カテゴリ {label} ({category_name}): {count} 件")

print("\nテストデータ内の各カテゴリの分布:")
for label, count in test_label_counts.items():
    category_name = [v["カテゴリ名"] for k, v in label_mapping.items() if v["new_label"] == label][0]
    print(f"カテゴリ {label} ({category_name}): {count} 件")

train_encoding = tokenizer(filtered_train_data['title'], return_tensors='pt', padding=True, truncation=True, max_length=128)
train_input_ids = train_encoding['input_ids']
train_attention_mask = train_encoding['attention_mask']
train_labels = torch.tensor(filtered_train_data['new_label'])

test_encoding = tokenizer(filtered_test_data['title'], return_tensors='pt', padding=True, truncation=True, max_length=128)
test_input_ids = test_encoding['input_ids']
test_attention_mask = test_encoding['attention_mask']
test_labels = torch.tensor(filtered_test_data['new_label'])

train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=1e-5)

print("韓国語ニュース分類のトレーニングを開始します...")
best_test_accuracy = 0.0
best_model_path = './bert_checkpoint/best_model_state.bin' 


print("韓国語ニュース分類のトレーニングを開始します...")
for epoch in range(5):  
    model.train()
    epoch_loss = 0
    total_acc_train = 0

    for batch in tqdm(train_loader, desc=f"エポック {epoch + 1} - 訓練中"):
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
    total_acc_test = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="テスト中"):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask)
            acc = (outputs.argmax(dim=1) == labels).sum().item()
            total_acc_test += acc

    test_accuracy = total_acc_test / len(test_dataset)
    print(f"エポック {epoch + 1} - テスト精度: {test_accuracy:.3f}")

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)  
        print(f"エポック {epoch + 1}: モデルを保存しました（テスト精度: {test_accuracy:.3f}）")

print("トレーニングが完了しました。")
print(f"ベストモデルは {best_model_path} に保存されました。")

