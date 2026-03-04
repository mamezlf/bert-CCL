import torch
from transformers import AutoTokenizer, BertModel
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 3)  
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, **kwargs):  
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs, dim=1).item()
    return predicted_label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClassifier()
best_model_path = "./bert_checkpoint/best_model_state.bin"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()


tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

dataset = load_dataset("klue", "ynat")

selected_labels = [0, 3, 5]

def filter_labels(example):
    return example['label'] in selected_labels

filtered_test_data = dataset['validation'].filter(filter_labels)

label_mapping = {
    0: {"new_label": 0, "カテゴリ名": "IT"},
    3: {"new_label": 1, "カテゴリ名": "生活文化"},
    5: {"new_label": 2, "カテゴリ名": "スポーツ"},
}

filtered_test_data = filtered_test_data.map(lambda x: {"new_label": label_mapping[x['label']]["new_label"]})
filtered_test_data = filtered_test_data.shuffle(seed=42).select(range(min(len(filtered_test_data), 800)))

test_label_counts = Counter(filtered_test_data['new_label'])
print(f"テストデータの総サンプル数: {len(filtered_test_data)}")
for label, count in test_label_counts.items():
    category_name = [v["カテゴリ名"] for k, v in label_mapping.items() if v["new_label"] == label][0]
    print(f"カテゴリ {label} ({category_name}): {count} 件")

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs, dim=1).item()
    return predicted_label

correct = 0
total = len(filtered_test_data)

print("\n保存されたモデルでのテストを開始します...")
for example in tqdm(filtered_test_data):
    title = example['title']
    true_label = example['new_label']
    predicted_label = classify(title)

    if predicted_label == true_label:
        correct += 1

accuracy = correct / total
print(f"保存されたモデルでのテストセットの正解率: {accuracy:.3f}")

