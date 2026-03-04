from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from tqdm import tqdm 
from collections import Counter


model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
print("\nテストデータ内の各カテゴリの分布:")
for label, count in test_label_counts.items():
    category_name = [v["カテゴリ名"] for k, v in label_mapping.items() if v["new_label"] == label][0]
    print(f"カテゴリ {label} ({category_name}): {count} 件")

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return predicted_label

correct = 0
total = len(filtered_test_data)

print("\n未学習モデルでのテストを開始します...")
for example in tqdm(filtered_test_data):  
    title = example['title']  
    true_label = example['new_label'] 
    predicted_label = classify(title)
    
    if predicted_label == true_label:
        correct += 1

accuracy = correct / total
print(f"未学習モデルでのテストセットの精度: {accuracy:.3f}")
