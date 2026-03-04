import torch
from transformers import AutoTokenizer, BertModel
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

# **自定义 BertClassifier 类**
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 3)  # 假设分类类别为3
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, **kwargs):  # 添加 **kwargs
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

# **分类函数**
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs, dim=1).item()
    return predicted_label


# **加载自定义模型和保存的权重**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClassifier()
best_model_path = "./bert_checkpoint/best_model_state.bin"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

# **加载分词器**
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

# **加载并筛选 KLUE YNAT 数据集**
dataset = load_dataset("klue", "ynat")

# 筛选需要的标签
selected_labels = [0, 3, 5]

# 筛选函数
def filter_labels(example):
    return example['label'] in selected_labels

filtered_test_data = dataset['validation'].filter(filter_labels)

# 新的标签映射
label_mapping = {
    0: {"new_label": 0, "カテゴリ名": "IT"},
    3: {"new_label": 1, "カテゴリ名": "生活文化"},
    5: {"new_label": 2, "カテゴリ名": "スポーツ"},
}

filtered_test_data = filtered_test_data.map(lambda x: {"new_label": label_mapping[x['label']]["new_label"]})
filtered_test_data = filtered_test_data.shuffle(seed=42).select(range(min(len(filtered_test_data), 800)))

# 打印测试数据分布
test_label_counts = Counter(filtered_test_data['new_label'])
print(f"テストデータの総サンプル数: {len(filtered_test_data)}")
for label, count in test_label_counts.items():
    category_name = [v["カテゴリ名"] for k, v in label_mapping.items() if v["new_label"] == label][0]
    print(f"カテゴリ {label} ({category_name}): {count} 件")

# **分类函数**
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs, dim=1).item()
    return predicted_label

# **在测试数据上进行评估**
correct = 0
total = len(filtered_test_data)

print("\n保存されたモデルでのテストを開始します...")
for example in tqdm(filtered_test_data):
    title = example['title']
    true_label = example['new_label']
    predicted_label = classify(title)

    if predicted_label == true_label:
        correct += 1

# **计算并输出精度**
accuracy = correct / total
print(f"保存されたモデルでのテストセットの正解率: {accuracy:.3f}")

