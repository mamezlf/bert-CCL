from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from tqdm import tqdm  # 進捗バーを表示
from collections import Counter

# **加载一个未训练的多语言 BERT 模型**
model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 未训练的模型
model.eval()

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# **加载并筛选 KLUE YNAT 数据集**
dataset = load_dataset("klue", "ynat")

# 筛选需要的标签（例如：label 0, 3, 5）
selected_labels = [0, 3, 5]

# 筛选函数
def filter_labels(example):
    return example['label'] in selected_labels

# 筛选后的测试数据
filtered_test_data = dataset['validation'].filter(filter_labels)

# 新的标签映射
label_mapping = {
    0: {"new_label": 0, "カテゴリ名": "IT"},
    3: {"new_label": 1, "カテゴリ名": "生活文化"},
    5: {"new_label": 2, "カテゴリ名": "スポーツ"},
}

# 映射新的标签
filtered_test_data = filtered_test_data.map(lambda x: {"new_label": label_mapping[x['label']]["new_label"]})

# 将韩语测试数据随机采样到 800 条
filtered_test_data = filtered_test_data.shuffle(seed=42).select(range(min(len(filtered_test_data), 800)))

# **打印筛选后的测试数据分布**
test_label_counts = Counter(filtered_test_data['new_label'])
print(f"テストデータの総サンプル数: {len(filtered_test_data)}")
print("\nテストデータ内の各カテゴリの分布:")
for label, count in test_label_counts.items():
    category_name = [v["カテゴリ名"] for k, v in label_mapping.items() if v["new_label"] == label][0]
    print(f"カテゴリ {label} ({category_name}): {count} 件")

# **分类函数**
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return predicted_label

# **在筛选后的测试数据上进行分类**
correct = 0
total = len(filtered_test_data)

print("\n未学習モデルでのテストを開始します...")
for example in tqdm(filtered_test_data):  # 使用筛选后的测试数据
    title = example['title']  # 韩语新闻标题
    true_label = example['new_label']  # 从 `example` 中获取新的标签映射
    predicted_label = classify(title)
    
    if predicted_label == true_label:
        correct += 1

# **计算并输出精度**
accuracy = correct / total
print(f"未学習モデルでのテストセットの精度: {accuracy:.3f}")
