import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm  # 進捗バーを表示

# **自定义BertClassifier类**
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 3)  # 分类类别数为3
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

# **设备设置**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **加载自定义模型和分词器**
model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertClassifier()

# **加载最佳模型权重**
best_model_path = "./bert_checkpoint/best_model_state.bin"  # 替换为你的最佳模型路径
model.load_state_dict(torch.load(best_model_path, map_location=device))  # 加载最佳模型权重
model.to(device)
model.eval()

# **加载测试数据**
test_df = pd.read_csv('livedoor_sentence_test.csv')  # 替换为你的测试数据文件路径

# **对测试数据进行编码**
encoding = tokenizer(test_df['sentence'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
labels = torch.tensor(test_df['label'].tolist())

# **创建TensorDataset和DataLoader**
test_dataset = TensorDataset(input_ids, attention_mask, labels)
test_loader = DataLoader(test_dataset, batch_size=16)

# **在测试数据上进行评估**
correct_predictions = 0
total_predictions = len(test_dataset)

print("保存されたモデルでテストを開始します...")
print(f"テストデータセットの総サンプル数: {len(test_dataset)}")

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # **获取预测标签**
        predicted_labels = outputs.argmax(dim=1)
        
        # **计算准确率**
        correct_predictions += (predicted_labels == labels).sum().item()

# **计算并输出总体准确率**
accuracy = correct_predictions / total_predictions
print(f"保存されたモデルでのテストデータセットの正解率: {accuracy:.3f}")

