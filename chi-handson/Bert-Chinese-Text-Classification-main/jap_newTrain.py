import os
import torch
import random
from transformers import BertTokenizer, AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel
import torch.nn as nn
from tqdm import tqdm
import pandas as pd  # 如果是CSV数据需要

# 设置随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(0)

# 定义BertClassifier类
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 3)  # 假设分类有3个类别
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

# 保存路径
save_path = './bert_checkpoint'
os.makedirs(save_path, exist_ok=True)

# 初始化模型（不加载预训练权重）
model = BertClassifier()

# 定义保存最佳模型的路径
best_model_path = os.path.join(save_path, 'best_model_state.bin')

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 使用BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# 加载训练和测试数据集（需要自定义数据加载逻辑）
train_data = pd.read_csv('livedoor_sentence_train.csv')  # 替换为实际训练数据路径
test_data = pd.read_csv('livedoor_sentence_test.csv')  # 替换为实际测试数据路径

# 编码训练数据
train_encoding = tokenizer(train_data['sentence'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
train_input_ids = train_encoding['input_ids']
train_attention_mask = train_encoding['attention_mask']
train_labels = torch.tensor(train_data['label'].tolist())

# 编码测试数据
test_encoding = tokenizer(test_data['sentence'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
test_input_ids = test_encoding['input_ids']
test_attention_mask = test_encoding['attention_mask']
test_labels = torch.tensor(test_data['label'].tolist())

# 创建训练和测试DataLoader
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=1e-5)

# 初始化最优测试精度
best_test_accuracy = 0.0

# 开始训练
print("日语ニュース分類のトレーニングを開始します...")
for epoch in range(5):  # 假设训练5个epoch
    model.train()
    epoch_loss = 0
    total_acc_train = 0

    # **训练过程**
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

    # **测试过程**
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

    # 保存在测试集上表现最好的模型
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)  # 保存模型
        print(f"エポック {epoch + 1}: モデルを保存しました（テスト精度: {test_accuracy:.3f}）")

print("トレーニングが完了しました。")
print(f"ベストモデルは {best_model_path} に保存されました。")

