import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 進捗バーを表示

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT多言語モデルとトークナイザーのロード（新規モデルの初期化）
model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# モデルを新規に初期化（学習済みの分類ヘッドを使わない）
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # カテゴリ数を3と仮定
# NOTE: このモデルは完全に新規な状態で初期化されており、何も学習していない

model.to(device)
model.eval()

# テストデータの読み込み
test_df = pd.read_csv('livedoor_sentence_test.csv')  # 日本語テストデータ

# テストデータのエンコード
encoding = tokenizer(test_df['sentence'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
labels = torch.tensor(test_df['label'].tolist())

# TensorDatasetとDataLoaderの作成
test_dataset = TensorDataset(input_ids, attention_mask, labels)
test_loader = DataLoader(test_dataset, batch_size=16)

# テストデータセットでの評価
correct_predictions = 0
total_predictions = len(test_dataset)

print("未学習モデルでテストを開始します...")
print(f"テストデータセットの総サンプル数: {len(test_dataset)}")

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 予測ラベルの取得
        predicted_labels = outputs.logits.argmax(dim=1)
        
        # 正解率の計算
        correct_predictions += (predicted_labels == labels).sum().item()

# 総正解率の計算と出力
accuracy = correct_predictions / total_predictions
print(f"未学習モデルでのテストデータセットの正解率: {accuracy:.3f}")
