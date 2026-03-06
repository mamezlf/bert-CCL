import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


DEFAULT_TEXT_COL = "sentence"
DEFAULT_LABEL_COL = "label"


def load_ja_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if DEFAULT_TEXT_COL not in df.columns:
        raise ValueError(
            f"Column '{DEFAULT_TEXT_COL}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    if DEFAULT_LABEL_COL not in df.columns:
        raise ValueError(
            f"Column '{DEFAULT_LABEL_COL}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[DEFAULT_TEXT_COL, DEFAULT_LABEL_COL]].dropna().reset_index(drop=True)
    return df


def encode_texts(
    texts,
    tokenizer,
    max_length: int = 128,
):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def build_tensor_dataset(
    df: pd.DataFrame,
    tokenizer,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    max_length: int = 128,
) -> TensorDataset:
    encoding = encode_texts(
        texts=df[text_col].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    labels = torch.tensor(df[label_col].tolist(), dtype=torch.long)

    dataset = TensorDataset(
        encoding["input_ids"],
        encoding["attention_mask"],
        labels,
    )
    return dataset


def build_ja_datasets(
    tokenizer,
    train_csv_path: str,
    test_csv_path: str,
    max_length: int = 128,
):
    train_df = load_ja_dataframe(train_csv_path)
    test_df = load_ja_dataframe(test_csv_path)

    train_dataset = build_tensor_dataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = build_tensor_dataset(
        df=test_df,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    unique_labels = sorted(train_df[DEFAULT_LABEL_COL].unique().tolist())
    label_mapping = {int(label): int(label) for label in unique_labels}

    meta = {
        "language": "ja",
        "train_size": len(train_df),
        "test_size": len(test_df),
        "text_column": DEFAULT_TEXT_COL,
        "label_column": DEFAULT_LABEL_COL,
        "num_labels": len(unique_labels),
    }

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "label_mapping": label_mapping,
        "meta": meta,
    }


def build_ja_dataloaders(
    tokenizer,
    train_csv_path: str,
    test_csv_path: str,
    max_length: int = 128,
    batch_size: int = 16,
    shuffle_train: bool = True,
):
    bundle = build_ja_datasets(
        tokenizer=tokenizer,
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        max_length=max_length,
    )

    train_loader = DataLoader(
        bundle["train_dataset"],
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    test_loader = DataLoader(
        bundle["test_dataset"],
        batch_size=batch_size,
        shuffle=False,
    )

    bundle["train_loader"] = train_loader
    bundle["test_loader"] = test_loader
    return bundle