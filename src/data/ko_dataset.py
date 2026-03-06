import torch
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from collections import Counter


DEFAULT_SELECTED_LABELS = [0, 3, 5]

DEFAULT_LABEL_MAPPING = {
    0: {"new_label": 0, "category_name": "IT"},
    3: {"new_label": 1, "category_name": "生活文化"},
    5: {"new_label": 2, "category_name": "スポーツ"},
}


def filter_selected_labels(example, selected_labels):
    return example["label"] in selected_labels


def remap_label(example, label_mapping):
    original_label = example["label"]
    return {
        "new_label": label_mapping[original_label]["new_label"]
    }


def prepare_ko_split(
    split_name: str,
    selected_labels=None,
    label_mapping=None,
    sample_size: int | None = None,
    seed: int = 42,
):
    if selected_labels is None:
        selected_labels = DEFAULT_SELECTED_LABELS
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING

    dataset = load_dataset("klue", "ynat")[split_name]
    dataset = dataset.filter(lambda x: filter_selected_labels(x, selected_labels))
    dataset = dataset.map(lambda x: remap_label(x, label_mapping))

    if sample_size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(len(dataset), sample_size)))

    return dataset


def build_tensor_dataset_from_hf(
    hf_dataset,
    tokenizer,
    text_col: str = "title",
    label_col: str = "new_label",
    max_length: int = 128,
):
    encoding = tokenizer(
        hf_dataset[text_col],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = torch.tensor(hf_dataset[label_col], dtype=torch.long)

    dataset = TensorDataset(
        encoding["input_ids"],
        encoding["attention_mask"],
        labels,
    )
    return dataset


def summarize_label_distribution(hf_dataset, label_col="new_label"):
    return dict(Counter(hf_dataset[label_col]))


def build_ko_datasets(
    tokenizer,
    max_length: int = 128,
    train_sample_size: int = 3200,
    test_sample_size: int = 800,
    seed: int = 42,
    selected_labels=None,
    label_mapping=None,
):
    if selected_labels is None:
        selected_labels = DEFAULT_SELECTED_LABELS
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING

    train_data = prepare_ko_split(
        split_name="train",
        selected_labels=selected_labels,
        label_mapping=label_mapping,
        sample_size=train_sample_size,
        seed=seed,
    )

    test_data = prepare_ko_split(
        split_name="validation",
        selected_labels=selected_labels,
        label_mapping=label_mapping,
        sample_size=test_sample_size,
        seed=seed,
    )

    train_dataset = build_tensor_dataset_from_hf(
        hf_dataset=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = build_tensor_dataset_from_hf(
        hf_dataset=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    compact_label_mapping = {
        int(old_label): int(info["new_label"])
        for old_label, info in label_mapping.items()
    }

    meta = {
        "language": "ko",
        "dataset_name": "klue/ynat",
        "train_size": len(train_data),
        "test_size": len(test_data),
        "text_column": "title",
        "label_column": "new_label",
        "num_labels": len(set(compact_label_mapping.values())),
        "selected_labels": selected_labels,
        "train_label_distribution": summarize_label_distribution(train_data),
        "test_label_distribution": summarize_label_distribution(test_data),
    }

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "label_mapping": compact_label_mapping,
        "label_mapping_detail": label_mapping,
        "meta": meta,
    }


def build_ko_dataloaders(
    tokenizer,
    max_length: int = 128,
    batch_size: int = 16,
    shuffle_train: bool = True,
    train_sample_size: int = 3200,
    test_sample_size: int = 800,
    seed: int = 42,
    selected_labels=None,
    label_mapping=None,
):
    bundle = build_ko_datasets(
        tokenizer=tokenizer,
        max_length=max_length,
        train_sample_size=train_sample_size,
        test_sample_size=test_sample_size,
        seed=seed,
        selected_labels=selected_labels,
        label_mapping=label_mapping,
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