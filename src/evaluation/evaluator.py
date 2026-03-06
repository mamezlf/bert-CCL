from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    criterion=None,
    return_predictions: bool = False,
    desc: str = "Evaluating",
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataloader.

    Returns:
        {
            "loss": float,
            "accuracy": float,
            "predictions": optional list[int],
            "labels": optional list[int]
        }
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions: List[int] = []
    all_labels: List[int] = []

    progress_bar = tqdm(dataloader, desc=desc)

    for batch in progress_bar:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        correct = (preds == labels).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += batch_size

        if return_predictions:
            all_predictions.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    result = {
        "loss": avg_loss,
        "accuracy": accuracy,
    }

    if return_predictions:
        result["predictions"] = all_predictions
        result["labels"] = all_labels

    return result


def load_model_checkpoint(model, checkpoint_path: str, device: torch.device):
    """
    Load a saved state_dict into the given model.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(
    model,
    checkpoint_path: str,
    dataloader: DataLoader,
    device: torch.device,
    criterion=None,
    return_predictions: bool = False,
    desc: str = "Evaluating checkpoint",
) -> Dict[str, Any]:
    """
    Load checkpoint, then evaluate.
    """
    model = load_model_checkpoint(model, checkpoint_path, device)

    return evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        criterion=criterion,
        return_predictions=return_predictions,
        desc=desc,
    )


def predict_texts(
    model,
    tokenizer,
    texts,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 16,
):
    """
    Predict labels for raw texts.
    """
    model.eval()
    predictions = []

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=1)

        predictions.extend(preds.detach().cpu().tolist())

    return predictions