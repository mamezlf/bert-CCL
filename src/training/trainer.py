import os
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    criterion=None,
    epoch_index: Optional[int] = None,
    desc: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Returns:
        {
            "loss": float,
            "accuracy": float
        }
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if desc is None:
        desc = f"Train Epoch {epoch_index + 1}" if epoch_index is not None else "Training"

    progress_bar = tqdm(dataloader, desc=desc)

    for batch in progress_bar:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


@torch.no_grad()
def evaluate_one_epoch(
    model,
    dataloader: DataLoader,
    device: torch.device,
    criterion=None,
    desc: str = "Evaluating",
) -> Dict[str, float]:
    """
    Evaluate model for one epoch.

    Returns:
        {
            "loss": float,
            "accuracy": float
        }
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=desc)

    for batch in progress_bar:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def save_checkpoint(model, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def fit(
    model,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    optimizer,
    device: torch.device,
    num_epochs: int = 5,
    criterion=None,
    save_best: bool = True,
    best_model_path: Optional[str] = None,
    monitor: str = "accuracy",
    maximize_metric: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generic training loop.

    Args:
        model: PyTorch model
        train_loader: training DataLoader
        eval_loader: validation/test DataLoader used for model selection
        optimizer: optimizer
        device: torch.device
        num_epochs: number of epochs
        criterion: loss function, default CrossEntropyLoss
        save_best: whether to save best model
        best_model_path: save path
        monitor: metric to monitor, "accuracy" or "loss"
        maximize_metric: True for accuracy, False for loss
        verbose: print logs

    Returns:
        {
            "history": {
                "train_loss": [...],
                "train_accuracy": [...],
                "eval_loss": [...],
                "eval_accuracy": [...],
            },
            "best_score": float or None,
            "best_epoch": int or None
        }
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "eval_loss": [],
        "eval_accuracy": [],
    }

    best_score = None
    best_epoch = None

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            epoch_index=epoch,
            desc=f"Epoch {epoch + 1}/{num_epochs} - train",
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])

        if verbose:
            print(
                f"[Epoch {epoch + 1}] "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.4f}"
            )

        current_score = None

        if eval_loader is not None:
            eval_metrics = evaluate_one_epoch(
                model=model,
                dataloader=eval_loader,
                device=device,
                criterion=criterion,
                desc=f"Epoch {epoch + 1}/{num_epochs} - eval",
            )

            history["eval_loss"].append(eval_metrics["loss"])
            history["eval_accuracy"].append(eval_metrics["accuracy"])

            if verbose:
                print(
                    f"[Epoch {epoch + 1}] "
                    f"eval_loss={eval_metrics['loss']:.4f}, "
                    f"eval_acc={eval_metrics['accuracy']:.4f}"
                )

            if monitor not in eval_metrics:
                raise ValueError(
                    f"monitor='{monitor}' not found in eval metrics. "
                    f"Available metrics: {list(eval_metrics.keys())}"
                )

            current_score = eval_metrics[monitor]

            is_better = False
            if best_score is None:
                is_better = True
            elif maximize_metric and current_score > best_score:
                is_better = True
            elif (not maximize_metric) and current_score < best_score:
                is_better = True

            if is_better:
                best_score = current_score
                best_epoch = epoch + 1

                if save_best and best_model_path is not None:
                    save_checkpoint(model, best_model_path)
                    if verbose:
                        print(
                            f"Saved best model to {best_model_path} "
                            f"(epoch={best_epoch}, {monitor}={best_score:.4f})"
                        )

        else:
            history["eval_loss"].append(None)
            history["eval_accuracy"].append(None)

    return {
        "history": history,
        "best_score": best_score,
        "best_epoch": best_epoch,
    }