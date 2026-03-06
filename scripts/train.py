import os
import sys
import argparse
import torch
from transformers import BertTokenizer, AdamW

# 让 scripts/ 下运行时也能找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.bert_classifier import BertClassifier
from src.data.ja_dataset import build_ja_dataloaders
from src.data.ko_dataset import build_ko_dataloaders
from src.training.trainer import fit, load_checkpoint
from src.utils.seed import seed_everything


MODEL_NAME = "bert-base-multilingual-uncased"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training script for JA/KO classification")

    parser.add_argument("--lang", type=str, required=True, choices=["ja", "ko"], help="Language/task to train on")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Tokenizer max length")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to initialize model from",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save best checkpoint",
    )

    parser.add_argument(
        "--monitor",
        type=str,
        default="accuracy",
        choices=["accuracy", "loss"],
        help="Metric to monitor for best model",
    )

    parser.add_argument(
        "--data_dir_ja",
        type=str,
        default="data/ja",
        help="JA dataset directory",
    )

    return parser.parse_args()


def build_dataloaders(args, tokenizer):
    if args.lang == "ja":
        train_csv = os.path.join(args.data_dir_ja, "livedoor_sentence_train.csv")
        test_csv = os.path.join(args.data_dir_ja, "livedoor_sentence_test.csv")

        bundle = build_ja_dataloaders(
            tokenizer=tokenizer,
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle_train=True,
        )
        return bundle

    if args.lang == "ko":
        bundle = build_ko_dataloaders(
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle_train=True,
        )
        return bundle

    raise ValueError(f"Unsupported lang: {args.lang}")


def resolve_save_path(args):
    if args.save_dir is None:
        save_dir = os.path.join("checkpoints", args.lang, "scratch" if args.init_checkpoint is None else "continued")
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, "best_model_state.bin")


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bundle = build_dataloaders(args, tokenizer)

    print("Dataset meta:")
    for key, value in bundle["meta"].items():
        print(f"  {key}: {value}")

    model = BertClassifier(model_name=MODEL_NAME, num_labels=3).to(device)

    if args.init_checkpoint is not None:
        print(f"Loading checkpoint from: {args.init_checkpoint}")
        model = load_checkpoint(model, args.init_checkpoint, device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_model_path = resolve_save_path(args)
    maximize_metric = args.monitor != "loss"

    print("\nStart training...")
    print(f"Best model will be saved to: {best_model_path}")

    result = fit(
        model=model,
        train_loader=bundle["train_loader"],
        eval_loader=bundle["test_loader"],
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        criterion=torch.nn.CrossEntropyLoss(),
        save_best=True,
        best_model_path=best_model_path,
        monitor=args.monitor,
        maximize_metric=maximize_metric,
        verbose=True,
    )

    print("\nTraining finished.")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best {args.monitor}: {result['best_score']}")
    print(f"Checkpoint saved at: {best_model_path}")


if __name__ == "__main__":
    main()