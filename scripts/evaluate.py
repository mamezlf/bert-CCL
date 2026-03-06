import os
import sys
import argparse
import torch
from transformers import BertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.bert_classifier import BertClassifier
from src.data.ja_dataset import build_ja_dataloaders
from src.data.ko_dataset import build_ko_dataloaders
from src.evaluation.evaluator import evaluate_model, evaluate_checkpoint
from src.utils.seed import seed_everything


MODEL_NAME = "bert-base-multilingual-uncased"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified evaluation script for JA/KO classification")

    parser.add_argument("--lang", type=str, required=True, choices=["ja", "ko"], help="Language/task to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Tokenizer max length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, evaluate base model.",
    )

    parser.add_argument(
        "--return_predictions",
        action="store_true",
        help="Whether to return predictions and labels",
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
            shuffle_train=False,
        )
        return bundle

    if args.lang == "ko":
        bundle = build_ko_dataloaders(
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle_train=False,
        )
        return bundle

    raise ValueError(f"Unsupported lang: {args.lang}")


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

    if args.checkpoint is not None:
        print(f"\nEvaluating trained model from checkpoint: {args.checkpoint}")
        result = evaluate_checkpoint(
            model=model,
            checkpoint_path=args.checkpoint,
            dataloader=bundle["test_loader"],
            device=device,
            criterion=torch.nn.CrossEntropyLoss(),
            return_predictions=args.return_predictions,
            desc="Evaluating trained checkpoint",
        )
    else:
        print("\nEvaluating base model (random classifier head on top of pretrained mBERT encoder)")
        result = evaluate_model(
            model=model,
            dataloader=bundle["test_loader"],
            device=device,
            criterion=torch.nn.CrossEntropyLoss(),
            return_predictions=args.return_predictions,
            desc="Evaluating base model",
        )

    print("\nEvaluation result:")
    print(f"  loss: {result['loss']:.4f}")
    print(f"  accuracy: {result['accuracy']:.4f}")

    if args.return_predictions:
        print(f"  num_predictions: {len(result['predictions'])}")
        print(f"  num_labels: {len(result['labels'])}")


if __name__ == "__main__":
    main()