import argparse
import pandas as pd
from src.trainer_logic import run_training
from src.inference_logic import run_inference

def main():
    """CLI entry point for the Russian PII Detection project."""
    parser = argparse.ArgumentParser(description="Russian NER Project with RuBERT")
    subparsers = parser.add_subparsers(dest="command", help="Available commands: train or predict")

    # Arguments for Training mode
    train_parser = subparsers.add_parser("train", help="Train the NER model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to training tsv file")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--output", type=str, default="./best_model", help="Directory to save the trained model")

    # Arguments for Prediction mode
    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained model")
    predict_parser.add_argument("--model", type=str, required=True, help="Path to the saved model directory")
    predict_parser.add_argument("--data", type=str, required=True, help="Path to the test CSV file")
    predict_parser.add_argument("--save", type=str, default="submission.csv", help="Filename for output predictions")

    args = parser.parse_args()

    # Route logic based on subcommand
    if args.command == "train":
        run_training(args.data, args.output, args.epochs)
    elif args.command == "predict":
        run_inference(args.model, args.data, args.save)
    else:
        # Show help if no valid subcommand is provided
        parser.print_help()

if __name__ == "__main__":
    main()