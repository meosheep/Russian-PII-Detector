import argparse
import pandas as pd
from src.trainer_logic import run_training
from src.inference_logic import run_inference

def main():
    parser = argparse.ArgumentParser(description="Russian NER Project with RuBERT")
    subparsers = parser.add_subparsers(dest="command", help="Команды: train или predict")

    train_parser = subparsers.add_parser("train", help="Запустить обучение модели")
    train_parser.add_argument("--data", type=str, required=True, help="Путь к train.tsv")
    train_parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    train_parser.add_argument("--output", type=str, default="./best_model", help="Путь для сохранения")

    predict_parser = subparsers.add_parser("predict", help="Запустить инференс")
    predict_parser.add_argument("--model", type=str, required=True, help="Путь к сохраненной модели")
    predict_parser.add_argument("--data", type=str, required=True, help="Путь к тестовому CSV")
    predict_parser.add_argument("--save", type=str, default="submission.csv", help="Путь для сохранения результата")

    args = parser.parse_args()

    if args.command == "train":
        run_training(args.data, args.output, args.epochs)
    elif args.command == "predict":
        run_inference(args.model, args.data, args.save)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()