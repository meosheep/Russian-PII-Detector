import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, DataCollatorForTokenClassification, EarlyStoppingCallback
from datasets import Dataset
from seqeval.metrics import f1_score, classification_report
from .model_utils import apply_multi_sample_dropout, WeightedTrainer
from .data_utils import prepare_raw_data, get_label_maps, get_process_func

def run_training(data_path, output_dir, epochs):
    df = prepare_raw_data(data_path)
    id2label, label2id = get_label_maps(df)
    
    model_checkpoint = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[id2label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[id2label[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        print("\n" + classification_report(true_labels, true_predictions, digits=4))
        return {"f1": f1_score(true_labels, true_predictions)}

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model = apply_multi_sample_dropout(model)

    raw_dataset = Dataset.from_pandas(df[['text', 'target']])
    process_func = get_process_func(tokenizer, label2id)
    tokenized_dataset = raw_dataset.map(process_func, batched=True, remove_columns=raw_dataset.column_names)
    ds_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    training_args = TrainingArguments(
        output_dir="./temp_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        save_total_limit=1
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_split["train"],
        eval_dataset=ds_split["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)