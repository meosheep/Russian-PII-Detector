import pandas as pd
from ast import literal_eval
from datasets import Dataset

def prepare_raw_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['target'] = df['target'].apply(literal_eval)
    df['target'] = df['target'].apply(lambda annotations:
        [{'start': item[0], 'end': item[1], 'label': item[2]} for item in annotations]
    )
    return df

def get_label_maps(df):
    labels = sorted(list(set(label for t in df['target'] for it in t for label in [it['label']])))
    id2label = {0: "O"}
    for i, l in enumerate(labels):
        id2label[i*2 + 1] = f"B-{l}"
        id2label[i*2 + 2] = f"I-{l}"
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id

def align_labels_with_tokens(text, annotations, tokenizer, label2id, max_length=512):
    tokenized_input = tokenizer(
        text, 
        truncation=True, 
        return_offsets_mapping=True, 
        stride=128,
        max_length=max_length
    )
    offset_mapping = tokenized_input.pop("offset_mapping")
    token_labels = [0] * len(offset_mapping)

    for annotation in annotations:
        start_char, end_char, label = annotation['start'], annotation['end'], annotation['label']
        label_b, label_i = label2id[f"B-{label}"], label2id[f"I-{label}"]
        first_token = True
        for i, (start_t, end_t) in enumerate(offset_mapping):
            if start_t == end_t: 
                token_labels[i] = -100
                continue
            if start_t >= start_char and end_t <= end_char:
                token_labels[i] = label_b if first_token else label_i
                first_token = False
    tokenized_input["labels"] = token_labels
    return tokenized_input

def get_process_func(tokenizer, label2id):
    def process_data(examples):
        all_ins = {"input_ids": [], "attention_mask": [], "labels": []}
        for text, targets in zip(examples["text"], examples["target"]):
            res = align_labels_with_tokens(text, targets, tokenizer, label2id)
            all_ins["input_ids"].append(res["input_ids"])
            all_ins["attention_mask"].append(res["attention_mask"])
            all_ins["labels"].append(res["labels"])
        return all_ins
    return process_data