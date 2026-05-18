# Russian PII Detector (NER)

This repository contains a Named Entity Recognition (NER) system designed to identify and mask Personally Identifiable Information (PII) in Russian language texts.

### Project Context
This project was developed for the AI Champions Hackathon (ITMO University) addressing a business case provided by Alfa-Bank.

The objective was to create a reliable mechanism for data anonymization. In many enterprise workflows, sensitive user data must be masked before being transmitted to Large Language Models (LLMs) to ensure privacy and regulatory compliance. This project solves the problem by treating it as a sequence labeling task.

### Performance
The current model architecture achieves a final F1-score of 0.979 on the evaluation dataset.

### Technical Implementation Details
*   **Base Architecture:** the system utilizes the `DeepPavlov/rubert-base-cased` transformer model as the backbone.
*   **Multi-Sample Dropout:** implemented to improve regularization and prevent overfitting, leading to more stable training.
*   **Weighted Loss Function:** a custom Cross-Entropy loss with class weights was used to mitigate the extreme class imbalance typical for NER tasks (where the majority of tokens belong to the "O" class).
*   **Training Optimization:** includes Early Stopping and FP16 mixed-precision training for faster convergence and reduced memory footprint.

### Project Structure
*   `src/`: Contains modular source code for data alignment, model configuration, and training/inference logic.
*   `main.py`: The primary command-line interface for the project.
*   `data/`: Directory for input datasets (contains .gitkeep to preserve structure).
*   `model/`: Directory for exported model weights and configurations.

### Usage Guide

#### 1. Requirements
Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

#### 2. Training
To initiate the training process, provide the path to the training dataset and specify the output directory:
```bash
python main.py train --data data/train_dataset.tsv --epochs 20 --output ./model/best_model
```

#### 3. Inference
To generate predictions for new data using a trained model:
```bash
python main.py predict --model ./model/best_model --data data/test.csv --save submission.csv
```

### Data Anonymization Example
**Original Text:** "please send the internal report to Ivanov Ivan at Arbat St 1, Moscow."  
**Masked Output:** "please send the internal report to [NAME] at [ADDRESS]."