import torch
from torch import nn
from transformers import Trainer

class MultiSampleDropoutHead(nn.Module):
    def __init__(self, in_features, num_labels, num_samples=5, dropout_prob=0.2):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_samples)])
        self.classifier = nn.Linear(in_features, num_labels)
        
    def forward(self, x):
        # Прогоняем через разные дропауты и усредняем логиты
        logits = torch.stack([self.classifier(drop(x)) for drop in self.dropouts], dim=0)
        return logits.mean(dim=0)

def apply_multi_sample_dropout(model, num_samples=5, dropout_prob=0.2):
    old_classifier = model.classifier
    in_features = old_classifier.in_features
    num_labels = old_classifier.out_features
    model.classifier = MultiSampleDropoutHead(in_features, num_labels, num_samples, dropout_prob)
    return model

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        num_labels = model.config.num_labels
        weights = torch.ones(num_labels).to(logits.device)
        # Увеличиваем вес сущностей (всех, кроме индекса 0 - 'O')
        weights[1:] = 2.5 
        
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss