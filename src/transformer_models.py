"""
Transformer-based models for financial sentiment analysis.
Includes BERT, FinBERT fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from typing import Dict, List
import os
import json

class SentimentDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for sentiment classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerModels:
    """Transformer models for sentiment classification."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load pre-trained model and tokenizer."""
        print(f"\nLoading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        self.model.to(self.device)
        
        print(f"Model loaded successfully!")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_datasets(self, train_texts, train_labels, val_texts, val_labels, 
                       test_texts, test_labels, max_length: int = 128):
        """Create PyTorch datasets."""
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, max_length)
        test_dataset = SentimentDataset(test_texts, test_labels, self.tokenizer, max_length)
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, pred):
        """Compute metrics for evaluation."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_dataset, val_dataset, output_dir: str, 
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Fine-tune the model."""
        print(f"\n{'='*50}")
        print(f"Training {self.model_name}")
        print(f"{'='*50}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\nModel saved to: {output_dir}")
        
        return trainer
    
    def evaluate(self, trainer, test_dataset, model_name: str) -> Dict:
        """Evaluate model on test set."""
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        
        y_true = predictions.label_ids
        y_pred = predictions.predictions.argmax(-1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class': {
                'negative': {
                    'precision': float(precision_per_class[0]),
                    'recall': float(recall_per_class[0]),
                    'f1': float(f1_per_class[0]),
                    'support': int(support[0])
                },
                'neutral': {
                    'precision': float(precision_per_class[1]),
                    'recall': float(recall_per_class[1]),
                    'f1': float(f1_per_class[1]),
                    'support': int(support[1])
                },
                'positive': {
                    'precision': float(precision_per_class[2]),
                    'recall': float(recall_per_class[2]),
                    'f1': float(f1_per_class[2]),
                    'support': int(support[2])
                }
            },
            'confusion_matrix': cm.tolist()
        }
        
        print(f"\n{model_name} Test Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        return metrics
    
    def predict(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Predict sentiment for new texts."""
        self.model.eval()
        
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = self.model(**encodings)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)

def save_metrics(metrics: Dict, save_path: str):
    """Save metrics to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {save_path}")

if __name__ == "__main__":
    from data_loader import FinancialDataLoader
    
    # Load data
    loader = FinancialDataLoader()
    df = loader.load_data()
    df = loader.preprocess(df)
    train, val, test = loader.split_data(df)
    
    # Prepare data
    train_texts = train['text'].tolist()
    train_labels = train['label'].tolist()
    val_texts = val['text'].tolist()
    val_labels = val['label'].tolist()
    test_texts = test['text'].tolist()
    test_labels = test['label'].tolist()
    
    # Base directory for models
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    # Train BERT
    print("\n" + "="*70)
    print("TRAINING BERT MODEL")
    print("="*70)
    
    bert_model = TransformerModels(model_name='bert-base-uncased', num_labels=3)
    bert_model.load_model()
    
    train_dataset, val_dataset, test_dataset = bert_model.create_datasets(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    )
    
    bert_output_dir = os.path.join(base_dir, 'bert')
    bert_trainer = bert_model.train(train_dataset, val_dataset, bert_output_dir, epochs=3)
    bert_metrics = bert_model.evaluate(bert_trainer, test_dataset, "BERT")
    
    save_metrics(bert_metrics, os.path.join(bert_output_dir, 'bert_metrics.json'))
    
    # Train FinBERT
    print("\n" + "="*70)
    print("TRAINING FINBERT MODEL")
    print("="*70)
    
    finbert_model = TransformerModels(model_name='ProsusAI/finbert', num_labels=3)
    finbert_model.load_model()
    
    train_dataset, val_dataset, test_dataset = finbert_model.create_datasets(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    )
    
    finbert_output_dir = os.path.join(base_dir, 'finbert')
    finbert_trainer = finbert_model.train(train_dataset, val_dataset, finbert_output_dir, epochs=3)
    finbert_metrics = finbert_model.evaluate(finbert_trainer, test_dataset, "FinBERT")
    
    save_metrics(finbert_metrics, os.path.join(finbert_output_dir, 'finbert_metrics.json'))
    
    print("\n" + "="*70)
    print("ALL TRANSFORMER MODELS TRAINED!")
    print("="*70)
