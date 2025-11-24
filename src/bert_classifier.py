"""
Simplified transformer models using sentence-transformers for embeddings + classifier.
More lightweight and deployment-friendly approach.
"""

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import joblib
import json
import os
from typing import Dict

class FinancialBERTClassifier:
    """BERT-based classifier using sentence embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a sentence transformer model.
        all-MiniLM-L6-v2: Fast, lightweight, good performance
        paraphrase-mpnet-base-v2: Higher quality, slower
        """
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.classifier = None
        
    def encode_texts(self, texts):
        """Encode texts to embeddings."""
        return self.encoder.encode(texts, show_progress_bar=True)
    
    def train(self, train_texts, train_labels, class_weights=None):
        """Train classifier on BERT embeddings."""
        print(f"\n{'='*50}")
        print(f"Training BERT Classifier ({self.model_name})")
        print(f"{'='*50}")
        
        # Encode texts
        print("Encoding training texts...")
        X_train = self.encode_texts(train_texts)
        
        # Train classifier
        print("Training logistic regression classifier...")
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced' if class_weights is None else class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, train_labels)
        
        print("Training complete!")
        
        return self
    
    def evaluate(self, test_texts, test_labels, model_name: str = "BERT") -> Dict:
        """Evaluate model on test set."""
        print(f"\nEvaluating {model_name}...")
        
        # Encode texts
        X_test = self.encode_texts(test_texts)
        
        # Predict
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, y_pred, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            test_labels, y_pred, average=None
        )
        
        cm = confusion_matrix(test_labels, y_pred)
        
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
    
    def predict(self, texts):
        """Predict sentiment for texts."""
        X = self.encode_texts(texts)
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """Get prediction probabilities."""
        X = self.encode_texts(texts)
        return self.classifier.predict_proba(X)
    
    def save(self, save_dir: str):
        """Save the model."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save classifier
        classifier_path = os.path.join(save_dir, 'classifier.pkl')
        joblib.dump(self.classifier, classifier_path)
        
        # Save model info
        info = {'encoder_model': self.model_name}
        info_path = os.path.join(save_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f)
        
        print(f"\nModel saved to: {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str):
        """Load a saved model."""
        # Load info
        info_path = os.path.join(load_dir, 'model_info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Create instance
        instance = cls(model_name=info['encoder_model'])
        
        # Load classifier
        classifier_path = os.path.join(load_dir, 'classifier.pkl')
        instance.classifier = joblib.load(classifier_path)
        
        return instance

def save_metrics(metrics: Dict, save_path: str):
    """Save metrics to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {save_path}")

if __name__ == "__main__":
    from data_loader import FinancialDataLoader
    
    # Load data
    print("Loading dataset...")
    loader = FinancialDataLoader()
    df = loader.load_data()
    df = loader.preprocess(df)
    train, val, test = loader.split_data(df)
    class_weights = loader.get_class_weights(train)
    
    # Prepare data
    train_texts = train['text'].tolist()
    train_labels = train['label'].tolist()
    test_texts = test['text'].tolist()
    test_labels = test['label'].tolist()
    
    # Base directory for models
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    # Train BERT model (lightweight)
    print("\n" + "="*70)
    print("TRAINING BERT-BASED CLASSIFIER")
    print("="*70)
    
    bert_model = FinancialBERTClassifier(model_name='all-MiniLM-L6-v2')
    bert_model.train(train_texts, train_labels, class_weights)
    bert_metrics = bert_model.evaluate(test_texts, test_labels, "BERT (Sentence Embeddings)")
    
    # Save BERT model
    bert_output_dir = os.path.join(base_dir, 'bert_embeddings')
    bert_model.save(bert_output_dir)
    save_metrics(bert_metrics, os.path.join(bert_output_dir, 'bert_metrics.json'))
    
    # Train with better encoder (if time permits)
    print("\n" + "="*70)
    print("TRAINING ENHANCED BERT CLASSIFIER")
    print("="*70)
    
    bert_large = FinancialBERTClassifier(model_name='paraphrase-mpnet-base-v2')
    bert_large.train(train_texts, train_labels, class_weights)
    bert_large_metrics = bert_large.evaluate(test_texts, test_labels, "BERT (MPNet)")
    
    # Save enhanced model
    bert_large_dir = os.path.join(base_dir, 'bert_mpnet')
    bert_large.save(bert_large_dir)
    save_metrics(bert_large_metrics, os.path.join(bert_large_dir, 'bert_mpnet_metrics.json'))
    
    print("\n" + "="*70)
    print("TRANSFORMER TRAINING COMPLETE!")
    print("="*70)
    print("\nModel Summary:")
    print(f"BERT (MiniLM):  F1={bert_metrics['f1_score']:.4f}")
    print(f"BERT (MPNet):   F1={bert_large_metrics['f1_score']:.4f}")
