"""
Traditional ML models for financial sentiment analysis.
Includes Logistic Regression, Random Forest, XGBoost, and SVM.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib
import os
import json
from typing import Tuple, Dict

class TraditionalModels:
    """Traditional ML models for sentiment classification."""
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer = None
        self.models = {}
        
    def create_tfidf_features(self, train_texts, val_texts=None, test_texts=None):
        """Create TF-IDF features."""
        print("Creating TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 3),  # unigrams, bigrams, trigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        X_train = self.vectorizer.fit_transform(train_texts)
        print(f"TF-IDF shape: {X_train.shape}")
        
        results = {'train': X_train}
        
        if val_texts is not None:
            X_val = self.vectorizer.transform(val_texts)
            results['val'] = X_val
            
        if test_texts is not None:
            X_test = self.vectorizer.transform(test_texts)
            results['test'] = X_test
            
        return results
    
    def train_logistic_regression(self, X_train, y_train, class_weights=None):
        """Train Logistic Regression model."""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced' if class_weights is None else class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train, class_weights=None):
        """Train Random Forest model."""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced' if class_weights is None else class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train, class_weights=None):
        """Train XGBoost model."""
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        # Convert class weights to sample weights
        if class_weights:
            sample_weights = np.array([class_weights[label] for label in y_train])
        else:
            sample_weights = None
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        self.models['xgboost'] = model
        
        return model
    
    def train_svm(self, X_train, y_train, class_weights=None):
        """Train SVM model."""
        print("\n" + "="*50)
        print("Training SVM...")
        print("="*50)
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced' if class_weights is None else class_weights,
            random_state=42,
            probability=True  # Enable probability estimates
        )
        
        model.fit(X_train, y_train)
        self.models['svm'] = model
        
        return model
    
    def evaluate_model(self, model, X, y, model_name: str) -> Dict:
        """Evaluate a model and return metrics."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y, y_pred, average=None
        )
        
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
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        return metrics
    
    def save_models(self, save_dir: str):
        """Save all trained models and vectorizer."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vectorizer
        vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"\nSaved vectorizer: {vectorizer_path}")
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"Saved model: {model_path}")
    
    def save_metrics(self, metrics: Dict, save_path: str):
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
    class_weights = loader.get_class_weights(train)
    
    # Initialize traditional models
    trad_models = TraditionalModels(max_features=5000)
    
    # Create TF-IDF features
    features = trad_models.create_tfidf_features(
        train['text'].values,
        val['text'].values,
        test['text'].values
    )
    
    X_train, X_val, X_test = features['train'], features['val'], features['test']
    y_train, y_val, y_test = train['label'].values, val['label'].values, test['label'].values
    
    # Train all models
    all_metrics = {}
    
    # 1. Logistic Regression
    lr_model = trad_models.train_logistic_regression(X_train, y_train, class_weights)
    lr_metrics = trad_models.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    all_metrics['logistic_regression'] = lr_metrics
    
    # 2. Random Forest
    rf_model = trad_models.train_random_forest(X_train, y_train, class_weights)
    rf_metrics = trad_models.evaluate_model(rf_model, X_test, y_test, "Random Forest")
    all_metrics['random_forest'] = rf_metrics
    
    # 3. XGBoost
    xgb_model = trad_models.train_xgboost(X_train, y_train, class_weights)
    xgb_metrics = trad_models.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    all_metrics['xgboost'] = xgb_metrics
    
    # 4. SVM
    svm_model = trad_models.train_svm(X_train, y_train, class_weights)
    svm_metrics = trad_models.evaluate_model(svm_model, X_test, y_test, "SVM")
    all_metrics['svm'] = svm_metrics
    
    # Save models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'traditional')
    trad_models.save_models(models_dir)
    
    # Save metrics
    metrics_path = os.path.join(models_dir, 'traditional_models_metrics.json')
    trad_models.save_metrics(all_metrics, metrics_path)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
