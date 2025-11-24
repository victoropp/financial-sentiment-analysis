"""
Data loader for Financial Phrase Bank dataset.
Handles loading, preprocessing, and splitting of financial sentiment data.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import os

class FinancialDataLoader:
    """Load and preprocess Financial Phrase Bank dataset."""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            # Default to datasets directory
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            self.data_path = os.path.join(base, 'datasets', 'nlp', 'financial_sentiment', 'all-data.csv')
        else:
            self.data_path = data_path
    
    def load_data(self) -> pd.DataFrame:
        """Load the financial sentiment dataset."""
        print(f"Loading data from: {self.data_path}")
        
        # Load with proper encoding
        df = pd.read_csv(self.data_path, encoding='latin-1', header=None, names=['sentiment', 'text'])
        
        print(f"Loaded {len(df)} samples")
        print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-]', '', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset."""
        df = df.copy()
        
        # Clean text
        print("Cleaning text...")
        df['text'] = df['text'].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['text'].str.len() > 0]
        
        # Encode labels
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['label'] = df['sentiment'].map(label_map)
        
        # Add text length feature
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        print(f"After preprocessing: {len(df)} samples")
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        
        # First split: train+val and test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label']
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, 
            test_size=val_size_adjusted, 
            random_state=random_state, 
            stratify=train_val['label']
        )
        
        print(f"\nData split:")
        print(f"Train: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
        print(f"Val:   {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
        print(f"Test:  {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
        
        return train, val, test
    
    def get_class_weights(self, df: pd.DataFrame) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(df['label'])
        weights = compute_class_weight('balanced', classes=classes, y=df['label'])
        
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        
        print(f"\nClass weights: {class_weights}")
        
        return class_weights
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(df),
            'avg_text_length': df['text_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'label_distribution': df['label'].value_counts().to_dict()
        }
        
        return stats

if __name__ == "__main__":
    # Test the data loader
    loader = FinancialDataLoader()
    
    # Load data
    df = loader.load_data()
    
    # Preprocess
    df = loader.preprocess(df)
    
    # Split data
    train, val, test = loader.split_data(df)
    
    # Get statistics
    stats = loader.get_statistics(df)
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Get class weights
    class_weights = loader.get_class_weights(train)
