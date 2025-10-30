"""
Sentiment Classification Model using BERT
Trains and evaluates a transformer-based sentiment classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import joblib
import json
import warnings
import os
warnings.filterwarnings('ignore')


class FeedbackDataset(Dataset):
    """Custom Dataset for feedback data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
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


class SentimentClassifier:
    """BERT-based Sentiment Classification Model"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # For IMDB: positive=1, negative=0
        self.label_map = {'negative': 0, 'positive': 1}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        """Prepare train, validation, and test datasets"""
        # Encode labels
        df['label_encoded'] = df['sentiment'].map(self.label_map)
        
        # Remove any rows with unmapped labels
        df = df.dropna(subset=['label_encoded'])
        df['label_encoded'] = df['label_encoded'].astype(int)
        
        # Split data
        X = df['processed_text'].values
        y = df['label_encoded'].values
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
        )
        
        print(f"Train size: {len(X_train)}")
        print(f"Validation size: {len(X_val)}")
        print(f"Test size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16):
        """Create PyTorch DataLoaders"""
        train_dataset = FeedbackDataset(X_train, y_train, self.tokenizer)
        val_dataset = FeedbackDataset(X_val, y_val, self.tokenizer)
        test_dataset = FeedbackDataset(X_test, y_test, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """Train the model"""
        # Use torch.optim.AdamW instead of transformers AdamW
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc="Training")
            
            for batch in train_pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            avg_val_loss, val_accuracy = self.evaluate(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return train_losses, val_losses
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def get_predictions(self, data_loader):
        """Get predictions and true labels"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Getting predictions"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels)
    
    def predict_text(self, text):
        """Predict sentiment for a single text"""
        self.model.eval()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
        
        return self.reverse_label_map[pred]
    
    def save_model(self, path='models/'):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path + 'sentiment_model')
        self.tokenizer.save_pretrained(path + 'sentiment_model')
        
        # Save label maps
        joblib.dump({
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map
        }, path + 'label_maps.pkl')
        
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/'):
        """Load saved model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            path + 'sentiment_model'
        ).to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(path + 'sentiment_model')
        
        label_maps = joblib.load(path + 'label_maps.pkl')
        self.label_map = label_maps['label_map']
        self.reverse_label_map = label_maps['reverse_label_map']
        
        print(f"Model loaded from {path}")


def plot_training_history(train_losses, val_losses):
    """Plot training history"""
    os.makedirs('models', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved")


def evaluate_model_performance(y_true, y_pred, label_map):
    """Evaluate and visualize model performance"""
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    target_names = [reverse_label_map[i] for i in sorted(reverse_label_map.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    print(f"\nOverall Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Save metrics
    os.makedirs('models', exist_ok=True)
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved")
    
    return metrics


def main():
    """Main training pipeline"""
    # Load processed data
    print("Loading processed data...")
    
    try:
        df = pd.read_csv('data/processed/customer_feedback_processed.csv')
    except FileNotFoundError:
        print("\n" + "=" * 50)
        print("ERROR: Processed data not found!")
        print("=" * 50)
        print("\nPlease run data preprocessing first:")
        print("  python data_preprocessing.py")
        return None, None
    
    # Initialize classifier (2 labels for positive/negative)
    classifier = SentimentClassifier(num_labels=2)
    
    # Prepare data
    print("\nPreparing data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(df)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = classifier.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16
    )
    
    # Train model
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)
    train_losses, val_losses = classifier.train(train_loader, val_loader, epochs=3)
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)
    y_pred, y_true = classifier.get_predictions(test_loader)
    metrics = evaluate_model_performance(y_true, y_pred, classifier.label_map)
    
    # Save model
    classifier.save_model()
    
    # Test predictions
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    test_texts = [
        "This movie is absolutely amazing! Best film I've ever seen.",
        "Terrible movie. Complete waste of time and money.",
        "The acting was superb and the story kept me engaged throughout."
    ]
    
    for text in test_texts:
        prediction = classifier.predict_text(text)
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {prediction}")
    
    print("\n" + "=" * 50)
    print("SUCCESS!")
    print("=" * 50)
    print("✓ Model trained successfully")
    print(f"✓ Accuracy: {metrics['accuracy']:.2%}")
    print("✓ Model saved to: models/sentiment_model/")
    print("\nNext step: Run summarization")
    print("  python summarization.py")
    
    return classifier, metrics


if __name__ == "__main__":
    classifier, metrics = main()