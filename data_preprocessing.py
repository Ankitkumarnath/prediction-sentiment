"""
Data Preprocessing Module for Customer Feedback Analysis
Handles data cleaning and preprocessing using IMDB Dataset
"""

import pandas as pd
import numpy as np
import re
import string
from datetime import datetime, timedelta
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import os

# Download required NLTK data with error handling
def download_nltk_data():
    """Download NLTK data with proper error handling"""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)

# Download at import
download_nltk_data()


class IMDBDataLoader:
    """Load and prepare IMDB dataset"""
    
    def __init__(self, filepath='data/raw/IMDB Dataset.csv'):
        self.filepath = filepath
        
    def load_data(self, sample_size=None):
        """Load IMDB dataset"""
        print(f"Loading data from {self.filepath}...")
        
        try:
            df = pd.read_csv(self.filepath)
            print(f"Loaded {len(df)} records")
            
            # Check required columns
            if 'review' not in df.columns or 'sentiment' not in df.columns:
                raise ValueError("Dataset must contain 'review' and 'sentiment' columns")
            
            # Rename columns for consistency
            df = df.rename(columns={'review': 'feedback_text'})
            
            # Sample if requested
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                print(f"Sampled {sample_size} records")
            
            # Add metadata
            df = self._add_metadata(df)
            
            return df
            
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            print("Please ensure 'IMDB Dataset.csv' is in the data/raw/ directory")
            raise
    
    def _add_metadata(self, df):
        """Add metadata columns for better analysis"""
        num_records = len(df)
        
        # Add feedback IDs
        df['feedback_id'] = [f"FB{i+1:05d}" for i in range(num_records)]
        
        # Add timestamps (simulate last 6 months)
        start_date = datetime.now() - timedelta(days=180)
        timestamps = [start_date + timedelta(days=random.randint(0, 180)) 
                     for _ in range(num_records)]
        df['timestamp'] = timestamps
        
        # Add sources
        sources = ['email', 'chat', 'twitter', 'facebook', 'survey', 'app_review']
        df['source'] = [random.choice(sources) for _ in range(num_records)]
        
        # Add customer IDs
        df['customer_id'] = [f"CUST{random.randint(1000, 9999)}" for _ in range(num_records)]
        
        # Add product categories (based on review content)
        products = ['movie', 'film', 'cinema', 'entertainment', 'streaming']
        df['product'] = [random.choice(products) for _ in range(num_records)]
        
        # Add ratings based on sentiment
        df['rating'] = df['sentiment'].apply(
            lambda x: random.randint(4, 5) if x.lower() == 'positive' else random.randint(1, 2)
        )
        
        # Normalize sentiment values
        df['sentiment'] = df['sentiment'].str.lower()
        
        return df


class FeedbackPreprocessor:
    """Clean and preprocess customer feedback text"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            print("Downloading required NLTK data...")
            download_nltk_data()
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        
    def remove_duplicates(self, df):
        """Remove duplicate records"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['feedback_text'], keep='first')
        removed = initial_count - len(df)
        print(f"Removed {removed} duplicate records")
        return df
    
    def handle_missing_data(self, df):
        """Handle missing values"""
        initial_count = len(df)
        # Remove rows with missing feedback text
        df = df.dropna(subset=['feedback_text'])
        # Fill missing ratings with median
        if 'rating' in df.columns:
            df['rating'].fillna(df['rating'].median(), inplace=True)
        removed = initial_count - len(df)
        print(f"Handled missing data: {removed} rows removed")
        return df
    
    def clean_text(self, text):
        """Clean individual text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        if not text or not isinstance(text, str):
            return []
        
        try:
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    try:
                        lemma = self.lemmatizer.lemmatize(token)
                        processed_tokens.append(lemma)
                    except Exception as e:
                        # If lemmatization fails, use original token
                        processed_tokens.append(token)
            
            return processed_tokens
            
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return []
    
    def preprocess_dataframe(self, df):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 2: Handle missing data
        df = self.handle_missing_data(df)
        
        # Step 3: Clean text
        print("Cleaning text...")
        df['cleaned_text'] = df['feedback_text'].apply(self.clean_text)
        
        # Step 4: Tokenize and lemmatize
        print("Tokenizing and lemmatizing...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x) if x else '')
        
        # Remove rows with empty processed text
        df = df[df['processed_text'].str.len() > 0]
        
        # Ensure we have at least 1000 records for the assignment
        if len(df) < 1000:
            print(f"Warning: Only {len(df)} records available after preprocessing")
        
        print(f"Preprocessing complete. Final dataset: {len(df)} records")
        
        return df
    
    def save_processed_data(self, df):
        """Save processed data"""
        # Create directory if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)
        
        df.to_csv('data/processed/customer_feedback_processed.csv', index=False)
        print("Processed data saved to data/processed/customer_feedback_processed.csv")
        
        # Save summary statistics
        summary = {
            "total_records": int(len(df)),
            "sentiment_distribution": df['sentiment'].value_counts().to_dict(),
            "source_distribution": df['source'].value_counts().to_dict(),
            "avg_rating": float(df['rating'].mean()) if 'rating' in df.columns else None,
            "date_range": {
                "start": str(df['timestamp'].min()),
                "end": str(df['timestamp'].max())
            }
        }
        
        with open('data/processed/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("Summary statistics saved")


def main():
    """Main execution function"""
    import os
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("=" * 50)
    print("LOADING IMDB DATASET")
    print("=" * 50)
    
    # Load IMDB data
    loader = IMDBDataLoader('data/raw/IMDB Dataset.csv')
    
    try:
        # Load data (use 5000 samples for faster processing, or None for all)
        df_raw = loader.load_data(sample_size=5000)
        
        print(f"\nLoaded {len(df_raw)} records")
        print(f"Sentiment distribution:\n{df_raw['sentiment'].value_counts()}")
        
        # Save raw data with metadata
        df_raw.to_csv('data/raw/customer_feedback_raw.csv', index=False)
        print("\nRaw data with metadata saved")
        
    except FileNotFoundError:
        print("\n" + "=" * 50)
        print("ERROR: IMDB Dataset not found!")
        print("=" * 50)
        print("\nPlease ensure 'IMDB Dataset.csv' is located at:")
        print("  data/raw/IMDB Dataset.csv")
        print("\nThe file should contain columns: 'review' and 'sentiment'")
        return None
    
    # Preprocess data
    print("\n" + "=" * 50)
    print("PREPROCESSING DATA")
    print("=" * 50)
    preprocessor = FeedbackPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df_raw)
    preprocessor.save_processed_data(df_processed)
    
    # Display sample
    print("\n" + "=" * 50)
    print("SAMPLE PROCESSED DATA")
    print("=" * 50)
    print(df_processed[['feedback_id', 'feedback_text', 'processed_text', 'sentiment']].head())
    
    print("\n" + "=" * 50)
    print("SUCCESS!")
    print("=" * 50)
    print(f"✓ Processed {len(df_processed)} records")
    print("✓ Data saved to: data/processed/customer_feedback_processed.csv")
    print("\nNext step: Run sentiment model training")
    print("  python sentiment_model.py")
    
    return df_processed


if __name__ == "__main__":
    df = main()