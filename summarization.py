"""
Text Summarization Module
Implements both abstractive (T5) and extractive (TF-IDF) summarization
"""

import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import os
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class AbstractiveSummarizer:
    """T5-based abstractive summarization"""
    
    def __init__(self, model_name='t5-small'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading T5 model on {self.device}...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("T5 model loaded successfully")
    
    def summarize(self, text, max_length=150, min_length=40, length_penalty=2.0):
        """Generate abstractive summary"""
        # Prepare input
        input_text = "summarize: " + text
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=4,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def batch_summarize(self, texts, max_length=150):
        """Summarize multiple texts"""
        summaries = []
        for text in texts:
            try:
                summary = self.summarize(text, max_length=max_length)
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing text: {e}")
                summaries.append("Summary generation failed")
        return summaries


class ExtractiveSummarizer:
    """TF-IDF and cosine similarity based extractive summarization"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def summarize(self, text, num_sentences=3):
        """Generate extractive summary"""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Convert to numpy array (fix for the error)
            tfidf_array = tfidf_matrix.toarray()
            
            # Calculate document centroid (mean of all sentence vectors)
            document_vector = np.mean(tfidf_array, axis=0).reshape(1, -1)
            
            # Calculate sentence scores based on similarity to document centroid
            sentence_scores = cosine_similarity(tfidf_array, document_vector).flatten()
            
        except Exception as e:
            print(f"Error in extractive summarization: {e}")
            return ' '.join(sentences[:num_sentences])
        
        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices_sorted = sorted(top_indices)
        
        # Maintain original order
        summary_sentences = [sentences[i] for i in top_indices_sorted]
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def batch_summarize(self, texts, num_sentences=3):
        """Summarize multiple texts"""
        summaries = []
        for text in texts:
            try:
                summary = self.summarize(text, num_sentences=num_sentences)
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing text: {e}")
                summaries.append(text[:200] + "...")
        return summaries


class FeedbackSummarizer:
    """Combined summarization system for customer feedback"""
    
    def __init__(self, use_abstractive=True, use_extractive=True):
        self.use_abstractive = use_abstractive
        self.use_extractive = use_extractive
        
        if use_abstractive:
            self.abstractive = AbstractiveSummarizer()
        if use_extractive:
            self.extractive = ExtractiveSummarizer()
    
    def summarize_feedback(self, text, summary_type='both'):
        """
        Summarize feedback text
        
        Args:
            text: Input text
            summary_type: 'abstractive', 'extractive', or 'both'
        """
        results = {}
        
        if summary_type in ['abstractive', 'both'] and self.use_abstractive:
            try:
                # Short summary
                results['abstractive_short'] = self.abstractive.summarize(
                    text, max_length=50, min_length=20
                )
                # Detailed summary
                results['abstractive_detailed'] = self.abstractive.summarize(
                    text, max_length=150, min_length=50
                )
            except Exception as e:
                print(f"Error in abstractive summarization: {e}")
                results['abstractive_short'] = "Summary generation failed"
                results['abstractive_detailed'] = "Summary generation failed"
        
        if summary_type in ['extractive', 'both'] and self.use_extractive:
            try:
                # Short summary (2 sentences)
                results['extractive_short'] = self.extractive.summarize(text, num_sentences=2)
                # Detailed summary (4 sentences)
                results['extractive_detailed'] = self.extractive.summarize(text, num_sentences=4)
            except Exception as e:
                print(f"Error in extractive summarization: {e}")
                results['extractive_short'] = text[:200] + "..."
                results['extractive_detailed'] = text[:400] + "..."
        
        return results
    
    def summarize_by_sentiment(self, df, sentiment='positive', sample_size=50):
        """Generate summaries for feedback grouped by sentiment"""
        sentiment_df = df[df['sentiment'] == sentiment].head(sample_size)
        
        if len(sentiment_df) == 0:
            print(f"No {sentiment} feedback found")
            return {}
        
        print(f"\nSummarizing {sentiment} feedback ({len(sentiment_df)} samples)...")
        
        # Combine multiple feedbacks into one text
        combined_text = '. '.join(sentiment_df['feedback_text'].astype(str).tolist())
        
        # Limit text length for T5 (max 512 tokens)
        if len(combined_text) > 2000:
            combined_text = combined_text[:2000]
        
        summaries = self.summarize_feedback(combined_text)
        
        return summaries
    
    def summarize_by_category(self, df, category_col='product', sample_size=30):
        """Generate summaries for each product/category"""
        results = {}
        
        categories = df[category_col].unique()
        
        for category in categories[:5]:  # Limit to top 5 categories
            print(f"\nSummarizing feedback for: {category}")
            category_df = df[df[category_col] == category].head(sample_size)
            
            if len(category_df) == 0:
                continue
            
            combined_text = '. '.join(category_df['feedback_text'].astype(str).tolist())
            
            # Limit text length
            if len(combined_text) > 2000:
                combined_text = combined_text[:2000]
            
            summaries = self.summarize_feedback(combined_text)
            results[category] = summaries
        
        return results


def generate_summary_report(df, summarizer):
    """Generate comprehensive summary report"""
    report = {
        'overall_summary': {},
        'sentiment_summaries': {},
        'product_summaries': {}
    }
    
    print("=" * 60)
    print("GENERATING FEEDBACK SUMMARIES")
    print("=" * 60)
    
    # Overall summary
    print("\n1. Overall Feedback Summary")
    print("-" * 60)
    all_feedback = '. '.join(df['feedback_text'].astype(str).head(100).tolist())
    
    # Limit text length
    if len(all_feedback) > 2000:
        all_feedback = all_feedback[:2000]
    
    overall = summarizer.summarize_feedback(all_feedback)
    report['overall_summary'] = overall
    
    print("\nAbstractive Summary (Short):")
    print(overall.get('abstractive_short', 'N/A'))
    print("\nAbstractive Summary (Detailed):")
    print(overall.get('abstractive_detailed', 'N/A'))
    
    # Sentiment-based summaries
    print("\n\n2. Sentiment-Based Summaries")
    print("-" * 60)
    
    available_sentiments = df['sentiment'].unique()
    
    for sentiment in available_sentiments:
        print(f"\n{sentiment.upper()} Feedback:")
        summaries = summarizer.summarize_by_sentiment(df, sentiment, sample_size=50)
        report['sentiment_summaries'][sentiment] = summaries
        
        if summaries:
            print(f"\nShort Summary:")
            print(summaries.get('abstractive_short', 'N/A'))
    
    # Product-based summaries
    print("\n\n3. Product-Based Summaries")
    print("-" * 60)
    
    if 'product' in df.columns:
        product_summaries = summarizer.summarize_by_category(df, 'product', sample_size=30)
        report['product_summaries'] = product_summaries
        
        # Display top 3 products
        top_products = df['product'].value_counts().head(3).index.tolist()
        for product in top_products:
            if product in product_summaries:
                print(f"\n{product.upper()}:")
                print(product_summaries[product].get('abstractive_short', 'N/A'))
    
    return report


def save_summaries(report, output_file='models/feedback_summaries.txt'):
    """Save summaries to file"""
    os.makedirs('models', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CUSTOMER FEEDBACK SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall Summary
        f.write("1. OVERALL SUMMARY\n")
        f.write("-" * 60 + "\n")
        overall = report['overall_summary']
        f.write(f"\nShort Summary:\n{overall.get('abstractive_short', 'N/A')}\n")
        f.write(f"\nDetailed Summary:\n{overall.get('abstractive_detailed', 'N/A')}\n")
        
        # Sentiment Summaries
        f.write("\n\n2. SENTIMENT-BASED SUMMARIES\n")
        f.write("-" * 60 + "\n")
        for sentiment, summaries in report['sentiment_summaries'].items():
            f.write(f"\n{sentiment.upper()} Feedback:\n")
            f.write(f"Short: {summaries.get('abstractive_short', 'N/A')}\n")
            f.write(f"Detailed: {summaries.get('abstractive_detailed', 'N/A')}\n")
        
        # Product Summaries
        if report['product_summaries']:
            f.write("\n\n3. PRODUCT-BASED SUMMARIES\n")
            f.write("-" * 60 + "\n")
            for product, summaries in report['product_summaries'].items():
                f.write(f"\n{product.upper()}:\n")
                f.write(f"Short: {summaries.get('abstractive_short', 'N/A')}\n")
    
    print(f"\n\nSummaries saved to {output_file}")


def demonstrate_summarization():
    """Demonstrate summarization with examples"""
    print("\n" + "=" * 60)
    print("SUMMARIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Example long feedback
    example_feedback = """
    I recently purchased the new laptop from your company and I have mixed feelings about it. 
    On the positive side, the design is sleek and modern, the screen quality is excellent with 
    vibrant colors, and the keyboard is comfortable for long typing sessions. The battery life 
    is impressive, lasting almost 10 hours on a single charge. However, I've encountered several 
    issues. The laptop tends to overheat when running multiple applications, and the fan noise 
    becomes quite loud and distracting. The trackpad is sometimes unresponsive, requiring multiple 
    attempts to register clicks. Additionally, the pre-installed software is bloated with 
    unnecessary applications that slow down the system. The customer service was helpful when 
    I contacted them, but the wait time was quite long. Overall, it's a decent laptop with 
    good hardware, but the software experience and heating issues need improvement. For the 
    price point, I expected better quality control and performance optimization.
    """
    
    summarizer = FeedbackSummarizer()
    
    print("\nOriginal Feedback:")
    print(example_feedback)
    
    summaries = summarizer.summarize_feedback(example_feedback)
    
    print("\n\nABSTRACTIVE SUMMARIES:")
    print("-" * 60)
    print(f"\nShort Summary:\n{summaries.get('abstractive_short', 'N/A')}")
    print(f"\nDetailed Summary:\n{summaries.get('abstractive_detailed', 'N/A')}")
    
    print("\n\nEXTRACTIVE SUMMARIES:")
    print("-" * 60)
    print(f"\nShort Summary:\n{summaries.get('extractive_short', 'N/A')}")
    print(f"\nDetailed Summary:\n{summaries.get('extractive_detailed', 'N/A')}")


def main():
    """Main execution function"""
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
    
    # Initialize summarizer
    summarizer = FeedbackSummarizer(use_abstractive=True, use_extractive=True)
    
    # Demonstrate summarization
    demonstrate_summarization()
    
    # Generate comprehensive report
    report = generate_summary_report(df, summarizer)
    
    # Save summaries
    save_summaries(report)
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print("✓ Summaries generated successfully")
    print("✓ Summaries saved to: models/feedback_summaries.txt")
    print("\nNext step: Run insights generation")
    print("  python insights.py")
    
    return summarizer, report


if __name__ == "__main__":
    summarizer, report = main()