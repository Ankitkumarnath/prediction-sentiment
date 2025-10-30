"""
Predictive Insights Generation Module
Analyzes feedback patterns and forecasts customer satisfaction trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# For time series forecasting
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# For word analysis
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FeedbackInsightsAnalyzer:
    """Analyze feedback patterns and generate insights"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['date'] = self.df['timestamp'].dt.date
        self.df['month'] = self.df['timestamp'].dt.to_period('M')
        
    def analyze_sentiment_trends(self):
        """Analyze sentiment distribution and trends over time"""
        print("Analyzing sentiment trends...")
        
        # Overall sentiment distribution
        sentiment_counts = self.df['sentiment'].value_counts()
        sentiment_pct = (sentiment_counts / len(self.df) * 100).round(2)
        
        # Sentiment over time
        sentiment_timeline = self.df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
        
        insights = {
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentage': sentiment_pct.to_dict(),
            'sentiment_timeline': sentiment_timeline
        }
        
        return insights
    
    def identify_recurring_issues(self, top_n=10):
        """Identify most common issues in negative feedback"""
        print("Identifying recurring issues...")
        
        negative_feedback = self.df[self.df['sentiment'] == 'negative']
        
        # Extract common words from negative feedback
        all_words = []
        for text in negative_feedback['processed_text']:
            words = str(text).split()
            all_words.extend(words)
        
        # Count word frequency
        word_freq = Counter(all_words)
        common_issues = word_freq.most_common(top_n)
        
        # Product-specific issues
        product_issues = negative_feedback.groupby('product').size().sort_values(ascending=False)
        
        # Source of negative feedback
        source_negative = negative_feedback['source'].value_counts()
        
        insights = {
            'common_keywords': common_issues,
            'problematic_products': product_issues.head(5).to_dict(),
            'negative_sources': source_negative.to_dict()
        }
        
        return insights
    
    def analyze_rating_trends(self):
        """Analyze rating patterns"""
        print("Analyzing rating trends...")
        
        # Average rating over time
        rating_timeline = self.df.groupby('month')['rating'].agg(['mean', 'count']).round(2)
        
        # Rating by product
        product_ratings = self.df.groupby('product')['rating'].mean().sort_values(ascending=False)
        
        # Rating by source
        source_ratings = self.df.groupby('source')['rating'].mean().sort_values(ascending=False)
        
        insights = {
            'rating_timeline': rating_timeline,
            'product_ratings': product_ratings.to_dict(),
            'source_ratings': source_ratings.to_dict(),
            'overall_avg_rating': self.df['rating'].mean()
        }
        
        return insights
    
    def analyze_customer_behavior(self):
        """Analyze customer feedback patterns"""
        print("Analyzing customer behavior...")
        
        # Feedback frequency by customer
        customer_feedback_count = self.df.groupby('customer_id').size()
        
        # Repeat customers
        repeat_customers = (customer_feedback_count > 1).sum()
        total_customers = len(customer_feedback_count)
        
        # Average sentiment by customer type
        customers_with_multiple = customer_feedback_count[customer_feedback_count > 1].index
        self.df['is_repeat'] = self.df['customer_id'].isin(customers_with_multiple)
        
        sentiment_by_type = self.df.groupby(['is_repeat', 'sentiment']).size().unstack(fill_value=0)
        
        insights = {
            'total_customers': total_customers,
            'repeat_customers': repeat_customers,
            'repeat_rate': round(repeat_customers / total_customers * 100, 2),
            'sentiment_by_customer_type': sentiment_by_type
        }
        
        return insights


class SatisfactionForecaster:
    """Forecast customer satisfaction trends"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['month'] = self.df['timestamp'].dt.to_period('M')
        
    def prepare_time_series(self):
        """Prepare time series data for forecasting"""
        # Calculate average satisfaction score per month
        monthly_satisfaction = self.df.groupby('month')['rating'].mean().reset_index()
        monthly_satisfaction['month'] = monthly_satisfaction['month'].dt.to_timestamp()
        monthly_satisfaction = monthly_satisfaction.sort_values('month')
        
        return monthly_satisfaction
    
    def forecast_satisfaction(self, periods=3):
        """Forecast satisfaction for next periods"""
        print(f"Forecasting satisfaction for next {periods} months...")
        
        monthly_data = self.prepare_time_series()
        
        # Simple moving average forecast
        ma_window = 3
        monthly_data['ma_forecast'] = monthly_data['rating'].rolling(window=ma_window).mean()
        
        # Linear regression forecast
        monthly_data['month_num'] = range(len(monthly_data))
        X = monthly_data['month_num'].values.reshape(-1, 1)
        y = monthly_data['rating'].values
        
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # Forecast future months
        last_month_num = monthly_data['month_num'].iloc[-1]
        future_months = np.arange(last_month_num + 1, last_month_num + periods + 1).reshape(-1, 1)
        lr_forecast = lr_model.predict(future_months)
        
        # Create forecast dataframe
        last_date = monthly_data['month'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
        
        forecast_df = pd.DataFrame({
            'month': forecast_dates,
            'forecast_rating': lr_forecast,
            'trend': 'increasing' if lr_model.coef_[0] > 0 else 'decreasing'
        })
        
        insights = {
            'historical_data': monthly_data,
            'forecast': forecast_df,
            'trend_coefficient': float(lr_model.coef_[0]),
            'current_avg': float(monthly_data['rating'].iloc[-1]),
            'forecast_avg': float(lr_forecast.mean())
        }
        
        return insights
    
    def forecast_sentiment_distribution(self):
        """Forecast sentiment distribution trends"""
        print("Forecasting sentiment distribution...")
        
        # Monthly sentiment counts
        sentiment_monthly = self.df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
        sentiment_monthly.index = sentiment_monthly.index.to_timestamp()
        
        # Calculate percentages
        sentiment_pct = sentiment_monthly.div(sentiment_monthly.sum(axis=1), axis=0) * 100
        
        # Simple trend analysis
        trends = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in sentiment_pct.columns:
                recent_avg = sentiment_pct[sentiment].tail(3).mean()
                overall_avg = sentiment_pct[sentiment].mean()
                trend = 'increasing' if recent_avg > overall_avg else 'decreasing'
                trends[sentiment] = {
                    'current_percentage': float(recent_avg),
                    'historical_average': float(overall_avg),
                    'trend': trend
                }
        
        return trends


class InsightsVisualizer:
    """Visualize insights and generate reports"""
    
    def __init__(self, df):
        self.df = df
        
    def plot_sentiment_distribution(self, ax=None):
        """Plot sentiment distribution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentages
        total = sentiment_counts.sum()
        for i, (sentiment, count) in enumerate(sentiment_counts.items()):
            pct = count / total * 100
            ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def plot_sentiment_timeline(self, ax=None):
        """Plot sentiment over time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        sentiment_timeline = self.df.groupby([self.df['timestamp'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
        sentiment_timeline.index = sentiment_timeline.index.to_timestamp()
        
        sentiment_timeline.plot(kind='line', ax=ax, marker='o', linewidth=2)
        ax.set_title('Sentiment Trends Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Feedback Count', fontsize=12)
        ax.legend(title='Sentiment', loc='best')
        ax.grid(True, alpha=0.3)
    
    def plot_rating_trends(self, forecast_data=None, ax=None):
        """Plot rating trends and forecast"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        monthly_ratings = self.df.groupby(self.df['timestamp'].dt.to_period('M'))['rating'].mean()
        monthly_ratings.index = monthly_ratings.index.to_timestamp()
        
        ax.plot(monthly_ratings.index, monthly_ratings.values, marker='o', linewidth=2, 
                label='Historical', color='#3498db')
        
        if forecast_data is not None:
            forecast_df = forecast_data['forecast']
            ax.plot(forecast_df['month'], forecast_df['forecast_rating'], 
                   marker='s', linewidth=2, linestyle='--', 
                   label='Forecast', color='#e74c3c')
        
        ax.set_title('Customer Satisfaction Score Trends', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Rating', fontsize=12)
        ax.set_ylim(0, 5.5)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def plot_product_ratings(self, ax=None):
        """Plot ratings by product"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        product_ratings = self.df.groupby('product')['rating'].mean().sort_values()
        
        colors = ['#e74c3c' if r < 3 else '#f39c12' if r < 4 else '#2ecc71' 
                  for r in product_ratings.values]
        
        product_ratings.plot(kind='barh', ax=ax, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Average Rating by Product', fontsize=16, fontweight='bold')
        ax.set_xlabel('Average Rating', fontsize=12)
        ax.set_ylabel('Product', fontsize=12)
        ax.set_xlim(0, 5.5)
        ax.grid(axis='x', alpha=0.3)
    
    def create_wordcloud(self, sentiment='negative', ax=None):
        """Create word cloud for specific sentiment"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        sentiment_text = ' '.join(self.df[self.df['sentiment'] == sentiment]['processed_text'].astype(str))
        
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            colormap='RdYlGn_r' if sentiment == 'negative' else 'RdYlGn',
                            max_words=100).generate(sentiment_text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Common Words in {sentiment.capitalize()} Feedback', 
                    fontsize=16, fontweight='bold')
    
    def generate_comprehensive_report(self, forecast_data=None):
        """Generate comprehensive visualization report"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Sentiment distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_sentiment_distribution(ax1)
        
        # Sentiment timeline
        ax2 = fig.add_subplot(gs[0, 1:])
        self.plot_sentiment_timeline(ax2)
        
        # Rating trends with forecast
        ax3 = fig.add_subplot(gs[1, :])
        self.plot_rating_trends(forecast_data, ax3)
        
        # Product ratings
        ax4 = fig.add_subplot(gs[2, 0])
        self.plot_product_ratings(ax4)
        
        # Negative feedback word cloud
        ax5 = fig.add_subplot(gs[2, 1:])
        self.create_wordcloud('negative', ax5)
        
        plt.suptitle('Customer Feedback Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.995)
        
        plt.savefig('models/AI_insights_dashboard.png', dpi=300, bbox_inches='tight')
        print("Comprehensive dashboard saved")


def generate_insights_report(df):
    """Generate complete insights report"""
    print("=" * 60)
    print("GENERATING PREDICTIVE INSIGHTS")
    print("=" * 60)
    
    # Initialize analyzers
    analyzer = FeedbackInsightsAnalyzer(df)
    forecaster = SatisfactionForecaster(df)
    visualizer = InsightsVisualizer(df)
    
    # Analyze current trends
    print("\n1. Sentiment Trends")
    sentiment_insights = analyzer.analyze_sentiment_trends()
    print(f"   - Positive: {sentiment_insights['sentiment_percentage'].get('positive', 0)}%")
    print(f"   - Negative: {sentiment_insights['sentiment_percentage'].get('negative', 0)}%")
    print(f"   - Neutral: {sentiment_insights['sentiment_percentage'].get('neutral', 0)}%")
    
    print("\n2. Recurring Issues")
    issues = analyzer.identify_recurring_issues()
    print(f"   Top issues mentioned: {[word for word, count in issues['common_keywords'][:5]]}")
    print(f"   Most problematic products: {list(issues['problematic_products'].keys())[:3]}")
    
    print("\n3. Rating Analysis")
    rating_insights = analyzer.analyze_rating_trends()
    print(f"   Overall average rating: {rating_insights['overall_avg_rating']:.2f}")
    print(f"   Best rated product: {max(rating_insights['product_ratings'], key=rating_insights['product_ratings'].get)}")
    
    print("\n4. Customer Behavior")
    customer_insights = analyzer.analyze_customer_behavior()
    print(f"   Total customers: {customer_insights['total_customers']}")
    print(f"   Repeat customers: {customer_insights['repeat_customers']} ({customer_insights['repeat_rate']}%)")
    
    # Forecast future trends
    print("\n5. Satisfaction Forecast")
    forecast_data = forecaster.forecast_satisfaction(periods=3)
    print(f"   Current average: {forecast_data['current_avg']:.2f}")
    print(f"   Forecasted average: {forecast_data['forecast_avg']:.2f}")
    print(f"   Trend: {forecast_data['forecast']['trend'].iloc[0]}")
    
    sentiment_forecast = forecaster.forecast_sentiment_distribution()
    print("\n6. Sentiment Trends Forecast")
    for sentiment, data in sentiment_forecast.items():
        print(f"   {sentiment.capitalize()}: {data['trend']} (current: {data['current_percentage']:.1f}%)")
    
    # Generate visualizations
    print("\n7. Generating Visualizations...")
    visualizer.generate_comprehensive_report(forecast_data)
    
    # Compile report
    full_report = {
        'sentiment_insights': sentiment_insights,
        'issues': issues,
        'rating_insights': rating_insights,
        'customer_insights': customer_insights,
        'forecast_data': forecast_data,
        'sentiment_forecast': sentiment_forecast
    }
    
    return full_report


def main():
    """Main execution function"""
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('data/processed/customer_feedback_processed.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Generate insights
    report = generate_insights_report(df)
    
    print("\n" + "=" * 60)
    print("INSIGHTS GENERATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - models/AI_insights_dashboard.png")
    
    return report


if __name__ == "__main__":
    report = main()