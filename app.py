"""
Streamlit Web Application for Customer Feedback Analysis
Provides interactive interface for feedback analysis and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from data_preprocessing import FeedbackPreprocessor
    from summarization import FeedbackSummarizer
    from insights import FeedbackInsightsAnalyzer, SatisfactionForecaster
except:
    st.warning("Running in standalone mode. Some features may be limited.")

# Page configuration
st.set_page_config(
    page_title="Customer Feedback Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed feedback data"""
    try:
        df = pd.read_csv('data/processed/customer_feedback_processed.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        # Load sentiment model would go here
        # models['sentiment'] = load_sentiment_model()
        models['loaded'] = True
    except:
        models['loaded'] = False
    return models


def display_header():
    """Display application header"""
    st.markdown('<h1 class="main-header">🎯 Intelligent Customer Feedback Analysis System</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")


def display_overview(df):
    """Display overview metrics"""
    st.header("📈 Overview Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", f"{len(df):,}")
    
    with col2:
        avg_rating = df['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
    
    with col3:
        positive_pct = (df['sentiment'] == 'positive').sum() / len(df) * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    
    with col4:
        unique_customers = df['customer_id'].nunique()
        st.metric("Unique Customers", f"{unique_customers:,}")
    
    st.markdown("---")


def display_sentiment_analysis(df):
    """Display sentiment analysis section"""
    st.header("💭 Sentiment Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sentiment distribution pie chart
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment over time
        df_time = df.copy()
        df_time['month'] = df_time['timestamp'].dt.to_period('M').astype(str)
        sentiment_time = df_time.groupby(['month', 'sentiment']).size().reset_index(name='count')
        
        fig = px.line(
            sentiment_time,
            x='month',
            y='count',
            color='sentiment',
            title="Sentiment Trends Over Time",
            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        )
        st.plotly_chart(fig, use_container_width=True)


def display_rating_analysis(df):
    """Display rating analysis section"""
    st.header("⭐ Rating Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Rating distribution
        rating_counts = df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Rating Distribution",
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_counts.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average rating by product
        product_ratings = df.groupby('product')['rating'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=product_ratings.values,
            y=product_ratings.index,
            orientation='h',
            title="Average Rating by Product",
            labels={'x': 'Average Rating', 'y': 'Product'},
            color=product_ratings.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)


def display_insights(df):
    """Display key insights"""
    st.header("🔍 Key Insights")
    
    # Recurring issues
    negative_df = df[df['sentiment'] == 'negative']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚠️ Top Issues")
        if len(negative_df) > 0:
            product_issues = negative_df['product'].value_counts().head(5)
            for product, count in product_issues.items():
                st.markdown(f"- **{product}**: {count} negative mentions")
        else:
            st.info("No negative feedback found!")
    
    with col2:
        st.subheader("🏆 Top Performers")
        positive_df = df[df['sentiment'] == 'positive']
        if len(positive_df) > 0:
            product_praise = positive_df['product'].value_counts().head(5)
            for product, count in product_praise.items():
                st.markdown(f"- **{product}**: {count} positive mentions")


def display_forecast(df):
    """Display satisfaction forecast"""
    st.header("🔮 Satisfaction Forecast")
    
    try:
        forecaster = SatisfactionForecaster(df)
        forecast_data = forecaster.forecast_satisfaction(periods=3)
        
        # Prepare data for plotting
        historical = forecast_data['historical_data']
        forecast = forecast_data['forecast']
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['month'],
            y=historical['rating'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#3498db', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['month'],
            y=forecast['forecast_rating'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Customer Satisfaction Score Forecast",
            xaxis_title="Month",
            yaxis_title="Average Rating",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Average", f"{forecast_data['current_avg']:.2f}")
        
        with col2:
            st.metric("Forecast Average", f"{forecast_data['forecast_avg']:.2f}")
        
        with col3:
            trend = forecast['trend'].iloc[0]
            trend_emoji = "📈" if trend == "increasing" else "📉"
            st.metric("Trend", f"{trend_emoji} {trend.capitalize()}")
        
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")


def display_text_analyzer(df):
    """Interactive text analysis and prediction"""
    st.header("🔬 Text Analyzer")
    
    user_text = st.text_area(
        "Enter customer feedback to analyze:",
        placeholder="Type or paste feedback here...",
        height=150
    )
    
    if st.button("Analyze Feedback", type="primary"):
        if user_text:
            with st.spinner("Analyzing..."):
                # Simple rule-based sentiment (placeholder)
                text_lower = user_text.lower()
                positive_words = ['good', 'great', 'excellent', 'love', 'best', 'amazing', 'wonderful']
                negative_words = ['bad', 'poor', 'terrible', 'worst', 'hate', 'disappointed', 'awful']
                
                pos_count = sum(word in text_lower for word in positive_words)
                neg_count = sum(word in text_lower for word in negative_words)
                
                if pos_count > neg_count:
                    sentiment = "Positive 😊"
                    color = "green"
                elif neg_count > pos_count:
                    sentiment = "Negative 😞"
                    color = "red"
                else:
                    sentiment = "Neutral 😐"
                    color = "gray"
                
                st.markdown(f"### Predicted Sentiment: :{color}[{sentiment}]")
                
                # Display word count
                word_count = len(user_text.split())
                st.info(f"Word count: {word_count}")
        else:
            st.warning("Please enter some text to analyze.")


def display_data_upload():
    """Allow users to upload and analyze their own data"""
    st.header("📤 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Upload customer feedback CSV file",
        type=['csv'],
        help="CSV file should contain columns: feedback_text, rating, timestamp"
    )
    
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df_new)} records!")
            
            st.subheader("Data Preview")
            st.dataframe(df_new.head(10))
            
            if st.button("Analyze Uploaded Data"):
                st.info("Analysis feature will be implemented here.")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def main():
    """Main application function"""
    display_header()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Sentiment Analysis", "Rating Analysis", "Insights & Forecast", 
         "Text Analyzer", "Upload Data"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "📊 **About**\n\n"
        "This AI-powered system analyzes customer feedback "
        "to provide actionable insights and predictions."
    )
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("⚠️ Data file not found. Please run data preprocessing first.")
        st.code("python data_preprocessing.py")
        return
    
    # Display selected page
    if page == "Dashboard":
        display_overview(df)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Recent feedback
            st.subheader("📝 Recent Feedback")
            recent = df.nlargest(5, 'timestamp')[['timestamp', 'feedback_text', 'sentiment', 'rating']]
            for _, row in recent.iterrows():
                sentiment_emoji = "😊" if row['sentiment'] == 'positive' else "😞" if row['sentiment'] == 'negative' else "😐"
                st.markdown(f"**{row['timestamp'].date()}** {sentiment_emoji} ({row['rating']}⭐)")
                st.caption(row['feedback_text'][:100] + "...")
                st.markdown("---")
        
        with col2:
            # Source distribution
            st.subheader("📱 Feedback Sources")
            source_counts = df['source'].value_counts()
            fig = px.bar(
                x=source_counts.values,
                y=source_counts.index,
                orientation='h',
                title="Feedback by Source",
                color=source_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Sentiment Analysis":
        display_sentiment_analysis(df)
    
    elif page == "Rating Analysis":
        display_rating_analysis(df)
    
    elif page == "Insights & Forecast":
        display_insights(df)
        st.markdown("---")
        display_forecast(df)
    
    elif page == "Text Analyzer":
        display_text_analyzer(df)
    
    elif page == "Upload Data":
        display_data_upload()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit | Customer Feedback Analysis System v1.0"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()