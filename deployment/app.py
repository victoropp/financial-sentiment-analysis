"""
Financial Sentiment Analysis Dashboard
Interactive Streamlit application for real-time sentiment analysis and model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
import sys
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page config
st.set_page_config(page_title="Financial Sentiment Analysis", layout="wide", page_icon="💹")

# Custom CSS for high contrast
st.markdown("""
<style>
    .stApp {background-color: #0e1117; color: #ffffff;}
    h1, h2, h3 {color: #00CC96 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: bold !important;}
    [data-testid=\"stMetricValue\"] {color: #00FF00 !important; font-size: 1.8rem !important; font-weight: bold !important;}
    [data-testid=\"stMetricLabel\"] {color: #ffffff !important; font-size: 1.1rem !important; font-weight: 600 !important;}
    .stButton > button {background-color: #00CC96; color: #000000 !important; border: 2px solid #00CC96; border-radius: 8px; font-weight: bold; font-size: 1.1rem; padding: 0.6rem 1.2rem;}
    .stButton > button:hover {background-color: #00FF00; color: #000000 !important; border-color: #00FF00; box-shadow: 0 0 10px #00CC96;}
    .stTextArea label, .stSelectbox label {color: #ffffff !important; font-size: 1.2rem !important; font-weight: bold !important;}
    .stAlert {background-color: #1a1a1a; border: 2px solid #00CC96; color: #ffffff !important; font-weight: 600 !important;}
    .stTabs [data-baseweb=\"tab\"] {color: #ffffff !important; font-weight: bold; font-size: 1.1rem;}
    .stTabs [aria-selected=\"true\"] {background-color: #00CC96 !important; color: #000000 !important;}
</style>
""", unsafe_allow_html=True)

# Sidebar with professional branding
with st.sidebar:
    st.title("Victor Collins Oppon")
    st.markdown("**NLP Engineer • Data Scientist • Chartered Accountant (ACCA)**")
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 15px; border-radius: 5px; border: 2px solid #00CC96;'>
        <h4 style='margin:0; color: #00CC96;'>🎯 NLP & Finance Expertise</h4>
        <p style='margin:5px 0; color: #ffffff; font-weight: 600;'>
            • Transformer Models (BERT, FinBERT) for financial text<br>
            • Traditional ML (XGBoost, SVM, RF) for sentiment scoring<br>
            • Financial reporting & risk‑analytics integration<br>
            • ACCA‑level accounting & audit rigor<br>
            • Production‑ready APIs & dashboards
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 15px; border-radius: 5px; border: 2px solid #AB63FA;'>
        <h4 style='margin:0; color: #AB63FA;'>📊 Dataset</h4>
        <p style='margin:5px 0; color: #ffffff; font-size: 0.95rem;'>
            <b>Financial Phrase Bank v1.0</b><br>
            4,846 expert‑annotated sentences<br>
            3 classes: Positive, Neutral, Negative<br>
            Domain: Financial news & earnings
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    st.markdown("**Tech Stack:** Python, Transformers, PyTorch, Scikit‑learn, XGBoost, Streamlit")

# Main title
st.title("💹 Financial Sentiment Analysis Platform")
st.markdown("""
<div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 5px solid #00CC96; border-right: 5px solid #AB63FA;'>
    <h3 style='margin:0; color: #00CC96;'>NLP-Powered Market Intelligence</h3>
    <p style='margin:10px 0 0 0; color: #ffffff; font-size: 1.05rem; line-height: 1.6;'>
        <b>State-of-the-art platform</b> combining <span style='color: #00CC96;'>Transformer Models</span> (BERT, FinBERT) and <span style='color: #AB63FA;'>Traditional ML</span> (XGBoost, SVM, Random Forest) for <b>real-time financial sentiment classification</b> across news and earnings reports.
    </p>
    <hr style='border-color: #4c4c4c; margin: 15px 0;'>
    <p style='margin:0; color: #e0e0e0; font-size: 0.95rem;'>
        <b>Capabilities:</b> Multi-model comparison - Real-time prediction - Model explainability - Batch processing - Production-ready API
    </p>
</div>
<br>
""", unsafe_allow_html=True)

# Model loading functions
@st.cache_resource
def load_traditional_models():
    """Load traditional ML models and vectorizer."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'traditional')
    models = {}
    try:
        vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        models['vectorizer'] = vectorizer
        for model_name in ['logistic_regression', 'random_forest', 'xgboost', 'svm']:
            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        metrics_path = os.path.join(models_dir, 'traditional_models_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                models['metrics'] = json.load(f)
        return models
    except Exception as e:
        st.error(f"Error loading traditional models: {e}")
        return None

@st.cache_resource
def load_bert_models():
    """Load BERT models."""
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from bert_classifier import FinancialBERTClassifier
    models_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    bert_models = {}
    try:
        bert_mini_dir = os.path.join(models_base, 'bert_embeddings')
        if os.path.exists(bert_mini_dir):
            bert_models['bert_mini'] = FinancialBERTClassifier.load(bert_mini_dir)
            metrics_path = os.path.join(bert_mini_dir, 'bert_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    bert_models['bert_mini_metrics'] = json.load(f)
        bert_mpnet_dir = os.path.join(models_base, 'bert_mpnet')
        if os.path.exists(bert_mpnet_dir):
            bert_models['bert_mpnet'] = FinancialBERTClassifier.load(bert_mpnet_dir)
            metrics_path = os.path.join(bert_mpnet_dir, 'bert_mpnet_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    bert_models['bert_mpnet_metrics'] = json.load(f)
        return bert_models
    except Exception as e:
        st.error(f"Error loading BERT models: {e}")
        return {}

# Load models
models = load_traditional_models()
bert_models = load_bert_models()

# Tabs
tabs = st.tabs(["🎯 Sentiment Analyzer", "📊 Model Performance", "🔍 Model Comparison", "📈 Batch Analysis"])

# Tab 1: Sentiment Analyzer
with tabs[0]:
    st.header("🎯 Real-Time Sentiment Analysis")
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 18px; border-radius: 8px; border: 2px solid #00CC96;'>
        <h4 style='margin:0; color: #00CC96;'>Financial Text Classification</h4>
        <p style='margin:8px 0 0 0; color: #ffffff; font-size: 1rem; line-height: 1.5;'>
            Analyze financial news, earnings reports, and market commentary using state-of-the-art NLP models. Get instant sentiment predictions with confidence scores.
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        text_input = st.text_area("Enter Financial Text", placeholder="Example: The company reported strong earnings growth of 25% year-over-year, exceeding analyst expectations.", height=150)
        selected_model = st.selectbox("Select Model", ["BERT (MPNet) - Best", "XGBoost", "BERT (MiniLM)", "Random Forest", "Logistic Regression", "SVM"])
        if st.button("Analyze Sentiment"):
            if text_input:
                with st.spinner("Analyzing..."):
                    if "BERT" in selected_model and bert_models:
                        if "MPNet" in selected_model and "bert_mpnet" in bert_models:
                            bert_model = bert_models["bert_mpnet"]
                        elif "MiniLM" in selected_model and "bert_mini" in bert_models:
                            bert_model = bert_models["bert_mini"]
                        else:
                            st.error("BERT model not found!")
                            st.stop()
                        prediction = bert_model.predict([text_input])[0]
                        probabilities = bert_model.predict_proba([text_input])[0]
                    elif models:
                        vectorizer = models["vectorizer"]
                        X = vectorizer.transform([text_input])
                        model_map = {"XGBoost": "xgboost", "Random Forest": "random_forest", "Logistic Regression": "logistic_regression", "SVM": "svm"}
                        model = models[model_map[selected_model]]
                        prediction = model.predict(X)[0]
                        probabilities = model.predict_proba(X)[0]
                    else:
                        st.error("Models not loaded!")
                        st.stop()
                    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                    sentiment = label_map[prediction]
                    confidence = probabilities[prediction] * 100
                    st.markdown("### Prediction Result")
                    color_map = {"Negative": "#FF4B4B", "Neutral": "#FFD700", "Positive": "#00CC96"}
                    st.markdown(f"""
                    <div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; border: 3px solid {color_map[sentiment]};'>
                        <h2 style='margin:0; color: {color_map[sentiment]};'>{sentiment}</h2>
                        <p style='margin:5px 0 0 0; color: #ffffff; font-size: 1.2rem;'>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("### Probability Distribution")
                    prob_df = pd.DataFrame({"Sentiment": ["Negative", "Neutral", "Positive"], "Probability": probabilities * 100})
                    fig = px.bar(prob_df, x='Sentiment', y='Probability', color='Sentiment', color_discrete_map=color_map, text='Probability')
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter text to analyze")
    with col2:
        st.markdown("### Example Texts")
        examples = {
            "Positive": "The company's quarterly revenue surged 30%, driven by strong demand and operational efficiency.",
            "Neutral": "The board of directors announced a regular dividend payment for Q4 shareholders.",
            "Negative": "Profit margins declined significantly due to rising costs and supply chain disruptions."
        }
        for sentiment, example in examples.items():
            if st.button(f"📝 {sentiment} Example", key=f"ex_{sentiment}"):
                st.text_area("Example", value=example, height=100, disabled=True)

# (Tabs 2-4 remain unchanged for brevity)

