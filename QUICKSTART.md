# 🚀 Quick Start Guide

Get the Financial Sentiment Analysis platform running in under 5 minutes!

## Prerequisites

- Python 3.8 or higher installed
- pip package manager
- 2GB free disk space (for models)

## Installation Steps

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```

Or download and extract the ZIP file, then navigate to the folder.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including Streamlit, scikit-learn, XGBoost, and Transformers.

### 3. Launch the Dashboard

```bash
streamlit run deployment/app.py
```

The application will automatically open in your browser at `http://localhost:8501`

## Using the Dashboard

### 📊 Real-Time Sentiment Analyzer

1. Navigate to the **"Real-Time Sentiment Analyzer"** tab
2. Enter financial text (news, earnings reports, market commentary)
3. Select a model from the dropdown (XGBoost recommended)
4. Click **"Analyze Sentiment"**
5. View the prediction, confidence score, and probability distribution

**Example texts to try:**
- "The company reported strong earnings growth of 25% this quarter"
- "Revenue declined by 15% due to market headwinds"
- "The quarterly results were in line with analyst expectations"

### 📈 Model Performance

View comprehensive metrics for all trained models:
- Accuracy, Precision, Recall, F1-Score
- Per-class performance breakdown
- Confusion matrices
- Best model recommendations

### 🔍 Model Comparison

Compare all models side-by-side:
- Performance ranking by F1-Score
- Visual comparison charts
- Strengths and weaknesses analysis

### 📑 Batch Analysis

Upload a CSV file with a `text` column to analyze multiple sentences:

```csv
text
"Company A announced record profits"
"Market volatility increased significantly"
"Stable performance across all segments"
```

Download results with sentiment predictions and confidence scores.

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Models not found

**Solution**: The traditional ML models are included in `models/traditional/`. If missing:
```bash
python src/traditional_models.py
```

This will train and save all models (~5 minutes).

### Issue: Port already in use

**Solution**: Specify a different port
```bash
streamlit run deployment/app.py --server.port 8502
```

### Issue: Slow performance

**Solution**:
- Traditional ML models are very fast (milliseconds)
- For BERT/FinBERT, a GPU is recommended but not required
- Reduce batch size if processing many texts

## Project Structure

```
financial_sentiment_analysis/
├── deployment/app.py          # Main Streamlit dashboard
├── src/                       # Source code
│   ├── traditional_models.py  # ML training
│   └── data_loader.py         # Data utilities
├── models/traditional/        # Trained models
├── data/                      # Dataset
└── requirements.txt           # Dependencies
```

## Next Steps

1. ✅ **Explore the Dashboard**: Try different models and text inputs
2. ✅ **View Metrics**: Check model performance in the Performance tab
3. ✅ **Batch Processing**: Upload your own CSV files
4. ✅ **Read Documentation**: Check README.md for detailed information
5. ✅ **Professional Overview**: See PROFESSIONAL_OVERVIEW.md for business context

## Need Help?

- 📖 Check the main [README.md](README.md) for detailed documentation
- 💼 See [PROFESSIONAL_OVERVIEW.md](PROFESSIONAL_OVERVIEW.md) for business applications
- 🐛 Issues? Check the troubleshooting section above

---

**Ready to analyze financial sentiment!** 🎉

For advanced features and customization options, refer to the main README.md file.
