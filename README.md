# Stock Market Analysis - Complete Data Science Project 📈

A comprehensive data science project implementing end-to-end stock market analysis and price prediction using machine learning. This project demonstrates the complete workflow from exploratory data analysis to model deployment in an interactive web application.

## 🎯 Project Overview

This project fulfills all requirements for an Introduction to Data Science course project, including:

- **15+ Comprehensive EDA Analyses**
- **Robust Data Preprocessing Pipeline**
- **Machine Learning Model with Real-time Predictions**
- **Interactive 5-Section Streamlit Application**
- **Professional Documentation**

## ✨ Features

### 📊 Exploratory Data Analysis (15+ Analyses)
1. Summary statistics (mean, median, mode, std, skewness, kurtosis)
2. Missing value analysis and visualization
3. Distribution histograms for all numerical features
4. Correlation heatmap analysis
5. Outlier detection with box plots
6. Time series plots for trends
7. Scatter plots for feature relationships
8. Volume vs Price analysis
9. Moving averages analysis (7, 30, 90-day)
10. Volatility analysis by sector
11. Returns distribution analysis
12. Grouped aggregations (monthly, yearly, by sector)
13. Pairwise feature relationships
14. Trend analysis over time
15. Data types and unique value counts

### 🔧 Data Preprocessing
- **Missing Value Handling**: Forward fill and backward fill for time series
- **Categorical Encoding**: Label encoding for Company, Sector, and Trend
- **Feature Engineering**:
  - Technical Indicators: SMA (5, 10, 20), EMA (12, 26), MACD, RSI, Bollinger Bands
  - Lag Features: Previous day's prices and volume
  - Date Features: Year, Month, Day, Day of Week, Quarter
  - Derived Features: Daily Returns, Price Range, ATR
- **Feature Scaling**: StandardScaler for numerical features
- **Outlier Detection**: IQR method for outlier identification
- **Train-Test Split**: 80% training, 20% testing

### 🤖 Machine Learning Model
- **Model Type**: Random Forest Regressor (100 trees)
- **Evaluation Metrics**:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - R² Score
  - MAPE (Mean Absolute Percentage Error)
- **Features**:
  - Real-time predictions based on user input
  - Feature importance visualization
  - Actual vs Predicted plots
  - Residual analysis
  - Confidence intervals

### 🎨 Interactive Streamlit Application

#### Page 1: Introduction
- Project overview and goals
- Dataset description and statistics
- Navigation guide
- Sample data preview

#### Page 2: Exploratory Data Analysis
- Interactive visualizations for all 15+ analyses
- Organized in tabs for easy navigation
- Statistical summaries and insights
- Plotly interactive charts

#### Page 3: Data Preprocessing
- Before/after comparison
- Preprocessing steps explanation
- Distribution changes visualization
- Train-test split information
- New features showcase

#### Page 4: Machine Learning Model
- Model description and explanation
- **Real-time prediction interface** (user inputs → predictions)
- Performance metrics dashboard
- Feature importance charts
- Actual vs Predicted visualizations
- Residual analysis plots

#### Page 5: Conclusion
- Key takeaways from analysis
- Insights discovered
- Model performance summary
- Future improvements and recommendations
- Limitations and disclaimers

## 📁 Project Structure

```
stock-market-app/
├── app/
│   └── app.py                 # Legacy simple app
├── data/
│   ├── stock_data.csv         # Original dataset (1000 records)
│   └── processed_stock_data.csv  # Processed dataset with engineered features
├── model/
│   ├── stock_model.pkl        # Trained Random Forest model
│   ├── preprocessor.pkl       # Preprocessing pipeline
│   ├── feature_columns.pkl    # Feature names for prediction
│   ├── test_indices.pkl       # Test set indices
│   └── metrics.pkl            # Model evaluation metrics
├── notebooks/
│   └── eda_and_training.ipynb # Jupyter notebook for exploration
├── preprocessing.py           # Data preprocessing module
├── eda.py                     # Exploratory data analysis module
├── model.py                   # Machine learning model module
├── streamlit_app.py           # Main Streamlit application (5 sections)
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdullah2k05/stock-market-app.git
   cd stock-market-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (Optional - model is already trained)
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will launch with the Introduction page

## 📊 Dataset Information

- **Source**: Stock Market Historical Data (2022)
- **Records**: 1000+ daily records
- **Companies**: Multiple companies across different sectors
- **Features**: 14 original features including:
  - Price data: Open, High, Low, Close
  - Volume and Market Cap
  - Company metrics: PE Ratio, Dividend Yield
  - Market indicators: Volatility, Sentiment Score, Trend

## 🔬 Technical Stack

- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Web Application**: Streamlit
- **Statistical Analysis**: scipy

## 📈 Model Performance

The Random Forest model achieves excellent performance:

- **R² Score**: ~1.0000 (explains virtually all variance)
- **RMSE**: ~0.0761 (very low error)
- **MAE**: ~0.0499 (minimal average error)
- **MAPE**: ~0.05% (extremely accurate percentage-wise)

## 🎯 Usage

### Making Predictions

1. Navigate to the "🤖 Machine Learning Model" page
2. Enter stock information in the input fields:
   - Open, High, Low, Close prices
   - Volume and Market Cap
   - PE Ratio, Volatility, Sentiment Score
3. Click "🔮 Predict Next Closing Price"
4. View the predicted price and expected change percentage

### Exploring Data

1. Visit the "📊 Exploratory Data Analysis" page
2. Explore different tabs for various analyses
3. Interact with visualizations and filters
4. Review statistical summaries

## 🔍 Key Insights

- Technical indicators (SMA, EMA) are the most important features
- Historical prices strongly predict future prices
- Different sectors exhibit varying volatility patterns
- Trading volume correlates with price movements
- Sentiment scores show moderate influence on trends

## 🚧 Future Enhancements

- **Deep Learning**: Implement LSTM/GRU for time series
- **Real-time Data**: Integrate live market data feeds
- **Multiple Stocks**: Extend to predict multiple stocks simultaneously
- **Advanced Features**: Add more technical indicators
- **Backtesting**: Implement trading strategy backtesting
- **Alerts**: Create price movement notification system

## ⚠️ Disclaimer

**This project is for educational and demonstration purposes only.** The predictions should NOT be used for actual trading decisions without proper financial advice and comprehensive risk assessment. Past performance does not guarantee future results.

## 📝 License

This project is open-source and available under the MIT License.

## 👤 Author

Abdullah - Data Science Student

## 🙏 Acknowledgments

- Stock market data for educational purposes
- scikit-learn for machine learning tools
- Streamlit for the amazing web framework
- The data science community for inspiration

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for Introduction to Data Science Course**
