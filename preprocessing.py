"""
Data Preprocessing Module for Stock Market Analysis
Handles data cleaning, feature engineering, and preparation for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os


class StockDataPreprocessor:
    """Comprehensive preprocessing pipeline for stock market data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_features = []
        self.categorical_features = []
        
    def load_data(self, filepath):
        """Load stock data from CSV file"""
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values using forward fill for time series data"""
        # Check for missing values
        missing_before = df.isnull().sum()
        
        # Forward fill for time series data
        df = df.sort_values('Date')
        df = df.ffill()
        
        # Backward fill for any remaining
        df = df.bfill()
        
        missing_after = df.isnull().sum()
        
        return df, missing_before, missing_after
    
    def encode_categorical_variables(self, df, fit=True):
        """Encode categorical variables using Label Encoding"""
        categorical_cols = ['Company', 'Sector', 'Trend']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        df[f'{col}_Encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def create_technical_indicators(self, df):
        """Create technical indicators for stock analysis"""
        # Sort by date and company
        df = df.sort_values(['Company', 'Date'])
        
        # Simple Moving Averages (SMA)
        df['SMA_5'] = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df['SMA_10'] = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        df['SMA_20'] = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        
        # Exponential Moving Average (EMA)
        df['EMA_12'] = df.groupby('Company')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        df['EMA_26'] = df.groupby('Company')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        # Daily Returns
        df['Daily_Return'] = df.groupby('Company')['Close'].pct_change()
        df['Daily_Return'] = df['Daily_Return'].fillna(0)
        
        # Price Range
        df['Price_Range'] = df['High'] - df['Low']
        
        # Average True Range (simplified)
        df['ATR'] = df['Price_Range']
        
        # RSI (Relative Strength Index) - simplified
        delta = df.groupby('Company')['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        
        # Bollinger Bands
        df['BB_Middle'] = df['SMA_20']
        rolling_std = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
        df['BB_Upper'] = df['BB_Middle'] + (2 * rolling_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * rolling_std)
        
        return df
    
    def create_lag_features(self, df):
        """Create lag features for time series prediction"""
        df = df.sort_values(['Company', 'Date'])
        
        # Lag features for Close price
        df['Close_Lag1'] = df.groupby('Company')['Close'].shift(1)
        df['Close_Lag2'] = df.groupby('Company')['Close'].shift(2)
        df['Close_Lag3'] = df.groupby('Company')['Close'].shift(3)
        
        # Lag features for Volume
        df['Volume_Lag1'] = df.groupby('Company')['Volume'].shift(1)
        
        # Fill NaN values from lag features
        df = df.bfill()
        
        return df
    
    def create_date_features(self, df):
        """Create date-based features"""
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        return df
    
    def detect_outliers_iqr(self, df, columns):
        """Detect outliers using IQR method"""
        outlier_info = {}
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return outlier_info
    
    def handle_outliers(self, df, columns, method='clip'):
        """Handle outliers using clipping or removal"""
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if method == 'clip':
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        return df_clean
    
    def scale_features(self, df, feature_columns, fit=True):
        """Scale numerical features using StandardScaler"""
        df_scaled = df.copy()
        
        if fit:
            df_scaled[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            df_scaled[feature_columns] = self.scaler.transform(df[feature_columns])
        
        return df_scaled
    
    def prepare_features_for_modeling(self, df):
        """Prepare final feature set for modeling"""
        # Define feature columns (excluding target and non-predictive columns)
        exclude_cols = ['Date', 'Company', 'Sector', 'Trend', 'Close']  # Close is target
        
        # Get numerical features
        feature_cols = [col for col in df.columns if col not in exclude_cols 
                       and df[col].dtype in ['float64', 'int64']]
        
        return feature_cols
    
    def create_train_test_split(self, df, target_col='Close', test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        feature_cols = self.prepare_features_for_modeling(df)
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def full_preprocessing_pipeline(self, filepath, save_artifacts=True):
        """Complete preprocessing pipeline"""
        # Load data
        print("Loading data...")
        df = self.load_data(filepath)
        df_original = df.copy()
        
        # Handle missing values
        print("Handling missing values...")
        df, missing_before, missing_after = self.handle_missing_values(df)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        df = self.encode_categorical_variables(df, fit=True)
        
        # Create technical indicators
        print("Creating technical indicators...")
        df = self.create_technical_indicators(df)
        
        # Create lag features
        print("Creating lag features...")
        df = self.create_lag_features(df)
        
        # Create date features
        print("Creating date features...")
        df = self.create_date_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        # Prepare features
        feature_cols = self.prepare_features_for_modeling(df)
        
        # Store feature columns for later use
        self.numerical_features = feature_cols
        
        # Detect outliers (before handling)
        print("Detecting outliers...")
        outlier_info = self.detect_outliers_iqr(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Save artifacts
        if save_artifacts:
            print("Saving preprocessing artifacts...")
            os.makedirs('model', exist_ok=True)
            with open('model/preprocessor.pkl', 'wb') as f:
                pickle.dump(self, f)
        
        print("Preprocessing complete!")
        return df, df_original, outlier_info
    
    def save(self, filepath='model/preprocessor.pkl'):
        """Save preprocessor to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath='model/preprocessor.pkl'):
        """Load preprocessor from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def get_preprocessing_summary(df_original, df_processed):
    """Generate summary statistics comparing original and processed data"""
    summary = {
        'original_shape': df_original.shape,
        'processed_shape': df_processed.shape,
        'features_added': df_processed.shape[1] - df_original.shape[1],
        'rows_removed': df_original.shape[0] - df_processed.shape[0],
        'new_features': [col for col in df_processed.columns if col not in df_original.columns]
    }
    return summary
