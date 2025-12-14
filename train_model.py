"""
Script to train the stock price prediction model
This script performs full preprocessing and model training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import StockDataPreprocessor
from model import StockPricePredictor, compare_models
import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("Stock Price Prediction - Model Training Pipeline")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor()
    
    # Run full preprocessing pipeline
    print("\n1. Running Preprocessing Pipeline...")
    df_processed, df_original, outlier_info = preprocessor.full_preprocessing_pipeline(
        'data/stock_data.csv',
        save_artifacts=True
    )
    
    print(f"\nOriginal data shape: {df_original.shape}")
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Features created: {df_processed.shape[1] - df_original.shape[1]}")
    
    # Create train-test split
    print("\n2. Creating Train-Test Split...")
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.create_train_test_split(
        df_processed, target_col='Close', test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Train Random Forest model
    print("\n3. Training Random Forest Model...")
    predictor = StockPricePredictor(model_type='random_forest')
    training_history = predictor.train(X_train, y_train)
    
    print("\nTraining Metrics:")
    print(f"  Train RMSE: {training_history['train_rmse']:.4f}")
    print(f"  Train MAE: {training_history['train_mae']:.4f}")
    print(f"  Train R²: {training_history['train_r2']:.4f}")
    
    # Evaluate on test set
    print("\n4. Evaluating on Test Set...")
    metrics = predictor.evaluate(X_test, y_test)
    
    print("\nTest Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R² Score: {metrics['R2_Score']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Save model
    print("\n5. Saving Model...")
    predictor.save_model('model/stock_model.pkl')
    
    # Save processed data for app
    print("\n6. Saving Processed Data...")
    df_processed.to_csv('data/processed_stock_data.csv', index=False)
    
    # Save feature columns
    import pickle
    with open('model/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save training/test split indices for visualization in app
    test_indices = X_test.index.tolist()
    with open('model/test_indices.pkl', 'wb') as f:
        pickle.dump(test_indices, f)
    
    # Save metrics
    with open('model/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\n" + "=" * 60)
    print("Training Complete! Model and artifacts saved.")
    print("=" * 60)
    
    return predictor, metrics


if __name__ == "__main__":
    main()
