"""
Machine Learning Model Module for Stock Price Prediction
Handles model training, evaluation, and prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


class StockPricePredictor:
    """Comprehensive ML model for stock price prediction"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize predictor with specified model type
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'linear_regression'
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.feature_names = None
        self.feature_importance = None
        self.training_history = {}
        
    def _create_model(self, model_type):
        """Create ML model based on type"""
        if model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear_regression':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        print(f"Training {self.model_type} model...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Store training metrics
        train_pred = self.model.predict(X_train)
        self.training_history['train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
        self.training_history['train_mae'] = mean_absolute_error(y_train, train_pred)
        self.training_history['train_r2'] = r2_score(y_train, train_pred)
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.training_history['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            self.training_history['val_mae'] = mean_absolute_error(y_val, val_pred)
            self.training_history['val_r2'] = r2_score(y_val, val_pred)
        
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("Training complete!")
        return self.training_history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape,
            'predictions': y_pred,
            'actuals': y_test
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        scores = {
            'rmse': -cross_val_score(self.model, X, y, cv=cv, 
                                    scoring='neg_root_mean_squared_error'),
            'mae': -cross_val_score(self.model, X, y, cv=cv, 
                                   scoring='neg_mean_absolute_error'),
            'r2': cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        }
        
        return {
            'rmse_mean': scores['rmse'].mean(),
            'rmse_std': scores['rmse'].std(),
            'mae_mean': scores['mae'].mean(),
            'mae_std': scores['mae'].std(),
            'r2_mean': scores['r2'].mean(),
            'r2_std': scores['r2'].std()
        }
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("Feature importance not available for this model")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, y='feature', x='importance', ax=ax, palette='viridis')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        
        return fig
    
    def plot_actual_vs_predicted(self, y_test, y_pred):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Actual vs Predicted Values', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Time series comparison (first 100 points)
        n_points = min(100, len(y_test))
        indices = range(n_points)
        
        if isinstance(y_test, pd.Series):
            actual_values = y_test.iloc[:n_points].values
        else:
            actual_values = y_test[:n_points]
            
        pred_values = y_pred[:n_points]
        
        axes[1].plot(indices, actual_values, label='Actual', marker='o', markersize=4)
        axes[1].plot(indices, pred_values, label='Predicted', marker='x', markersize=4)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Price')
        axes[1].set_title(f'Actual vs Predicted (First {n_points} Samples)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_test, y_pred):
        """Plot residual analysis"""
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values', fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Residuals distribution
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals', fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Residuals over index
        axes[1, 1].plot(residuals.values if isinstance(residuals, pd.Series) else residuals)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Over Sample Index', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_intervals(self, y_test, y_pred, confidence=0.95):
        """Plot predictions with confidence intervals"""
        # Calculate prediction intervals (simplified)
        residuals = y_test - y_pred
        std_residuals = np.std(residuals)
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        margin = z_score * std_residuals
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_points = min(100, len(y_test))
        indices = range(n_points)
        
        if isinstance(y_test, pd.Series):
            actual_values = y_test.iloc[:n_points].values
        else:
            actual_values = y_test[:n_points]
        
        pred_values = y_pred[:n_points]
        
        ax.plot(indices, actual_values, label='Actual', marker='o', markersize=4, linewidth=2)
        ax.plot(indices, pred_values, label='Predicted', marker='x', markersize=4, linewidth=2)
        ax.fill_between(indices, pred_values - margin, pred_values + margin, 
                        alpha=0.3, label=f'{int(confidence*100)}% Confidence Interval')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Price')
        ax.set_title(f'Predictions with {int(confidence*100)}% Confidence Intervals', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath='model/stock_model.pkl'):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='model/stock_model.pkl'):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_model_info(self):
        """Get model information"""
        info = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        return info


def compare_models(X_train, X_test, y_train, y_test):
    """
    Compare different model types
    
    Returns:
        Dictionary with comparison results
    """
    model_types = ['random_forest', 'gradient_boosting', 'linear_regression']
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        predictor = StockPricePredictor(model_type=model_type)
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        
        results[model_type] = {
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2_Score'],
            'MAPE': metrics['MAPE']
        }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    
    return comparison_df


def create_prediction_input(feature_names):
    """
    Create a template for prediction input
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary with feature names and default values
    """
    prediction_template = {}
    
    # Set reasonable defaults for common features
    for feature in feature_names:
        if 'Open' in feature or 'Close' in feature or 'High' in feature or 'Low' in feature:
            prediction_template[feature] = 100.0
        elif 'Volume' in feature:
            prediction_template[feature] = 100000
        elif 'Market_Cap' in feature:
            prediction_template[feature] = 5e11
        elif 'PE_Ratio' in feature:
            prediction_template[feature] = 20.0
        elif 'Volatility' in feature:
            prediction_template[feature] = 0.03
        elif 'Sentiment' in feature:
            prediction_template[feature] = 0.5
        elif 'Dividend' in feature:
            prediction_template[feature] = 2.0
        elif 'Return' in feature or 'Daily_Return' in feature:
            prediction_template[feature] = 0.01
        elif 'MA' in feature or 'SMA' in feature or 'EMA' in feature:
            prediction_template[feature] = 100.0
        elif 'RSI' in feature:
            prediction_template[feature] = 50.0
        elif 'MACD' in feature:
            prediction_template[feature] = 0.0
        elif 'Year' in feature:
            prediction_template[feature] = 2022
        elif 'Month' in feature:
            prediction_template[feature] = 6
        elif 'Day' in feature:
            prediction_template[feature] = 15
        elif 'Encoded' in feature:
            prediction_template[feature] = 0
        else:
            prediction_template[feature] = 0.0
    
    return prediction_template


def make_runtime_prediction(model, feature_dict, preprocessor=None):
    """
    Make a prediction from user input
    
    Args:
        model: Trained model
        feature_dict: Dictionary of feature values
        preprocessor: Optional preprocessor for scaling
        
    Returns:
        Predicted value
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([feature_dict])
    
    # Apply preprocessing if available
    if preprocessor is not None and hasattr(preprocessor, 'scaler'):
        feature_cols = [col for col in input_df.columns if col in preprocessor.numerical_features]
        if feature_cols:
            input_df[feature_cols] = preprocessor.scaler.transform(input_df[feature_cols])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return prediction
