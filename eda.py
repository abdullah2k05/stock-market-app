"""
Exploratory Data Analysis Module for Stock Market Data
Contains functions for comprehensive data analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class StockEDA:
    """Comprehensive EDA class for stock market analysis"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
    def get_summary_statistics(self):
        """1. Summary statistics for numerical features"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        summary = self.df[numerical_cols].describe()
        
        # Add additional statistics
        summary.loc['median'] = self.df[numerical_cols].median()
        summary.loc['mode'] = self.df[numerical_cols].mode().iloc[0]
        summary.loc['skewness'] = self.df[numerical_cols].skew()
        summary.loc['kurtosis'] = self.df[numerical_cols].kurtosis()
        
        return summary
    
    def analyze_missing_values(self):
        """2. Missing value analysis"""
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        
        return missing_data
    
    def plot_missing_values(self):
        """Missing value visualization"""
        missing_data = self.analyze_missing_values()
        
        if len(missing_data) == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'No Missing Values Found!', 
                   ha='center', va='center', fontsize=16, color='green')
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Missing_Percentage', y='Column', data=missing_data, ax=ax)
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.set_xlabel('Percentage Missing (%)')
        plt.tight_layout()
        return fig
    
    def plot_distribution_histograms(self, columns=None):
        """3. Distribution histograms for numerical features"""
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        n_cols = len(columns)
        fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(15, 4 * ((n_cols + 1) // 2)))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            if col in self.df.columns:
                axes[idx].hist(self.df[col], bins=50, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(alpha=0.3)
        
        # Remove extra subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self):
        """4. Correlation analysis with heatmap"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap of Features', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def detect_outliers_boxplot(self, columns=None):
        """5. Outlier detection using box plots"""
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        n_cols = len(columns)
        fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(15, 4 * ((n_cols + 1) // 2)))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            if col in self.df.columns:
                axes[idx].boxplot(self.df[col].dropna(), vert=True)
                axes[idx].set_title(f'Box Plot: {col}', fontweight='bold')
                axes[idx].set_ylabel(col)
                axes[idx].grid(alpha=0.3)
        
        # Remove extra subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        return fig
    
    def plot_time_series_trends(self):
        """6. Time series plots for stock trends"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Closing Price Over Time', 'Trading Volume Over Time'),
            vertical_spacing=0.12
        )
        
        # Group by date and calculate average
        daily_data = self.df.groupby('Date').agg({
            'Close': 'mean',
            'Volume': 'sum'
        }).reset_index()
        
        # Closing price
        fig.add_trace(
            go.Scatter(x=daily_data['Date'], y=daily_data['Close'], 
                      mode='lines', name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Scatter(x=daily_data['Date'], y=daily_data['Volume'], 
                      mode='lines', name='Volume', line=dict(color='orange'), fill='tozeroy'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Average Close Price", row=1, col=1)
        fig.update_yaxes(title_text="Total Volume", row=2, col=1)
        
        fig.update_layout(height=700, title_text="Stock Market Time Series Analysis", showlegend=False)
        
        return fig
    
    def plot_scatter_relationships(self):
        """7. Scatter plots for feature relationships"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Close vs Volume
        axes[0, 0].scatter(self.df['Volume'], self.df['Close'], alpha=0.5, s=10)
        axes[0, 0].set_xlabel('Volume')
        axes[0, 0].set_ylabel('Close Price')
        axes[0, 0].set_title('Close Price vs Volume', fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Open vs Close
        axes[0, 1].scatter(self.df['Open'], self.df['Close'], alpha=0.5, s=10)
        axes[0, 1].set_xlabel('Open Price')
        axes[0, 1].set_ylabel('Close Price')
        axes[0, 1].set_title('Open vs Close Price', fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Market Cap vs PE Ratio
        axes[1, 0].scatter(self.df['Market_Cap'], self.df['PE_Ratio'], alpha=0.5, s=10)
        axes[1, 0].set_xlabel('Market Cap')
        axes[1, 0].set_ylabel('PE Ratio')
        axes[1, 0].set_title('Market Cap vs PE Ratio', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Volatility vs Sentiment Score
        axes[1, 1].scatter(self.df['Volatility'], self.df['Sentiment_Score'], alpha=0.5, s=10)
        axes[1, 1].set_xlabel('Volatility')
        axes[1, 1].set_ylabel('Sentiment Score')
        axes[1, 1].set_title('Volatility vs Sentiment Score', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_volume_vs_price(self):
        """8. Volume vs Price analysis"""
        # Create volume bins
        self.df['Volume_Category'] = pd.qcut(self.df['Volume'], q=4, 
                                             labels=['Low', 'Medium', 'High', 'Very High'])
        
        volume_price_analysis = self.df.groupby('Volume_Category').agg({
            'Close': ['mean', 'std', 'min', 'max'],
            'Volume': ['mean', 'count']
        }).round(2)
        
        return volume_price_analysis
    
    def plot_volume_vs_price(self):
        """Volume vs Price visualization"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.df['Volume_Category'] = pd.qcut(self.df['Volume'], q=4, 
                                             labels=['Low', 'Medium', 'High', 'Very High'])
        
        sns.boxplot(data=self.df, x='Volume_Category', y='Close', ax=ax)
        ax.set_title('Price Distribution by Volume Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Volume Category')
        ax.set_ylabel('Close Price')
        plt.tight_layout()
        return fig
    
    def analyze_moving_averages(self):
        """9. Moving averages analysis"""
        # Calculate moving averages
        daily_close = self.df.groupby('Date')['Close'].mean().reset_index()
        daily_close = daily_close.sort_values('Date')
        
        daily_close['MA_7'] = daily_close['Close'].rolling(window=7, min_periods=1).mean()
        daily_close['MA_30'] = daily_close['Close'].rolling(window=30, min_periods=1).mean()
        daily_close['MA_90'] = daily_close['Close'].rolling(window=90, min_periods=1).mean()
        
        return daily_close
    
    def plot_moving_averages(self):
        """Moving averages visualization"""
        ma_data = self.analyze_moving_averages()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=ma_data['Date'], y=ma_data['Close'],
                                mode='lines', name='Close Price', line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(x=ma_data['Date'], y=ma_data['MA_7'],
                                mode='lines', name='7-Day MA', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=ma_data['Date'], y=ma_data['MA_30'],
                                mode='lines', name='30-Day MA', line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=ma_data['Date'], y=ma_data['MA_90'],
                                mode='lines', name='90-Day MA', line=dict(color='red', width=2)))
        
        fig.update_layout(title='Moving Averages Analysis',
                         xaxis_title='Date',
                         yaxis_title='Price',
                         height=500)
        
        return fig
    
    def analyze_volatility(self):
        """10. Volatility analysis"""
        volatility_stats = self.df.groupby('Sector')['Volatility'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(4)
        
        return volatility_stats
    
    def plot_volatility_by_sector(self):
        """Volatility visualization by sector"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(data=self.df, x='Sector', y='Volatility', ax=ax)
        ax.set_title('Volatility Distribution by Sector', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sector')
        ax.set_ylabel('Volatility')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def analyze_returns_distribution(self):
        """11. Returns distribution analysis"""
        # Calculate daily returns
        self.df = self.df.sort_values(['Company', 'Date'])
        self.df['Daily_Return'] = self.df.groupby('Company')['Close'].pct_change()
        
        returns_stats = self.df['Daily_Return'].describe()
        
        return returns_stats
    
    def plot_returns_distribution(self):
        """Returns distribution visualization"""
        self.df = self.df.sort_values(['Company', 'Date'])
        self.df['Daily_Return'] = self.df.groupby('Company')['Close'].pct_change()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.df['Daily_Return'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title('Distribution of Daily Returns', fontweight='bold')
        axes[0].set_xlabel('Daily Return')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].grid(alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(self.df['Daily_Return'].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot of Daily Returns', fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_grouped_aggregations(self):
        """12. Grouped aggregations by month, year, sector"""
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        
        # By Year
        yearly_agg = self.df.groupby('Year').agg({
            'Close': ['mean', 'std'],
            'Volume': 'sum',
            'Company': 'count'
        }).round(2)
        yearly_agg.columns = ['Avg_Close', 'Std_Close', 'Total_Volume', 'Record_Count']
        
        # By Month
        monthly_agg = self.df.groupby('Month').agg({
            'Close': ['mean', 'std'],
            'Volume': 'mean'
        }).round(2)
        monthly_agg.columns = ['Avg_Close', 'Std_Close', 'Avg_Volume']
        
        # By Sector
        sector_agg = self.df.groupby('Sector').agg({
            'Close': ['mean', 'std'],
            'Volume': 'mean',
            'Market_Cap': 'mean',
            'PE_Ratio': 'mean'
        }).round(2)
        
        return yearly_agg, monthly_agg, sector_agg
    
    def plot_grouped_trends(self):
        """Grouped trends visualization"""
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Average close by month
        monthly_avg = self.df.groupby('Month')['Close'].mean()
        axes[0, 0].bar(monthly_avg.index, monthly_avg.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Average Close Price by Month', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Close Price')
        axes[0, 0].grid(alpha=0.3)
        
        # Average volume by sector
        sector_vol = self.df.groupby('Sector')['Volume'].mean().sort_values()
        axes[0, 1].barh(sector_vol.index, sector_vol.values, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Average Volume by Sector', fontweight='bold')
        axes[0, 1].set_xlabel('Average Volume')
        axes[0, 1].grid(alpha=0.3)
        
        # Trend distribution
        trend_counts = self.df['Trend'].value_counts()
        axes[1, 0].pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Distribution of Market Trends', fontweight='bold')
        
        # Average PE Ratio by Sector
        sector_pe = self.df.groupby('Sector')['PE_Ratio'].mean().sort_values()
        axes[1, 1].barh(sector_pe.index, sector_pe.values, color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('Average PE Ratio by Sector', fontweight='bold')
        axes[1, 1].set_xlabel('Average PE Ratio')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_data_types_and_uniques(self):
        """13. Data types and unique value counts"""
        data_info = pd.DataFrame({
            'Column': self.df.columns,
            'Data_Type': self.df.dtypes.values,
            'Unique_Values': [self.df[col].nunique() for col in self.df.columns],
            'Sample_Value': [self.df[col].iloc[0] if len(self.df) > 0 else None for col in self.df.columns]
        })
        
        return data_info
    
    def analyze_trend_over_time(self):
        """14. Trend analysis over time (daily, monthly, yearly)"""
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        
        # Daily trend
        daily_trend = self.df.groupby('Date')['Close'].mean().reset_index()
        daily_trend['Change'] = daily_trend['Close'].pct_change()
        
        # Monthly trend
        monthly_trend = self.df.groupby(['Year', 'Month']).agg({
            'Close': 'mean',
            'Volume': 'sum'
        }).reset_index()
        
        return daily_trend, monthly_trend
    
    def plot_pairwise_relationships(self, features=None):
        """15. Pairwise feature relationships"""
        if features is None:
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Sample data if too large
        sample_df = self.df[features].sample(min(500, len(self.df)), random_state=42)
        
        fig = px.scatter_matrix(sample_df, dimensions=features,
                               title='Pairwise Relationships Between Features',
                               height=800, width=800)
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        
        return fig
    
    def generate_comprehensive_report(self):
        """Generate comprehensive EDA report"""
        report = {
            'dataset_shape': self.df.shape,
            'date_range': (self.df['Date'].min(), self.df['Date'].max()),
            'num_companies': self.df['Company'].nunique() if 'Company' in self.df.columns else 0,
            'num_sectors': self.df['Sector'].nunique() if 'Sector' in self.df.columns else 0,
            'total_trading_days': self.df['Date'].nunique(),
            'summary_stats': self.get_summary_statistics(),
            'missing_values': self.analyze_missing_values(),
            'data_types': self.analyze_data_types_and_uniques()
        }
        
        return report
