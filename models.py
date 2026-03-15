import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
from typing import Tuple, List, Union

warnings.filterwarnings('ignore')

def preprocess_and_feature_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, str, List[str]]:
    """Clean data and create features automatically with descriptive steps."""
    pipeline_steps = []
    df_clean = df.copy()
    
    # 1. Standardize column names
    df_clean.columns = [c.strip().capitalize() for c in df_clean.columns]
    pipeline_steps.append("Standardized column headers to Capitalized format.")

    # 2. Data Cleaning - Handle Missing Values
    null_count_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.ffill().bfill()
    null_count_after = df_clean.isnull().sum().sum()
    pipeline_steps.append(f"Handled missing values: {null_count_before} nulls resolved via forward/backward fill.")
    
    # 3. Datetime Indexing
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Date'])
        df_clean.set_index('Date', inplace=True)
        df_clean = df_clean.sort_index()
        pipeline_steps.append("Mapped 'Date' column to DatetimeIndex and sorted chronologically.")

    # 4. Numeric Conversion & Cleaning (Handing strings with commas like '1,234.56')
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj close']
    for col in numeric_cols:
        if col in df_clean.columns:
            if df_clean[col].dtype == object:
                # Remove commas and convert to float
                df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.strip()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    pipeline_steps.append("Cleaned numeric features (handled currency separators and string conversions).")

    # Find Target
    close_col = 'Close'
    if 'Close' not in df_clean.columns:
        # Try to find something similar
        for c in df_clean.columns:
            if 'close' in c.lower():
                close_col = c
                break
            
    if close_col not in df_clean.columns:
        return df_clean, False, "Missing 'Close' column.", pipeline_steps
    
    df_clean = df_clean.dropna(subset=[close_col])

    # 5. Feature Engineering
    if 'Ma_4' not in df_clean.columns:
        df_clean['Ma_4'] = df_clean[close_col].rolling(window=4).mean()
        pipeline_steps.append("Generated 4-day Moving Average (Ma_4) feature.")
        
    if 'Return' not in df_clean.columns:
        df_clean['Return'] = df_clean[close_col].pct_change()
        pipeline_steps.append("Calculated daily percentage returns.")
        
    if 'Volatility' not in df_clean.columns:
        df_clean['Volatility'] = df_clean['Return'].rolling(window=4).std()
        pipeline_steps.append("Engineered 4-day rolling volatility indicator.")

    if 'Log_return' not in df_clean.columns:
        df_clean['Log_return'] = np.log1p(df_clean['Return'].replace(-1, 0)) # safety for log(0)
        pipeline_steps.append("Generated Logarithmic Returns.")

    df_clean = df_clean.dropna()
    
    if len(df_clean) < 15:
        return df_clean, False, "Insufficient data (min 15 required).", pipeline_steps
    
    return df_clean, True, close_col, pipeline_steps

def get_correlation_matrix(df):
    """Calculate correlation for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()

def prepare_data(df, target_col='Close'):
    """Extract features and target arrays."""
    df_clean = df.dropna().copy()
    candidate_features = ['Open', 'High', 'Low', 'Volume', 'Ma_4', 'Volatility', 'Return', 'Log_return']
    features = [c for c in candidate_features if c in df_clean.columns]
    
    if not features:
        df_clean['Lag_1'] = df_clean[target_col].shift(1)
        df_clean = df_clean.dropna()
        features = ['Lag_1']
        
    X = df_clean[features]
    y = df_clean[target_col]
    return X, y, df_clean.index

def train_linear_regression(df, target_col='Close'):
    X, y, idx = prepare_data(df, target_col)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions)
    }
    
    return {
        'metrics': metrics,
        'y_test': y_test,
        'predictions': predictions,
        'test_index': y_test.index,
        'model': model
    }

def train_random_forest(df, target_col='Close'):
    X, y, idx = prepare_data(df, target_col)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Extract Feature Importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions)
    }
    
    return {
        'metrics': metrics,
        'y_test': y_test,
        'predictions': predictions,
        'test_index': y_test.index,
        'importance': importance,
        'model': model
    }

def train_arima(df, target_col='Close', order=(5, 1, 0), forecast_steps=30):
    y = df[target_col].values
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(y_test))
    
    metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions)
    }
        
    full_model = ARIMA(y, order=order)
    full_fit = full_model.fit()
    future_forecast = full_fit.forecast(steps=forecast_steps)
    
    return {
        'metrics': metrics,
        'y_test': y_test,
        'predictions': predictions,
        'future': future_forecast,
        'test_index': df.index[train_size:],
        'model': full_fit
    }
