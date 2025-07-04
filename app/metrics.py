import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE safely"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true != 0) & ~np.isnan(y_true) & ~np.isnan(y_pred) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def weighted_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Weighted MAPE (WMAPE) - gives more weight to periods with higher actual values"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.inf
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # WMAPE = (Σ|actual - predicted|) / (Σ|actual|) * 100
    # This gives more weight to periods with higher actual values
    numerator = np.sum(np.abs(y_true_filtered - y_pred_filtered))
    denominator = np.sum(np.abs(y_true_filtered))
    
    if denominator == 0:
        return np.inf
    
    return (numerator / denominator) * 100

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    wmape = weighted_mean_absolute_percentage_error(y_true, y_pred)
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape} 