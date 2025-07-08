import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE safely"""
    try:
        # Convert to numpy arrays and ensure float type
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Create mask for valid values
        mask = (y_true != 0) & ~np.isnan(y_true) & ~np.isnan(y_pred) & np.isfinite(y_true) & np.isfinite(y_pred)
        
        if mask.sum() == 0:
            return np.inf
            
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
    except (ValueError, TypeError) as e:
        # Return infinity if conversion fails
        return np.inf

def weighted_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Weighted MAPE (WMAPE) - gives more weight to periods with higher actual values"""
    try:
        # Convert to numpy arrays and ensure float type
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Create mask for valid values
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
        
    except (ValueError, TypeError) as e:
        # Return infinity if conversion fails
        return np.inf

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    try:
        # Convert to numpy arrays and ensure float type for sklearn
        y_true_clean = np.asarray(y_true, dtype=np.float64)
        y_pred_clean = np.asarray(y_pred, dtype=np.float64)
        
        # Remove any NaN or infinite values for sklearn metrics
        mask = np.isfinite(y_true_clean) & np.isfinite(y_pred_clean)
        
        if mask.sum() == 0:
            return {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf, "WMAPE": np.inf}
        
        y_true_valid = y_true_clean[mask]
        y_pred_valid = y_pred_clean[mask]
        
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mape = mean_absolute_percentage_error(y_true, y_pred)  # Use original data for MAPE
        wmape = weighted_mean_absolute_percentage_error(y_true, y_pred)  # Use original data for WMAPE
        
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape}
        
    except Exception as e:
        # Return infinity for all metrics if calculation fails
        return {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf, "WMAPE": np.inf}

def calculate_model_metrics(model_results, y_test):
    """Calculate metrics for all model results"""
    model_metrics = {}
    
    for model_name, result in model_results.items():
        try:
            predictions = result["predictions"]
            
            # Skip models with None predictions (e.g., Nixtla models without API key)
            if predictions is None:
                continue
            
            # Skip models with error field (failed models)
            if "error" in result:
                continue
                
            metrics = calculate_metrics(y_test, predictions)
            model_metrics[model_name] = {
                "WMAPE": metrics["WMAPE"],
                "MAPE": metrics["MAPE"], 
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "predictions": predictions
            }
        except Exception as e:
            # Skip models that failed
            continue
            
    return model_metrics 