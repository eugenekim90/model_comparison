import streamlit as st
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import Nixtla models
try:
    from nixtla_models import run_nixtla_models, NIXTLA_AVAILABLE
    NIXTLA_MODELS_AVAILABLE = True
except ImportError:
    NIXTLA_MODELS_AVAILABLE = False

def run_models(X_train, y_train, X_test, ts_data, models_to_run, use_optimized=True):
    """Run selected models and return results"""
    
    # Always use base features, no feature selection
    use_optimized = False
    
    results = {}
    
    # Remove verbose training messages
    
    # Convert to pandas for sklearn compatibility
    if hasattr(X_train, 'to_pandas'):
        X_train = X_train.to_pandas()
        X_test = X_test.to_pandas()
    
    # Fill any remaining NaN values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Remove any infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Feature selection for certain models - REMOVED since using base features only
    n_features = min(20, len(X_train.columns))  # Use top 20 or all available features
    
    forecast_periods = len(X_test)
    
    # Assess data quality for adaptive model selection
    train_length = len(X_train)
    is_short_series = train_length < 26  # Less than 6 months
    is_very_short = train_length < 15   # Less than 3.5 months
    
    # Remove data quality messages
    
    # Basic preprocessing for models that need scaled features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    selected_features = X_train.columns.tolist()
    
    # Remove Linear Regression and Neural Network sections
    
    # LightGBM (with adaptive parameters)
    if "LightGBM" in models_to_run:
        try:
            # Adaptive parameters based on data quality
            if is_very_short:
                n_estimators = 50
                learning_rate = 0.15
                reg_alpha = 0.5
                reg_lambda = 0.5
            elif is_short_series:
                n_estimators = 75
                learning_rate = 0.12
                reg_alpha = 0.3
                reg_lambda = 0.3
            else:
                n_estimators = 100
                learning_rate = 0.1
                reg_alpha = 0.1
                reg_lambda = 0.1
            
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42, 
                verbose=-1,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda
            )
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            pred = np.maximum(pred, 0)
            results["LightGBM"] = {"predictions": pred, "model": model, "feature_names": selected_features}
        except Exception as e:
            st.error(f"LightGBM failed: {str(e)}")
    
    # XGBoost (with adaptive parameters)
    if "XGBoost" in models_to_run:
        try:
            # Adaptive parameters based on data quality
            if is_very_short:
                n_estimators = 50
                learning_rate = 0.15
                reg_alpha = 0.5
                reg_lambda = 0.5
            elif is_short_series:
                n_estimators = 100
                learning_rate = 0.1
                reg_alpha = 0.2
                reg_lambda = 0.2
            else:
                n_estimators = 150
                learning_rate = 0.08
                reg_alpha = 0.1
                reg_lambda = 0.1
            
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42, 
                verbosity=0,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda
            )
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            pred = np.maximum(pred, 0)
            results["XGBoost"] = {"predictions": pred, "model": model, "feature_names": selected_features}
        except Exception as e:
            st.error(f"XGBoost failed: {str(e)}")
    
    # Random Forest (with adaptive parameters)
    if "Random Forest" in models_to_run:
        try:
            # Adaptive parameters based on data quality
            if is_very_short:
                n_estimators = 50
                max_depth = 5
                min_samples_split = 8
            elif is_short_series:
                n_estimators = 75
                max_depth = 8
                min_samples_split = 6
            else:
                n_estimators = 100
                max_depth = 10
                min_samples_split = 5
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42, 
                n_jobs=-1,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            pred = np.maximum(pred, 0)
            results["Random Forest"] = {"predictions": pred, "model": model, "feature_names": selected_features}
        except Exception as e:
            st.error(f"Random Forest failed: {str(e)}")
    
    # Remove Neural Network section
    
    # ETS (traditional time series model)
    if "ETS" in models_to_run:
        try:
            if len(ts_data) >= 24:
                model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=12)
            else:
                model = ExponentialSmoothing(ts_data, trend='add', seasonal=None)
            fitted = model.fit()
            pred = fitted.forecast(forecast_periods)
            pred = np.maximum(pred.values, 0)
            results["ETS"] = {"predictions": pred, "model": fitted, "feature_names": []}
        except Exception as e:
            st.error(f"ETS failed: {str(e)}")
    
    # TimeGPT (Nixtla)
    if "TimeGPT" in models_to_run:
        try:
            from nixtla import NixtlaClient
            # Check API key
            api_key = st.session_state.get('nixtla_api_key', '')
            if not api_key or api_key.strip() == '':
                st.error("⚠️ TimeGPT requires Nixtla API key. Please enter your API key in the sidebar.")
                results["TimeGPT"] = {"predictions": None, "model": None, "feature_names": [], "error": "API key required"}
            else:
                # Use the API key
                nixtla_client = NixtlaClient(api_key=api_key.strip())
                
                # Prepare data for Nixtla (ts_data is a pandas Series)
                forecast_df = ts_data.reset_index()
                forecast_df.columns = ['ds', 'y']
                forecast_df['unique_id'] = 'series_1'
                forecast_df = forecast_df[['unique_id', 'ds', 'y']]
                
                # Make forecast
                forecast = nixtla_client.forecast(
                    df=forecast_df,
                    h=forecast_periods,
                    time_col='ds',
                    target_col='y'
                )
                pred = forecast['TimeGPT'].values
                
                pred = np.maximum(pred, 0)
                results["TimeGPT"] = {"predictions": pred, "model": None, "feature_names": []}
            
        except Exception as e:
            st.error(f"TimeGPT failed: {str(e)}")
            results["TimeGPT"] = {"predictions": None, "model": None, "feature_names": [], "error": str(e)}

    # Nixtla AutoML
    if "Nixtla AutoML" in models_to_run:
        try:
            from nixtla import NixtlaClient
            # Check API key
            api_key = st.session_state.get('nixtla_api_key', '')
            if not api_key or api_key.strip() == '':
                st.error("⚠️ Nixtla AutoML requires Nixtla API key. Please enter your API key in the sidebar.")
                results["Nixtla AutoML"] = {"predictions": None, "model": None, "feature_names": [], "error": "API key required"}
            else:
                # Use the API key
                nixtla_client = NixtlaClient(api_key=api_key.strip())
                
                # Prepare data for Nixtla (ts_data is a pandas Series)
                forecast_df = ts_data.reset_index()
                forecast_df.columns = ['ds', 'y']
                forecast_df['unique_id'] = 'series_1'
                forecast_df = forecast_df[['unique_id', 'ds', 'y']]
                
                # Make forecast
                forecast = nixtla_client.forecast(
                    df=forecast_df,
                    h=forecast_periods,
                    time_col='ds',
                    target_col='y',
                    model='timegpt-1-long-horizon'
                )
                pred = forecast['TimeGPT'].values
                
                pred = np.maximum(pred, 0)
                results["Nixtla AutoML"] = {"predictions": pred, "model": None, "feature_names": []}
            
        except Exception as e:
            st.error(f"Nixtla AutoML failed: {str(e)}")
            results["Nixtla AutoML"] = {"predictions": None, "model": None, "feature_names": [], "error": str(e)}
    
    # Legacy Nixtla Models (if needed)
    nixtla_models = [model for model in models_to_run if model in ["Nixtla Statistical"]]
    if nixtla_models and NIXTLA_MODELS_AVAILABLE:
        try:
            nixtla_results = run_nixtla_models(ts_data, forecast_periods, nixtla_models)
            results.update(nixtla_results)
        except Exception:
            pass
    
    return results

def get_model_feature_importance(model_name, model, feature_names):
    """Get feature importance for tree-based models"""
    if model_name in ["LightGBM", "XGBoost", "Random Forest"] and hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    return None 