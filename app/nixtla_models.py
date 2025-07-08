import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Nixtla imports
try:
    from nixtla import NixtlaClient
    NIXTLA_AVAILABLE = True
except ImportError:
    NIXTLA_AVAILABLE = False

def get_nixtla_client():
    """Get Nixtla client with API key"""
    api_key = st.session_state.get('nixtla_api_key')
    if api_key:
        try:
            client = NixtlaClient(api_key=api_key)
            return client
        except Exception as e:
            st.error(f"Failed to initialize Nixtla client: {str(e)}")
            return None
    return None

def prepare_nixtla_data(ts_data, forecast_periods=8):
    """Prepare data in Nixtla format"""
    try:
        # Reset index to get date column
        if isinstance(ts_data.index, pd.DatetimeIndex):
            df_nixtla = ts_data.reset_index()
            df_nixtla.columns = ['ds', 'y']  # Nixtla expects 'ds' for dates, 'y' for values
        else:
            # If index is not datetime, create a simple date range
            dates = pd.date_range(start='2020-01-01', periods=len(ts_data), freq='W')
            df_nixtla = pd.DataFrame({
                'ds': dates,
                'y': ts_data.values
            })
        
        # Add unique_id column (required by Nixtla)
        df_nixtla['unique_id'] = 'ts_1'
        
        # Reorder columns for Nixtla format
        df_nixtla = df_nixtla[['unique_id', 'ds', 'y']]
        
        return df_nixtla
        
    except Exception as e:
        st.error(f"Error preparing data for Nixtla: {str(e)}")
        return None

def run_nixtla_models(ts_data, forecast_periods=8, models_to_run=None):
    """Run Nixtla models (TimeGPT and others)"""
    
    if not NIXTLA_AVAILABLE:
        st.error("Nixtla package not available")
        return {}
    
    # Get Nixtla client
    client = get_nixtla_client()
    if not client:
        st.warning("⚠️ Nixtla API key required to use Nixtla models")
        return {}
    
    # Prepare data
    df_nixtla = prepare_nixtla_data(ts_data, forecast_periods)
    if df_nixtla is None:
        return {}
    
    results = {}
    
    # Default models if none specified
    if models_to_run is None:
        models_to_run = ["TimeGPT"]
    
    # TimeGPT (Nixtla's flagship foundation model)
    if "TimeGPT" in models_to_run:
        try:
            timegpt_forecast = client.forecast(
                df=df_nixtla,
                h=forecast_periods,
                time_col='ds',
                target_col='y',
                model='timegpt-1'
            )
            
            predictions = timegpt_forecast['TimeGPT'].values
            predictions = np.maximum(predictions, 0)  # Ensure non-negative
            
            results["TimeGPT"] = {
                "predictions": predictions,
                "model": "TimeGPT",
                "feature_names": [],
                "feature_importance": None,
                "forecast_df": timegpt_forecast
            }
            
        except Exception as e:
            st.error(f"TimeGPT failed: {str(e)}")
    
    # Nixtla AutoML
    if "Nixtla AutoML" in models_to_run:
        try:
            automl_forecast = client.forecast(
                df=df_nixtla,
                h=forecast_periods,
                time_col='ds',
                target_col='y',
                model='timegpt-1-long-horizon'
            )
            
            predictions = automl_forecast['TimeGPT'].values
            predictions = np.maximum(predictions, 0)
            
            results["Nixtla AutoML"] = {
                "predictions": predictions,
                "model": "Nixtla AutoML",
                "feature_names": [],
                "feature_importance": None,
                "forecast_df": automl_forecast
            }
            
        except Exception as e:
            st.error(f"Nixtla AutoML failed: {str(e)}")
    
    return results 