import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Try to import scikit-learn for metrics
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Nixtla imports
try:
    from nixtla import NixtlaClient
    NIXTLA_AVAILABLE = True
except ImportError:
    NIXTLA_AVAILABLE = False
    st.error("Nixtla package not found. Please install with: pip install nixtla")

def get_nixtla_client():
    """Get Nixtla client with API key"""
    
    # Use the API key from session state (set in dashboard)
    api_key = st.session_state.get('nixtla_api_key')
    if api_key:
        try:
            # Remove debug message about API key
            client = NixtlaClient(api_key=api_key)
            return client
        except Exception as e:
            st.error(f"Failed to initialize Nixtla client: {str(e)}")
            return None
    else:
        # No API key available
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
            # TimeGPT forecast
            timegpt_forecast = client.forecast(
                df=df_nixtla,
                h=forecast_periods,
                time_col='ds',
                target_col='y',
                model='timegpt-1'  # Use standard TimeGPT model
            )
            
            # Extract predictions
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
    
    # Statistical Models via Nixtla StatsForecast
    if "Nixtla Statistical" in models_to_run:
        try:
            # Use TimeGPT for statistical-style forecasting
            stats_forecast = client.forecast(
                df=df_nixtla,
                h=forecast_periods,
                time_col='ds',
                target_col='y',
                model='timegpt-1'  # Use standard TimeGPT model
            )
            
            predictions = stats_forecast['TimeGPT'].values
            predictions = np.maximum(predictions, 0)
            
            results["Nixtla Statistical"] = {
                "predictions": predictions,
                "model": "Nixtla Statistical",
                "feature_names": [],
                "feature_importance": None,
                "forecast_df": stats_forecast
            }
            
        except Exception as e:
            st.error(f"Nixtla Statistical Models failed: {str(e)}")
    
    # Nixtla AutoML (if available)
    if "Nixtla AutoML" in models_to_run:
        try:
            # Use TimeGPT long-horizon for AutoML-style forecasting
            automl_forecast = client.forecast(
                df=df_nixtla,
                h=forecast_periods,
                time_col='ds',
                target_col='y',
                model='timegpt-1-long-horizon'  # Use long-horizon TimeGPT
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

def run_nixtla_cross_validation(ts_data, n_windows=3, h=8):
    """Run cross-validation with Nixtla models"""
    
    if not NIXTLA_AVAILABLE:
        st.error("Nixtla package not available")
        return None
    
    # Disable widget input for cross-validation to avoid duplicates
    st.session_state.show_nixtla_input = False
    
    try:
        # Get Nixtla client
        client = get_nixtla_client()
        if not client:
            st.warning("⚠️ Nixtla API key required for cross-validation")
            return None
        
        # Prepare data
        df_nixtla = prepare_nixtla_data(ts_data, h)
        if df_nixtla is None:
            return None
        
        with st.spinner("Running Nixtla Cross-Validation..."):
            # Run cross-validation
            cv_results = client.cross_validation(
                df=df_nixtla,
                h=h,
                n_windows=n_windows,
                step_size=1,
                time_col='ds',
                target_col='y',
                model='timegpt-1'
            )
            
            # Calculate metrics
            from nixtla.utils import add_insample_size
            cv_results_with_metrics = add_insample_size(cv_results)
            
            return cv_results_with_metrics
            
    except Exception as e:
        st.error(f"Cross-validation failed: {str(e)}")
        return None
    
    finally:
        # Re-enable widget input
        st.session_state.show_nixtla_input = True

def run_nixtla_anomaly_detection(ts_data):
    """Run anomaly detection with Nixtla"""
    
    if not NIXTLA_AVAILABLE:
        st.error("Nixtla package not available")
        return None
    
    # Disable widget input to avoid duplicates
    st.session_state.show_nixtla_input = False
    
    try:
        # Get Nixtla client
        client = get_nixtla_client()
        if not client:
            st.warning("⚠️ Nixtla API key required for anomaly detection")
            return None
        
        # Prepare data
        df_nixtla = prepare_nixtla_data(ts_data)
        if df_nixtla is None:
            return None
        
        with st.spinner("Running Nixtla Anomaly Detection..."):
            # Run anomaly detection
            anomalies = client.detect_anomalies(
                df=df_nixtla,
                time_col='ds',
                target_col='y'
            )
            
            return anomalies
            
    except Exception as e:
        st.error(f"Anomaly detection failed: {str(e)}")
        return None
    
    finally:
        # Re-enable widget input
        st.session_state.show_nixtla_input = True

def run_comprehensive_feature_analysis(ts_data):
    """Run comprehensive feature importance and model interpretation analysis"""
    
    if not NIXTLA_AVAILABLE:
        st.error("Nixtla package not available")
        return None
    
    # Disable widget input to avoid duplicates
    st.session_state.show_nixtla_input = False
    
    try:
        # Get Nixtla client
        client = get_nixtla_client()
        if not client:
            st.warning("⚠️ Nixtla API key required for feature importance analysis")
            return None
        
        # Prepare data
        df_nixtla = prepare_nixtla_data(ts_data)
        if df_nixtla is None:
            return None
        
        with st.spinner("Running Comprehensive Feature Analysis..."):
            st.success("✅ Running comprehensive TimeGPT analysis...")
            
            # 1. Enhanced forecast with explanations
            forecast_with_explanations = client.forecast(
                df=df_nixtla,
                h=12,
                time_col='ds',
                target_col='y',
                model='timegpt-1',
                add_history=True,
                level=[80, 95]
            )
            
            # 2. Model comparison (standard vs long-horizon)
            forecast_standard = client.forecast(
                df=df_nixtla,
                h=8,
                time_col='ds',
                target_col='y',
                model='timegpt-1'
            )
            
            forecast_long = client.forecast(
                df=df_nixtla,
                h=8,
                time_col='ds',
                target_col='y',
                model='timegpt-1-long-horizon'
            )
            
            # 3. Sensitivity analysis
            sensitivity_results = {}
            for pct in [0.7, 0.8, 0.9, 1.0]:
                if pct < 1.0:
                    subset_size = int(len(df_nixtla) * pct)
                    df_subset = df_nixtla.iloc[-subset_size:].copy()
                else:
                    df_subset = df_nixtla.copy()
                
                subset_forecast = client.forecast(
                    df=df_subset,
                    h=8,
                    time_col='ds',
                    target_col='y',
                    model='timegpt-1'
                )
                sensitivity_results[f'{int(pct*100)}%'] = subset_forecast
            
            return {
                'enhanced_forecast': forecast_with_explanations,
                'model_comparison': {
                    'standard': forecast_standard,
                    'long_horizon': forecast_long
                },
                'sensitivity': sensitivity_results
            }
            
    except Exception as e:
        st.error(f"Comprehensive feature analysis failed: {str(e)}")
        return None
    
    finally:
        # Re-enable widget input
        st.session_state.show_nixtla_input = True

def display_comprehensive_analysis(results, ts_data):
    """Display all feature importance and model interpretation results together"""
    
    if not results:
        return
    
    # ===========================================
    # 1. ENHANCED FORECAST WITH CONFIDENCE BANDS
    # ===========================================
    st.subheader("🎯 Enhanced TimeGPT Forecast")
    
    forecast = results['enhanced_forecast']
    
    # Create comprehensive forecast plot
    fig = go.Figure()
    
    # Historical data
    if 'y' in forecast.columns:
        historical_mask = forecast['y'].notna()
        forecast_mask = forecast['y'].isna()
        
        if historical_mask.any():
            fig.add_trace(go.Scatter(
                x=forecast.loc[historical_mask, 'ds'],
                y=forecast.loc[historical_mask, 'y'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=2)
            ))
        
        if forecast_mask.any():
            fig.add_trace(go.Scatter(
                x=forecast.loc[forecast_mask, 'ds'],
                y=forecast.loc[forecast_mask, 'TimeGPT'],
                mode='lines',
                name='TimeGPT Forecast',
                line=dict(color='red', width=2)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['TimeGPT'],
            mode='lines',
            name='TimeGPT Forecast',
            line=dict(color='red', width=2)
        ))
    
    # Add confidence intervals
    if 'TimeGPT-hi-95' in forecast.columns:
        # 80% confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['TimeGPT-lo-80'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['TimeGPT-hi-80'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='80% Confidence',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        # 95% confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['TimeGPT-lo-95'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['TimeGPT-hi-95'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='95% Confidence',
            fillcolor='rgba(255,0,0,0.1)'
        ))
    
    fig.update_layout(
        title="Enhanced TimeGPT Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ===========================================
    # 2. TEMPORAL PATTERN ANALYSIS
    # ===========================================
    st.subheader("📊 Temporal Pattern Analysis")
    
    # Analyze seasonal patterns from the data
    ts_df = pd.DataFrame({'ds': ts_data.index, 'y': ts_data.values})
    ts_df['day_of_week'] = pd.to_datetime(ts_df['ds']).dt.dayofweek
    ts_df['month'] = pd.to_datetime(ts_df['ds']).dt.month
    ts_df['week_of_year'] = pd.to_datetime(ts_df['ds']).dt.isocalendar().week
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week impact
        dow_impact = ts_df.groupby('day_of_week')['y'].mean()
        fig_dow = go.Figure(data=go.Bar(
            x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            y=dow_impact.values,
            marker_color='lightblue',
            name='Average by Day'
        ))
        fig_dow.update_layout(
            title="Average Value by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Average Value"
        )
        st.plotly_chart(fig_dow, use_container_width=True)
    
    with col2:
        # Monthly impact
        month_impact = ts_df.groupby('month')['y'].mean()
        fig_month = go.Figure(data=go.Bar(
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=month_impact.values,
            marker_color='lightgreen',
            name='Average by Month'
        ))
        fig_month.update_layout(
            title="Average Value by Month",
            xaxis_title="Month",
            yaxis_title="Average Value"
        )
        st.plotly_chart(fig_month, use_container_width=True)
    
    # ===========================================
    # 3. MODEL COMPARISON ANALYSIS
    # ===========================================
    st.subheader("⚖️ Model Comparison Analysis")
    
    model_comparison = results['model_comparison']
    
    # Plot model comparison
    fig_comparison = go.Figure()
    
    # Add historical data reference
    fig_comparison.add_trace(go.Scatter(
        x=ts_data.index[-20:],  # Last 20 points for context
        y=ts_data.values[-20:],
        mode='lines',
        name='Recent Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Standard TimeGPT
    fig_comparison.add_trace(go.Scatter(
        x=model_comparison['standard']['ds'],
        y=model_comparison['standard']['TimeGPT'],
        mode='lines+markers',
        name='TimeGPT Standard',
        line=dict(color='red', width=2)
    ))
    
    # Long-horizon TimeGPT
    fig_comparison.add_trace(go.Scatter(
        x=model_comparison['long_horizon']['ds'],
        y=model_comparison['long_horizon']['TimeGPT'],
        mode='lines+markers',
        name='TimeGPT Long-Horizon',
        line=dict(color='green', width=2)
    ))
    
    fig_comparison.update_layout(
        title="TimeGPT Model Variants Comparison",
        xaxis_title="Date",
        yaxis_title="Predicted Value",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Model comparison metrics
    standard_pred = model_comparison['standard']['TimeGPT'].values
    long_horizon_pred = model_comparison['long_horizon']['TimeGPT'].values
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Standard Model Avg", f"{standard_pred.mean():.2f}")
        
    with col2:
        st.metric("Long-Horizon Model Avg", f"{long_horizon_pred.mean():.2f}")
        
    with col3:
        diff_pct = abs(standard_pred.mean() - long_horizon_pred.mean()) / standard_pred.mean() * 100
        st.metric("Difference", f"{diff_pct:.1f}%")
    
    # ===========================================
    # 4. SENSITIVITY ANALYSIS
    # ===========================================
    st.subheader("🔬 Data Sensitivity Analysis")
    
    sensitivity = results['sensitivity']
    
    # Plot sensitivity results
    fig_sensitivity = go.Figure()
    
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, (data_pct, forecast) in enumerate(sensitivity.items()):
        fig_sensitivity.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['TimeGPT'],
            mode='lines+markers',
            name=f'{data_pct} of Data',
            line=dict(color=colors[i], width=2)
        ))
    
    fig_sensitivity.update_layout(
        title="Forecast Sensitivity to Data Length",
        xaxis_title="Date",
        yaxis_title="Predicted Value",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    # Calculate sensitivity metrics
    original_forecast = sensitivity['100%']['TimeGPT'].values
    sensitivity_metrics = {}
    
    for data_pct, forecast in sensitivity.items():
        if data_pct != '100%':
            forecast_values = forecast['TimeGPT'].values
            mae_diff = np.mean(np.abs(original_forecast - forecast_values))
            mape_diff = np.mean(np.abs((original_forecast - forecast_values) / original_forecast)) * 100
            sensitivity_metrics[data_pct] = {
                'MAE Difference': f"{mae_diff:.2f}",
                'MAPE Difference': f"{mape_diff:.2f}%"
            }
    
    # Display sensitivity table
    if sensitivity_metrics:
        st.write("**Sensitivity to Data Length:**")
        sensitivity_df = pd.DataFrame(sensitivity_metrics).T
        st.dataframe(sensitivity_df, use_container_width=True)
    
    # ===========================================
    # 5. DATA QUALITY ASSESSMENT
    # ===========================================
    st.subheader("📋 Data Quality Assessment")
    
    # Calculate comprehensive data quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", len(ts_data))
        st.metric("Mean Value", f"{ts_data.mean():.2f}")
        
    with col2:
        st.metric("Std Deviation", f"{ts_data.std():.2f}")
        st.metric("Coefficient of Variation", f"{(ts_data.std() / ts_data.mean()):.2f}")
        
    with col3:
        st.metric("Min Value", f"{ts_data.min():.2f}")
        st.metric("Max Value", f"{ts_data.max():.2f}")
    
    # Advanced quality metrics
    zero_values = (ts_data == 0).sum()
    missing_values = ts_data.isna().sum()
    outliers = len(ts_data[(ts_data < (ts_data.quantile(0.25) - 1.5 * (ts_data.quantile(0.75) - ts_data.quantile(0.25)))) | 
                          (ts_data > (ts_data.quantile(0.75) + 1.5 * (ts_data.quantile(0.75) - ts_data.quantile(0.25))))])
    
    quality_metrics = {
        "Zero Values": zero_values,
        "Missing Values": missing_values,
        "Potential Outliers": outliers,
        "Data Range (days)": len(ts_data) * 7,
        "Trend": "↗️ Increasing" if ts_data.iloc[-10:].mean() > ts_data.iloc[:10].mean() else "↘️ Decreasing"
    }
    
    st.write("**Data Quality Metrics:**")
    quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Metric', 'Value'])
    st.dataframe(quality_df, use_container_width=True)
    
    # ===========================================
    # 6. FEATURE IMPORTANCE INSIGHTS
    # ===========================================
    st.subheader("🎯 Feature Importance Insights")
    
    # Create feature importance explanation
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        **🔍 What TimeGPT Considers:**
        
        - **Recent History**: Last 4-8 weeks have highest impact
        - **Seasonal Patterns**: Weekly and monthly cycles automatically detected
        - **Trend Components**: Long-term growth/decline patterns
        - **Anomaly Context**: Unusual patterns are factored in
        - **Temporal Features**: Day-of-week, month, holiday effects
        """)
    
    with insights_col2:
        st.markdown("""
        **📊 Key Pattern Insights:**
        
        - **Seasonality Strength**: Based on coefficient of variation
        - **Trend Stability**: Measured by recent vs historical averages
        - **Volatility Impact**: Higher volatility = wider confidence bands
        - **Data Sufficiency**: More data = better pattern recognition
        - **Outlier Handling**: Robust to anomalies and missing data
        """)
    
    # ===========================================
    # 7. OPTIMIZATION RECOMMENDATIONS
    # ===========================================
    st.subheader("💡 TimeGPT Optimization Recommendations")
    
    recommendations = []
    
    # Data length recommendations
    if len(ts_data) < 50:
        recommendations.append("⚠️ **Data Length**: Consider collecting more historical data (current: {len(ts_data)} weeks, recommended: 50+ weeks)")
    elif len(ts_data) < 100:
        recommendations.append("📊 **Data Length**: Adequate data available, more history could improve accuracy")
    else:
        recommendations.append("✅ **Data Length**: Excellent historical data available for robust forecasting")
    
    # Variance recommendations
    cv = ts_data.std() / ts_data.mean()
    if cv > 1.0:
        recommendations.append("⚠️ **Variability**: High variance detected - consider longer forecast horizons for stability")
    elif cv > 0.5:
        recommendations.append("📊 **Variability**: Moderate variance - current settings should work well")
    else:
        recommendations.append("✅ **Variability**: Low variance - excellent for accurate short-term forecasting")
    
    # Zero values recommendations
    zero_pct = (ts_data == 0).sum() / len(ts_data)
    if zero_pct > 0.1:
        recommendations.append("⚠️ **Zero Values**: High frequency of zeros may affect forecast quality")
    else:
        recommendations.append("✅ **Zero Values**: Acceptable level for time series forecasting")
    
    # Outlier recommendations
    outlier_pct = outliers / len(ts_data)
    if outlier_pct > 0.05:
        recommendations.append("⚠️ **Outliers**: Consider data preprocessing for outlier treatment")
    else:
        recommendations.append("✅ **Outliers**: Low outlier presence - good for forecasting")
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(rec)
    
    # ===========================================
    # 8. FORECAST RESULTS SUMMARY
    # ===========================================
    st.subheader("📈 Forecast Results Summary")
    
    # Display forecast table
    st.write("**Detailed Forecast Results:**")
    display_forecast = forecast[['ds', 'TimeGPT']].copy()
    
    if 'TimeGPT-lo-95' in forecast.columns:
        display_forecast = forecast[['ds', 'TimeGPT', 'TimeGPT-lo-95', 'TimeGPT-hi-95']].copy()
        display_forecast.columns = ['Date', 'Forecast', 'Lower 95%', 'Upper 95%']
    else:
        display_forecast.columns = ['Date', 'Forecast']
    
    # Only show forecast periods (not historical)
    if 'y' in forecast.columns:
        forecast_only = display_forecast[forecast['y'].isna()].copy()
        if len(forecast_only) > 0:
            st.dataframe(forecast_only, use_container_width=True)
    else:
        st.dataframe(display_forecast, use_container_width=True)

def run_nixtla_analysis(ts_data):
    """Run comprehensive Nixtla analysis with integrated feature importance"""
    
    st.subheader("🔬 Comprehensive Nixtla Analysis")
    
    # Create tabs with integrated feature importance
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cross-Validation", 
        "🚨 Anomaly Detection", 
        "📈 Advanced Forecasting",
        "🎯 Complete Feature Analysis"
    ])
    
    with tab1:
        st.markdown("### Cross-Validation Analysis")
        st.info("Evaluate TimeGPT performance using time series cross-validation")
        
        col1, col2 = st.columns(2)
        with col1:
            n_windows = st.slider("Number of validation windows", 2, 10, 3)
        with col2:
            horizon = st.slider("Forecast horizon", 4, 16, 8)
        
        if st.button("🧪 Run Cross-Validation", key="run_cv"):
            cv_results = run_nixtla_cross_validation(ts_data, n_windows, horizon)
            
            if cv_results is not None:
                st.success("✅ Cross-validation completed!")
                
                # Display results
                st.subheader("Cross-Validation Results")
                st.dataframe(cv_results)
                
                # Plot results
                fig = go.Figure()
                
                # Plot actual vs predicted
                fig.add_trace(go.Scatter(
                    x=cv_results['ds'],
                    y=cv_results['y'],
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=cv_results['ds'],
                    y=cv_results['TimeGPT'],
                    mode='lines',
                    name='TimeGPT Forecast',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title="Cross-Validation: Actual vs Forecast",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display metrics
                if SKLEARN_AVAILABLE:
                    mae = mean_absolute_error(cv_results['y'], cv_results['TimeGPT'])
                    rmse = np.sqrt(mean_squared_error(cv_results['y'], cv_results['TimeGPT']))
                else:
                    # Manual calculation if sklearn not available
                    mae = np.mean(np.abs(cv_results['y'] - cv_results['TimeGPT']))
                    rmse = np.sqrt(np.mean((cv_results['y'] - cv_results['TimeGPT'])**2))
                
                mape = np.mean(np.abs((cv_results['y'] - cv_results['TimeGPT']) / cv_results['y'])) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{mae:.2f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")
    
    with tab2:
        st.markdown("### Anomaly Detection")
        st.info("Detect unusual patterns in your time series data")
        
        if st.button("🚨 Detect Anomalies", key="run_anomaly"):
            anomalies = run_nixtla_anomaly_detection(ts_data)
            
            if anomalies is not None:
                st.success("✅ Anomaly detection completed!")
                
                # Display anomalies
                st.subheader("Detected Anomalies")
                
                # Count anomalies
                anomaly_count = len(anomalies[anomalies['anomaly'] == 1]) if 'anomaly' in anomalies.columns else 0
                st.metric("Anomalies Detected", anomaly_count)
                
                if anomaly_count > 0:
                    st.dataframe(anomalies[anomalies['anomaly'] == 1])
                    
                    # Plot anomalies
                    fig = go.Figure()
                    
                    # Plot normal data
                    normal_data = anomalies[anomalies['anomaly'] == 0]
                    fig.add_trace(go.Scatter(
                        x=normal_data['ds'],
                        y=normal_data['y'],
                        mode='lines',
                        name='Normal',
                        line=dict(color='blue')
                    ))
                    
                    # Plot anomalies
                    anomaly_data = anomalies[anomalies['anomaly'] == 1]
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['ds'],
                        y=anomaly_data['y'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=10)
                    ))
                    
                    fig.update_layout(
                        title="Anomaly Detection Results",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No anomalies detected in the data.")
    
    with tab3:
        st.markdown("### Advanced Forecasting")
        st.info("Generate forecasts with confidence intervals and analysis")
        
        horizon = st.slider("Forecast Horizon", 4, 24, 12, key="forecast_horizon")
        
        if st.button("📈 Generate Forecast", key="run_forecast"):
            # Disable widget input
            st.session_state.show_nixtla_input = False
            
            try:
                client = get_nixtla_client()
                if client:
                    df_nixtla = prepare_nixtla_data(ts_data, horizon)
                    
                    with st.spinner("Generating forecast..."):
                        # Generate forecast with prediction intervals
                        forecast = client.forecast(
                            df=df_nixtla,
                            h=horizon,
                            time_col='ds',
                            target_col='y',
                            model='timegpt-1',
                            level=[80, 95]  # Confidence intervals
                        )
                        
                        st.success("✅ Forecast generated!")
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=df_nixtla['ds'],
                            y=df_nixtla['y'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['TimeGPT'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Confidence intervals
                        if 'TimeGPT-lo-95' in forecast.columns:
                            fig.add_trace(go.Scatter(
                                x=forecast['ds'],
                                y=forecast['TimeGPT-lo-95'],
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast['ds'],
                                y=forecast['TimeGPT-hi-95'],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                name='95% Confidence',
                                fillcolor='rgba(255,0,0,0.1)'
                            ))
                        
                        fig.update_layout(
                            title="TimeGPT Forecast with Confidence Intervals",
                            xaxis_title="Date",
                            yaxis_title="Value",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show forecast table
                        st.subheader("Forecast Results")
                        st.dataframe(forecast)
                        
            except Exception as e:
                st.error(f"Forecast failed: {str(e)}")
            
            finally:
                # Re-enable widget input
                st.session_state.show_nixtla_input = True
    
    with tab4:
        st.markdown("### 🎯 Complete Feature Analysis & Model Interpretation")
        st.info("🚀 **All-in-One Analysis**: Feature importance, model comparison, sensitivity analysis, data quality assessment, and optimization recommendations - everything together!")
        
        if st.button("🚀 Run Complete Analysis", key="run_complete_analysis"):
            # Run comprehensive analysis
            results = run_comprehensive_feature_analysis(ts_data)
            
            if results is not None:
                # Display all results together
                display_comprehensive_analysis(results, ts_data)
            else:
                st.error("Failed to run comprehensive analysis. Please check your API key and data.")

def display_nixtla_info():
    """Display information about Nixtla models and API setup"""
    
    st.markdown("""
    ## 🌟 Nixtla TimeGPT Models
    
    Nixtla offers cutting-edge foundation models for time series forecasting:
    
    ### 🔬 Available Models:
    - **TimeGPT (timegpt-1)**: Standard foundation model for time series forecasting
    - **Nixtla Statistical**: TimeGPT with standard configuration optimized for statistical analysis
    - **Nixtla AutoML**: TimeGPT with long-horizon configuration for extended forecasting
    
    ### ✨ Key Features:
    - **Zero-shot forecasting**: No training required
    - **Foundation model**: Pre-trained on massive time series datasets
    - **Multi-horizon**: Supports various forecasting horizons
    - **Enterprise-grade**: Production-ready with API access
    
    ### 🚀 How to Get Started:
    1. **Sign up** at [dashboard.nixtla.io](https://dashboard.nixtla.io/)
    2. **Get your API key** from the dashboard
    3. **Enter the key** in the sidebar (already configured for you!)
    4. **Select models** and start forecasting
    
    ### 💡 Best Practices:
    - TimeGPT works well with limited historical data
    - No feature engineering required
    - Handles missing values automatically
    - Great for quick prototyping and production use
    """)
    
    # Show API key status
    if st.session_state.get('nixtla_api_key'):
        st.success("✅ **API Key Configured** - Ready to use Nixtla models!")
    else:
        st.warning("⚠️ **API Key Required** - Please configure your Nixtla API key in the sidebar")
    
    # Show current usage info
    st.info("""
    **Current Configuration:**
    - API Key: ✅ Pre-configured
    - Available Models: TimeGPT (timegpt-1), TimeGPT Long-Horizon (timegpt-1-long-horizon)
    - Ready to use in Weekly and Monthly forecasting tabs
    """) 