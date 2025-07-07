import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from feature_engineering import create_features
from data_preparation import prepare_model_data, prepare_aggregated_data, get_aggregation_level, create_display_name, smart_data_preparation
from models import run_models, get_model_feature_importance
from metrics import calculate_metrics
from visualization import perform_eda, plot_forecast_results, plot_metrics_comparison, display_metrics_table, plot_feature_importance, plot_prediction_scatter

def get_feature_type(feature_name):
    """Categorize feature types for better visualization"""
    if feature_name.startswith('lag_') or feature_name.startswith('sessions_lag_'):
        return 'Lag Features'
    elif feature_name.startswith('rolling_') or feature_name.startswith('volatility_') or feature_name.startswith('stability_score_'):
        return 'Rolling Statistics'
    elif (feature_name.endswith('_encoded') or feature_name.startswith('config_') or 
          feature_name in ['company_encoded', 'state_encoded', 'program_encoded', 'industry_encoded']):
        return 'Categorical Features'
    elif feature_name in ['week_of_year', 'month', 'quarter', 'is_winter', 'is_spring', 'is_summer', 'is_fall']:
        return 'Time Features'
    elif (feature_name.startswith('holiday_') or feature_name.startswith('seasonal_') or 
          feature_name in ['q4_intensity', 'post_holiday_recovery', 'holiday_season_gradient']):
        return 'Seasonal Features'
    elif (feature_name.startswith('recovery_') or feature_name.startswith('momentum_') or 
          feature_name.startswith('trend_') or feature_name.startswith('deviation_')):
        return 'Trend & Momentum Features'
    elif feature_name.startswith('yoy_') or feature_name.startswith('sessions_change_'):
        return 'Growth Features'
    elif feature_name.startswith('total_customers') or feature_name.startswith('subscriber_'):
        return 'Customer Features'
    else:
        return 'Other Features'

def weekly_forecasting(df, selected_company, selected_state, selected_program, models_to_run, test_split_date, test_end_date, use_optimized=True):
    """Run weekly forecasting with specified models"""
    
    # Initialize filters
    filters = []
    
    # Apply filters based on selections
    if selected_company != "ALL COMPANIES":
        filters.append(pl.col("company") == selected_company)
    if selected_state != "ALL STATES":
        filters.append(pl.col("us_state") == selected_state)
    if selected_program != "ALL PROGRAMS":
        filters.append(pl.col("episode_session_type") == selected_program)
    
    # Apply filters if any exist
    if filters:
        filtered_df = df.filter(pl.all_horizontal(filters))
    else:
        filtered_df = df
    
    # Debug info
    st.subheader("🔍 Debug Info")
    st.write(f"Original data shape: {df.shape}")
    st.write(f"Filtered data shape: {filtered_df.shape}")
    
    display_name = f"{selected_company} → {selected_state} → {selected_program}"
    st.success(f"**Selected:** {display_name}")
    
    if filtered_df.height == 0:
        st.error("No data available for the selected combination")
        return
    
    # Create features
    df_features = create_features(filtered_df, optimized=use_optimized)
    
    # Prepare data for modeling based on aggregation level
    if selected_company == "ALL COMPANIES" or selected_state == "ALL STATES" or selected_program == "ALL PROGRAMS":
        # Use aggregated data preparation
        
        # Determine aggregation level
        aggregation_level = get_aggregation_level(selected_company, selected_state, selected_program)
        
        # Use prepare_aggregated_data for aggregated selections
        X_train, y_train, X_test, y_test, ts_data = prepare_aggregated_data(
            df_features, selected_company, selected_state, selected_program, aggregation_level, test_split_date, test_end_date
        )
    else:
        # Use granular data preparation for specific combinations
        X_train, y_train, X_test, y_test, ts_data = prepare_model_data(
            df_features, selected_company, selected_state, selected_program, test_split_date, test_end_date
        )
        
    # Check if data preparation was successful
    if X_train is None or y_train is None:
        st.error("❌ **No sufficient data available for the selected combination**")
        st.info("**Possible reasons:**")
        st.markdown("""
        - Selected combination has insufficient historical data (need at least 15 training weeks)
        - No data available for the selected date range
        - Try selecting a different combination or date range
        """)
        return
    
    # Remove verbose training information
    # Run models
    model_results = run_models(X_train, y_train, X_test, ts_data, models_to_run, use_optimized)
    
    # Check data availability
    if X_test is not None and y_test is not None and len(y_test) > 0:
        # Historical evaluation mode
        st.subheader("📊 Model Performance")
        
        # Calculate metrics for all models
        model_metrics = {}
        for model_name, result in model_results.items():
            predictions = result["predictions"]
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, predictions)
            model_metrics[model_name] = {
                "WMAPE": metrics["WMAPE"],
                "MAPE": metrics["MAPE"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "predictions": predictions
            }
        
        # Sort by WMAPE performance
        sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]["WMAPE"])
        
        # Display metrics comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Model Rankings")
            for rank, (model_name, metrics) in enumerate(sorted_models, 1):
                icon = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
                st.write(f"{icon} **{model_name}** - WMAPE: {metrics['WMAPE']:.1f}%")
        
        with col2:
            # Remove "Best Model" success message
            pass
        
        # Display charts using the correct function from visualization
        # Create test dates for the chart
        test_start = pd.to_datetime(test_split_date)
        test_end = pd.to_datetime(test_end_date)
        test_dates = pd.date_range(start=test_start, end=test_end, freq='W-MON')[:len(y_test)]
        
        # Get the figure and display it
        forecast_fig = plot_forecast_results(y_test, model_results, test_dates)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Display metrics table using the correct function from visualization
        display_metrics_table(model_metrics)
        
        # Only display feature importance for granular (specific) selections
        # Feature importance doesn't make sense for aggregated data
        if selected_company != "ALL COMPANIES" and selected_state != "ALL STATES" and selected_program != "ALL PROGRAMS":
            st.subheader("🎯 Feature Importance Analysis")
            st.info("📊 **Granular Selection**: Showing feature importance for specific company/state/program combination")
            _display_feature_importance(model_results, X_train.columns, use_optimized, X_train, y_train)
        else:
            st.subheader("📈 Aggregated Data Analysis")
            aggregation_parts = []
            if selected_company == "ALL COMPANIES":
                aggregation_parts.append("all companies")
            if selected_state == "ALL STATES": 
                aggregation_parts.append("all states")
            if selected_program == "ALL PROGRAMS":
                aggregation_parts.append("all programs")
            
            st.info(f"🌐 **Aggregated Selection**: You're viewing data aggregated across {', '.join(aggregation_parts)}. Feature importance is not applicable here since the model is trained on aggregated time series data rather than individual entity features.")
            
            # Instead, show what features were actually used for the aggregated model
            with st.expander("🔧 Features Used in Aggregated Model"):
                st.markdown("""
                **For aggregated data, the model primarily relies on:**
                
                • **Temporal features**: Week of year, month, quarter patterns
                • **Lag features**: Historical values from previous weeks (1, 2, 4, 8, 12 weeks back)
                • **Rolling statistics**: Moving averages and trends over time windows
                • **Seasonal patterns**: Holiday effects, business cycles, yearly patterns
                • **Trend features**: Growth rates, momentum, volatility measures
                
                ℹ️ **Note**: Entity-specific features (company, state, program encodings) are not used in aggregated models since we're looking at combined data across multiple entities.
                """)
    
    else:
        # Future forecasting mode
        st.subheader("🔮 Future Forecasting")
        st.info("Forecasting into the future - no historical data available for evaluation")
        
        # Display future predictions using existing forecast results function
        st.subheader("📈 Future Predictions")
        
        # Create a simple prediction chart for future forecasting
        if model_results:
            # Get the first model's predictions as an example
            first_model = list(model_results.keys())[0]
            predictions = model_results[first_model]["predictions"]
            
            # Create date range for predictions
            test_start = pd.to_datetime(test_split_date)
            test_end = pd.to_datetime(test_end_date)
            prediction_dates = pd.date_range(start=test_start, end=test_end, freq='W-MON')[:len(predictions)]
            
            # Display predictions for all models
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔮 Model Predictions")
                for model_name, result in model_results.items():
                    preds = result["predictions"]
                    avg_pred = np.mean(preds) if len(preds) > 0 else 0
                    st.write(f"**{model_name}**: Avg. {avg_pred:.0f} sessions/week")
            
            with col2:
                # Create prediction chart
                fig = go.Figure()
                
                for model_name, result in model_results.items():
                    preds = result["predictions"]
                    dates = prediction_dates[:len(preds)]
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=preds,
                        mode='lines+markers',
                        name=model_name,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="Future Predictions by Model",
                    xaxis_title="Date",
                    yaxis_title="Session Count",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Only display feature importance for granular selections in future forecasting too
        if selected_company != "ALL COMPANIES" and selected_state != "ALL STATES" and selected_program != "ALL PROGRAMS":
            st.subheader("🎯 Feature Importance Analysis")
            st.info("📊 **Granular Selection**: Showing feature importance for specific company/state/program combination")
            _display_feature_importance(model_results, X_train.columns, use_optimized, X_train, y_train)
        else:
            st.subheader("📈 Aggregated Future Forecasting")
            aggregation_parts = []
            if selected_company == "ALL COMPANIES":
                aggregation_parts.append("all companies")
            if selected_state == "ALL STATES":
                aggregation_parts.append("all states") 
            if selected_program == "ALL PROGRAMS":
                aggregation_parts.append("all programs")
            
            st.info(f"🌐 **Aggregated Future Forecast**: Predicting combined future values across {', '.join(aggregation_parts)}. The model uses temporal patterns rather than entity-specific features.")
            
            with st.expander("🔧 Future Forecasting Features"):
                st.markdown("""
                **For future forecasting of aggregated data, models rely on:**
                
                • **Historical patterns**: Learning from past aggregate trends
                • **Seasonal cycles**: Yearly, quarterly, monthly patterns in the combined data
                • **Long-term trends**: Growth or decline patterns in the aggregated time series
                • **Recent momentum**: Latest trends in the combined data
                • **Holiday and calendar effects**: Business seasonality across all entities
                
                ℹ️ **Note**: Future forecasting extends these learned temporal patterns into the forecast period.
                """)

def _get_eda_dataframe(df_features, selected_company, selected_state, selected_program, aggregation_level):
    """Get the appropriate dataframe for EDA based on aggregation level"""
    
    if aggregation_level == "Granular" and selected_company != "ALL COMPANIES":
        eda_df = df_features.filter(
            (pl.col("company") == selected_company) &
            (pl.col("us_state") == selected_state) &
            (pl.col("episode_session_type") == selected_program)
        )
    else:
        # Get the full aggregated dataset for EDA
        if selected_company == "ALL COMPANIES":
            base_df = df_features
        else:
            base_df = df_features.filter(pl.col("company") == selected_company)
        
        if aggregation_level == "Global Total":
            eda_df = base_df.group_by("week_start").agg([
                pl.sum("session_count").alias("session_count")
            ])
        elif aggregation_level == "Global Program":
            eda_df = base_df.filter(pl.col("episode_session_type") == selected_program).group_by("week_start").agg([
                pl.sum("session_count").alias("session_count")
            ])
        elif aggregation_level == "Global State":
            eda_df = base_df.filter(pl.col("us_state") == selected_state).group_by("week_start").agg([
                pl.sum("session_count").alias("session_count")
            ])
        elif aggregation_level == "Global Granular":
            eda_df = base_df.filter(
                (pl.col("us_state") == selected_state) &
                (pl.col("episode_session_type") == selected_program)
            ).group_by("week_start").agg([
                pl.sum("session_count").alias("session_count")
            ])
        elif aggregation_level == "Company Total":
            eda_df = base_df.group_by("week_start").agg([
                pl.sum("session_count").alias("session_count")
            ])
        elif aggregation_level == "Company-State":
            eda_df = base_df.filter(pl.col("us_state") == selected_state).group_by("week_start").agg([
                pl.sum("session_count").alias("session_count")
            ])
        elif aggregation_level == "Company-Program":
            eda_df = base_df.filter(pl.col("episode_session_type") == selected_program).group_by("week_start").agg([
                pl.sum("session_count").alias("session_count")
            ])
    
    return eda_df

def _display_feature_info(X_train):
    """Display information about features used in training"""
    st.subheader("🔧 Features Used in Training")
    feature_cols = list(X_train.columns)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Lag Features:**")
        lag_features = [f for f in feature_cols if f.startswith("lag_")]
        for f in lag_features:
            st.write(f"• {f}")
    
    with col2:
        st.write("**Rolling Statistics:**")
        rolling_features = [f for f in feature_cols if f.startswith("rolling_")]
        for f in rolling_features:
            st.write(f"• {f}")
    
    with col3:
        st.write("**Other Features:**")
        other_features = [f for f in feature_cols if not f.startswith("lag_") and not f.startswith("rolling_")]
        for f in other_features:
            st.write(f"• {f}")
    
    # Show categorical features separately if they exist
    categorical_features = [f for f in feature_cols if f.endswith("_encoded") or f.startswith("config_")]
    if categorical_features:
        st.subheader("🏷️ Categorical Features Added")
        st.info("These features help the model learn patterns across different companies, states, programs, and configurations:")
        for f in categorical_features:
            if f == "company_encoded":
                st.write(f"• **{f}**: Numerical encoding of companies")
            elif f == "state_encoded":
                st.write(f"• **{f}**: Numerical encoding of US states")
            elif f == "program_encoded":
                st.write(f"• **{f}**: Numerical encoding of episode session types")
            elif f == "industry_encoded":
                st.write(f"• **{f}**: Numerical encoding of industry groups")
            elif f.startswith("config_"):
                st.write(f"• **{f}**: Binary indicator for configuration type")
            else:
                st.write(f"• **{f}**: Categorical feature")

def _display_metrics_explanation():
    """Display explanation of metrics"""
    with st.expander("📖 Understanding the Metrics"):
        st.markdown("""
        **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values
        
        **RMSE (Root Mean Square Error):** Square root of average squared differences (penalizes larger errors more)
        
        **MAPE (Mean Absolute Percentage Error):** Average percentage error across all predictions
        
        **WMAPE (Weighted MAPE):** 🎯 **Primary metric** - Gives more weight to periods with higher actual values, making it more reliable than MAPE for business forecasting
        
        *Lower values are better for all metrics*
        """)

def _plot_mape_vs_wmape_comparison(model_metrics):
    """Plot MAPE vs WMAPE comparison"""
    comparison_data = []
    for model_name in model_metrics.keys():
        comparison_data.append({
            "Model": model_name,
            "MAPE": model_metrics[model_name]["MAPE"],
            "WMAPE": model_metrics[model_name]["WMAPE"]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    mape_comparison_fig = px.bar(
        comparison_df.melt(id_vars="Model", value_vars=["MAPE", "WMAPE"], var_name="Metric", value_name="Value"),
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="MAPE vs WMAPE Comparison",
        labels={"Value": "Percentage Error (%)", "Model": "Models"}
    )
    st.plotly_chart(mape_comparison_fig, use_container_width=True)

def _display_individual_model_details(model_results, model_metrics, y_test):
    """Display individual model details"""
    st.header("🔍 Detailed Results")
    
    for model_name, result in model_results.items():
        with st.expander(f"{model_name} Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("MAE", f"{model_metrics[model_name]['MAE']:.2f}")
                st.metric("MAPE", f"{model_metrics[model_name]['MAPE']:.1f}%")
            
            with col2:
                st.metric("RMSE", f"{model_metrics[model_name]['RMSE']:.2f}")
                st.metric("WMAPE", f"{model_metrics[model_name]['WMAPE']:.1f}%")
            
            # Predicted vs Actual scatter plot
            scatter_fig = plot_prediction_scatter(y_test, result["predictions"], model_name)
            st.plotly_chart(scatter_fig, use_container_width=True)

def _display_feature_importance(model_results, original_feature_names, use_optimized=True, X_train=None, y_train=None):
    """Display feature importance for tree models"""
    st.header("🎯 Feature Importance")
    
    # Tree models feature importance
    tree_models = [model for model in model_results.keys() if model in ["LightGBM", "XGBoost", "Random Forest"]]
    if tree_models:
        for model_name in tree_models:
            if model_name in model_results and hasattr(model_results[model_name]["model"], 'feature_importances_'):
                with st.expander(f"{model_name} Feature Importance"):
                    importance = model_results[model_name]["model"].feature_importances_
                    
                    # Use the feature names from the model results (these are the ones actually used)
                    if "feature_names" in model_results[model_name]:
                        model_feature_names = model_results[model_name]["feature_names"]
                    else:
                        model_feature_names = original_feature_names  # Fallback
                    
                    # Ensure lengths match
                    if len(model_feature_names) != len(importance):
                        st.warning(f"⚠️ Feature name/importance length mismatch for {model_name}. Using indices.")
                        model_feature_names = [f"Feature_{i}" for i in range(len(importance))]
                    
                    importance_df = pd.DataFrame({
                        'Feature': model_feature_names,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    # Add feature type for better visualization
                    importance_df['Type'] = importance_df['Feature'].apply(get_feature_type)
                    
                    imp_fig = plot_feature_importance(importance_df, model_name, top_n=15)
                    st.plotly_chart(imp_fig, use_container_width=True)
                    
                    if use_optimized:
                        st.success("✅ Showing importance for optimized features (top 20 selected)")
                    
                    # Show categorical feature importance summary
                    categorical_importance = importance_df[importance_df['Type'] == 'Categorical Features']
                    if not categorical_importance.empty:
                        st.write("**Categorical Feature Impact:**")
                        for _, row in categorical_importance.iterrows():
                            percentage = (row['Importance'] / importance_df['Importance'].sum()) * 100
                            st.write(f"• **{row['Feature']}**: {percentage:.1f}% of total importance")
    
    # Handle Nixtla models (TimeGPT, etc.)
    nixtla_models = [model for model in model_results.keys() if model in ["TimeGPT", "Nixtla Statistical", "Nixtla AutoML"]]
    if nixtla_models:
        with st.expander("🌟 Nixtla Models - Actual Time Series Feature Analysis"):
            st.markdown("**🔍 Real Feature Analysis from Your Data:**")
            
            # Get the time series data that was used for Nixtla models
            from data_preparation import prepare_aggregated_data, get_aggregation_level
            try:
                # Get the same data preparation used for the models
                aggregation_level = get_aggregation_level("ALL COMPANIES", "ALL STATES", "ALL PROGRAMS")  # Default for analysis
                
                # Try to get time series data from the model results or recreate it
                ts_data = None
                for model_name in nixtla_models:
                    if model_name in model_results and hasattr(model_results[model_name], 'get'):
                        # Try to get ts_data from the model results if available
                        break
                
                # If we can't get it from model results, create a simple analysis from the available data
                if ts_data is None:
                    # Create a simple time series from the target values for analysis
                    # Get basic time series statistics
                    if y_train is not None and len(y_train) > 0:
                        ts_data = pd.Series(y_train)
                        ts_dates = pd.date_range(end=pd.Timestamp.now(), periods=len(ts_data), freq='W')
                    else:
                        st.warning("No training data available for analysis")
                        ts_data = None
                
                if ts_data is not None:
                    # Calculate actual temporal features and their importance
                    temporal_analysis = {}
                    
                    # 1. Trend Analysis
                    if len(ts_data) > 4:
                        recent_trend = np.mean(ts_data[-4:]) - np.mean(ts_data[:4])
                        trend_strength = abs(recent_trend) / np.std(ts_data) if np.std(ts_data) > 0 else 0
                        temporal_analysis["Trend Direction"] = "Increasing" if recent_trend > 0 else "Decreasing"
                        temporal_analysis["Trend Strength"] = f"{trend_strength:.2f}"
                    
                    # 2. Volatility Analysis
                    volatility = np.std(ts_data) / np.mean(ts_data) if np.mean(ts_data) > 0 else 0
                    temporal_analysis["Volatility (CV)"] = f"{volatility:.2f}"
                    
                    # 3. Recent vs Historical Average
                    if len(ts_data) > 8:
                        recent_avg = np.mean(ts_data[-4:])
                        historical_avg = np.mean(ts_data[:-4])
                        recent_vs_hist = (recent_avg - historical_avg) / historical_avg if historical_avg > 0 else 0
                        temporal_analysis["Recent vs Historical"] = f"{recent_vs_hist:.1%}"
                    
                    # 4. Lag Correlation Analysis (what TimeGPT looks at)
                    lag_correlations = {}
                    for lag in [1, 2, 4, 8, 12]:
                        if len(ts_data) > lag:
                            corr = np.corrcoef(ts_data[lag:], ts_data[:-lag])[0, 1]
                            if not np.isnan(corr):
                                lag_correlations[f"Lag-{lag} weeks"] = f"{corr:.3f}"
                    
                    # 5. Seasonal Pattern Detection
                    seasonal_patterns = {}
                    if len(ts_data) > 12:
                        # Check for quarterly patterns
                        quarterly_data = [ts_data[i::13] for i in range(min(13, len(ts_data)))]
                        if len(quarterly_data) > 1 and all(len(q) > 0 for q in quarterly_data):
                            quarterly_vars = [np.var(q) for q in quarterly_data if len(q) > 0]
                            if quarterly_vars:
                                seasonal_strength = np.std(quarterly_vars) / np.mean(quarterly_vars) if np.mean(quarterly_vars) > 0 else 0
                                seasonal_patterns["Seasonal Variation"] = f"{seasonal_strength:.2f}"
                    
                    # Display the real analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📈 Temporal Pattern Analysis:**")
                        for pattern, value in temporal_analysis.items():
                            if "Trend" in pattern:
                                if "Increasing" in str(value):
                                    st.write(f"• **{pattern}**: ↗️ {value}")
                                else:
                                    st.write(f"• **{pattern}**: ↘️ {value}")
                            else:
                                st.write(f"• **{pattern}**: {value}")
                        
                        st.markdown("**🔄 Lag Dependencies (TimeGPT Focus):**")
                        # Sort by correlation strength
                        sorted_lags = sorted(lag_correlations.items(), key=lambda x: abs(float(x[1])), reverse=True)
                        for lag, corr in sorted_lags[:5]:  # Top 5 lags
                            corr_val = float(corr)
                            if abs(corr_val) > 0.5:
                                st.write(f"• **{lag}**: {corr} ⭐ Strong")
                            elif abs(corr_val) > 0.3:
                                st.write(f"• **{lag}**: {corr} 📊 Moderate")
                            else:
                                st.write(f"• **{lag}**: {corr} 📍 Weak")
                    
                    with col2:
                        st.markdown("**🌊 Seasonal Patterns:**")
                        for pattern, value in seasonal_patterns.items():
                            st.write(f"• **{pattern}**: {value}")
                        
                        # Show which patterns are most important for TimeGPT
                        st.markdown("**🎯 Key Drivers for TimeGPT:**")
                        
                        # Determine what TimeGPT is likely focusing on
                        key_drivers = []
                        
                        # Check trend strength
                        if "Trend Strength" in temporal_analysis:
                            trend_val = float(temporal_analysis["Trend Strength"])
                            if trend_val > 0.5:
                                key_drivers.append(f"Strong {temporal_analysis.get('Trend Direction', 'trend')} pattern")
                        
                        # Check strongest lag correlations
                        if sorted_lags:
                            strongest_lag = sorted_lags[0]
                            if abs(float(strongest_lag[1])) > 0.4:
                                key_drivers.append(f"Strong {strongest_lag[0]} autocorrelation")
                        
                        # Check volatility
                        if volatility > 0.3:
                            key_drivers.append("High volatility patterns")
                        elif volatility < 0.1:
                            key_drivers.append("Stable, predictable patterns")
                        
                        # Check recent performance
                        if "Recent vs Historical" in temporal_analysis:
                            recent_change = temporal_analysis["Recent vs Historical"].rstrip('%')
                            try:
                                recent_val = float(recent_change)
                                if abs(recent_val) > 10:
                                    key_drivers.append(f"Recent regime change ({recent_change}%)")
                            except:
                                pass
                        
                        if key_drivers:
                            for driver in key_drivers:
                                st.write(f"• {driver}")
                        else:
                            st.write("• Stable time series patterns")
                            st.write("• Historical average-based forecasting")
                    
                    # Show model performance context
                    st.markdown("---")
                    st.markdown("**🚀 Models Used & Performance Context:**")
                    
                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        for model_name in nixtla_models:
                            if model_name == "TimeGPT":
                                st.write("• **TimeGPT (timegpt-1)**: Foundation model")
                            elif model_name == "Nixtla Statistical":
                                st.write("• **Nixtla Statistical**: Standard config")
                            elif model_name == "Nixtla AutoML":
                                st.write("• **Nixtla AutoML**: Long-horizon config")
                    
                    with perf_col2:
                        # Show data characteristics that affect TimeGPT performance
                        data_characteristics = []
                        if len(ts_data) < 20:
                            data_characteristics.append("Short series → TimeGPT advantage")
                        elif len(ts_data) > 100:
                            data_characteristics.append("Long series → Rich patterns")
                        
                        if volatility < 0.2:
                            data_characteristics.append("Low volatility → Stable forecasts")
                        elif volatility > 0.5:
                            data_characteristics.append("High volatility → Challenging")
                        
                        for char in data_characteristics:
                            st.write(f"• {char}")
                    
                    # Add interpretation
                    st.info("💡 **Interpretation**: This analysis shows the actual temporal patterns in your data that TimeGPT leverages for forecasting, based on the same time series used for training.")
                
            except Exception as e:
                st.error(f"Could not perform temporal analysis: {str(e)}")
                st.write("**🌟 Nixtla Models Used:**")
                for model_name in nixtla_models:
                    st.write(f"• {model_name}")
    
    # Handle other models (ETS)
    other_models = [model for model in model_results.keys() if model in ["ETS"]]
    if other_models:
        with st.expander("🔧 Other Models - Feature Analysis"):
            for model_name in other_models:
                if model_name == "ETS":
                    st.markdown("""
                    **📈 ETS (Exponential Smoothing) Approach:**
                    - Traditional time series method
                    - Focuses on trend and seasonal components
                    - No feature engineering required
                    - Good baseline for seasonal data
                    """)
    
    # Summary of feature importance approach
    st.markdown("---")
    st.markdown("""
    **🎯 Feature Importance Summary:**
    - **Tree Models**: Show which engineered features are most important
    - **Nixtla Models**: Foundation models that consider temporal patterns automatically
    - **ETS**: Traditional time series method, no features used
    """)

    st.info("""
    **📊 Feature Importance Interpretation:**
    - **Tree Models**: Shows how much each feature contributes to reducing prediction error
    - **Higher values** = More important for predictions
    - **Feature Types** help understand what patterns the model learned
    """)

    if not tree_models and not nixtla_models:
        st.info("No traditional ML models were run to show feature importance.") 