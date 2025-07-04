import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from feature_engineering import create_features
from data_preparation import prepare_model_data, prepare_aggregated_data, get_aggregation_level, create_display_name
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
    else:
        return 'Other Features'

def weekly_forecasting(df, selected_company, selected_state, selected_program, models_to_run, test_split_date="2024-10-01", test_end_date="2024-12-31", use_optimized=True):
    """Weekly forecasting with optimized features and models"""
    
    with st.spinner("Creating weekly features..."):
        df_features = create_features(df, optimized=use_optimized)
    
    # Determine aggregation level
    aggregation_level = get_aggregation_level(selected_company, selected_state, selected_program)
    
    st.sidebar.info(f"Aggregation: {aggregation_level}")
    
    # Create display name
    display_name = create_display_name(selected_company, selected_state, selected_program)
    
    st.success(f"**Selected:** {display_name}")
    st.info(f"**Aggregation Level:** {aggregation_level}")
    
    # Check if we're forecasting into the future or evaluating historical data
    data_max_date = df.select(pl.max("week_start")).to_pandas().iloc[0, 0]
    test_start_date = pd.to_datetime(test_split_date)
    # Convert both to date for comparison
    is_future_forecast = test_start_date.date() > data_max_date.date()
    
    if is_future_forecast:
        st.warning(f"🔮 **Future Forecasting Mode**: Test period ({test_split_date}) is beyond available data ({data_max_date})")
        st.info("**Note:** No evaluation metrics will be calculated since we're forecasting into the future")
    else:
        st.info(f"📊 **Historical Evaluation Mode**: Test period is within available data range")
    
    with st.spinner("Running models..."):
        # Prepare data based on aggregation level
        if aggregation_level == "Granular" and selected_company != "ALL COMPANIES":
            X_train, y_train, X_test, y_test, ts_data = prepare_model_data(
                df_features, selected_company, selected_state, selected_program, test_split_date, test_end_date
            )
        else:
            X_train, y_train, X_test, y_test, ts_data = prepare_aggregated_data(
                df_features, selected_company, selected_state, selected_program, aggregation_level, test_split_date, test_end_date
            )
        
        if X_train is None:
            st.error("Insufficient data for the selected combination.")
            return
        
        if is_future_forecast and (y_test is None or len(y_test) == 0):
            # For future forecasting, generate dummy test periods for prediction
            st.info("📅 **Generating future periods for forecasting...**")
            # Calculate future weeks based on specified test end date
            future_weeks = (pd.to_datetime(test_end_date) - pd.to_datetime(test_split_date)).days // 7 + 1
            future_weeks = max(1, future_weeks)  # At least 1 week
            
            # Create dummy X_test for future periods (use last known values)
            if len(X_train) > 0:
                X_test = pd.DataFrame([X_train.iloc[-1]] * future_weeks)
                X_test.index = range(future_weeks)
                y_test = None  # No actual values for future
        
        st.info(f"Training samples: {len(X_train)} | Test/Forecast periods: {len(X_test) if X_test is not None else 0}")
        
        # Perform EDA on the selected data
        eda_df = _get_eda_dataframe(df_features, selected_company, selected_state, selected_program, aggregation_level)
        perform_eda(eda_df, display_name, aggregation_level)
        
        # Display features used in training
        _display_feature_info(X_train)
        
        # Run models
        model_results = run_models(X_train, y_train, X_test, ts_data, models_to_run, use_optimized)
        
        if not model_results:
            st.error("All models failed.")
            return
        
        if is_future_forecast:
            # Future forecasting mode - just show predictions
            st.header("🔮 Weekly Future Forecasting Results")
            st.info("Showing future weekly predictions (no actuals available for comparison)")
            
            # Forecast plot for future predictions
            test_dates = pd.date_range(start=test_split_date, periods=len(X_test), freq="W")
            
            fig = _plot_future_forecast_results(model_results, test_dates)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display future predictions table
            _display_future_forecast_table(model_results, test_dates)
            
        else:
            # Historical evaluation mode - full evaluation
            # Calculate metrics
            model_metrics = {}
            for model_name, result in model_results.items():
                model_metrics[model_name] = calculate_metrics(y_test, result["predictions"])
            
            # Display metrics table
            st.header("📊 Model Performance Metrics")
            
            # Add explanation of metrics
            _display_metrics_explanation()
            
            display_metrics_table(model_metrics, is_monthly=False)
            
            # Best model based on WMAPE
            best_model = min(model_metrics.keys(), key=lambda x: model_metrics[x]["WMAPE"])
            st.success(f"🏆 **Best Model:** {best_model} (WMAPE: {model_metrics[best_model]['WMAPE']:.1f}%)")
            
            # Show both metrics for comparison
            st.info(f"📈 **{best_model} Performance:** MAPE: {model_metrics[best_model]['MAPE']:.1f}% | WMAPE: {model_metrics[best_model]['WMAPE']:.1f}%")
            
            # Forecast plot
            st.header("📈 Forecasting Results")
            test_dates = pd.date_range(start=test_split_date, periods=len(y_test), freq="W")
            
            fig = plot_forecast_results(y_test, model_results, test_dates, "Model Predictions vs Actual")
            st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy comparison
            st.subheader("Model Accuracy Comparison")
            
            # WMAPE comparison (primary metric)
            wmape_fig = plot_metrics_comparison(model_metrics, "WMAPE")
            st.plotly_chart(wmape_fig, use_container_width=True)
            
            # MAPE vs WMAPE comparison
            _plot_mape_vs_wmape_comparison(model_metrics)
            
            # Individual model details
            _display_individual_model_details(model_results, model_metrics, y_test)
        
        # Feature importance for tree models (show in both modes)
        _display_feature_importance(model_results, X_train.columns, use_optimized)

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

def _display_feature_importance(model_results, original_feature_names, use_optimized=True):
    """Display feature importance for tree models"""
    st.header("🎯 Feature Importance")
    
    # Handle Linear Regression first (has different structure)
    if "Linear Regression" in model_results:
        with st.expander("Linear Regression Feature Coefficients"):
            if "feature_names" in model_results["Linear Regression"]:
                model_feature_names = model_results["Linear Regression"]["feature_names"]
                if hasattr(model_results["Linear Regression"]["model"], 'coef_') and len(model_feature_names) > 0:
                    coefficients = np.abs(model_results["Linear Regression"]["model"].coef_)
                    
                    importance_df = pd.DataFrame({
                        'Feature': model_feature_names,
                        'Coefficient': coefficients
                    }).sort_values('Coefficient', ascending=False)
                    
                    # Add feature type
                    importance_df['Type'] = importance_df['Feature'].apply(get_feature_type)
                    
                    # Plot feature coefficients
                    imp_fig = plot_feature_importance(importance_df.rename(columns={'Coefficient': 'Importance'}), 
                                                     "Linear Regression", top_n=15)
                    st.plotly_chart(imp_fig, use_container_width=True)
                    
                    if use_optimized:
                        st.success("✅ Showing importance for optimized features (top 20 selected)")
    
    # Handle tree-based models
    for model_name in ["LightGBM", "XGBoost", "Random Forest"]:
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