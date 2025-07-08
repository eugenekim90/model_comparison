import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from feature_engineering import create_features
from data_preparation import prepare_model_data, prepare_aggregated_data, get_aggregation_level, create_display_name, smart_data_preparation, remove_duplicate_columns
from models import run_models, get_model_feature_importance
from metrics import calculate_metrics, calculate_model_metrics
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
    
    from data_preparation import remove_duplicate_columns
    
    # Remove duplicate columns
    df, _ = remove_duplicate_columns(df)
    
    # Apply filters
    filters = []
    if selected_company != "ALL COMPANIES":
        filters.append(pl.col("company") == selected_company)
    if selected_state != "ALL STATES":
        filters.append(pl.col("state") == selected_state)
    if selected_program != "ALL PROGRAMS":
        filters.append(pl.col("program") == selected_program)
    
    filtered_df = df.filter(pl.all_horizontal(filters)) if filters else df
    
    if filtered_df.height == 0:
        st.error("No data available for the selected combination")
        return
    
    # Create features
    df_features = create_features(filtered_df, optimized=use_optimized)
    df_features, _ = remove_duplicate_columns(df_features)
    
    # Prepare data
    if selected_company == "ALL COMPANIES" or selected_state == "ALL STATES" or selected_program == "ALL PROGRAMS":
        aggregation_level = get_aggregation_level(selected_company, selected_state, selected_program)
        X_train, y_train, X_test, y_test, ts_data = prepare_aggregated_data(
            df_features, selected_company, selected_state, selected_program, aggregation_level, test_split_date, test_end_date
        )
    else:
        X_train, y_train, X_test, y_test, ts_data = smart_data_preparation(
            df_features, selected_company, selected_state, selected_program, test_split_date, test_end_date
        )
    
    if X_train is None:
        st.error("Data preparation failed. Try a different combination or adjust the test period.")
        return
    
    # Check if future forecasting
    data_max_date = df_features.select(pl.max("week_start")).item()
    test_start_date = pd.to_datetime(test_split_date)
    
    if isinstance(data_max_date, str):
        data_max_date = pd.to_datetime(data_max_date)
    elif hasattr(data_max_date, 'to_pandas'):
        data_max_date = pd.to_datetime(data_max_date.to_pandas())
    else:
        data_max_date = pd.to_datetime(data_max_date)
    
    is_future_forecast = test_start_date.date() > data_max_date.date()
    display_name = create_display_name(selected_company, selected_state, selected_program)
    
    # Run models
    model_results = run_models(X_train, y_train, X_test, ts_data, models_to_run, use_optimized)
    
    if not model_results:
        st.error("No models completed successfully")
        return
    
    if is_future_forecast:
        # Future forecasting
        forecast_dates = pd.date_range(start=test_start_date, periods=len(y_test), freq='W-MON')
        _display_future_predictions(model_results, forecast_dates, display_name)
    else:
        # Historical evaluation
        model_metrics = calculate_model_metrics(model_results, y_test)
        
        # Show results
        display_metrics_table(model_metrics)
        
        wmape_fig = plot_metrics_comparison(model_metrics, "WMAPE")
        st.plotly_chart(wmape_fig, use_container_width=True)
        
        test_dates = pd.date_range(start=test_start_date, periods=len(y_test), freq='W-MON')
        forecast_fig = plot_forecast_results(y_test, model_results, test_dates)
        if forecast_fig is not None:
            st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Feature importance only for specific selections
        if selected_company != "ALL COMPANIES" and selected_state != "ALL STATES" and selected_program != "ALL PROGRAMS":
            st.subheader("🎯 Feature Importance")
            _display_feature_importance(model_results, X_train.columns, use_optimized, X_train, y_train)


def _display_feature_importance(model_results, original_feature_names, use_optimized=True, X_train=None, y_train=None):
    """Display feature importance analysis"""
    
    # Tree-based models with detailed analysis
    tree_models = [model for model in model_results.keys() if model in ["LightGBM", "XGBoost", "Random Forest"]]
    
    if tree_models:
        st.write("**🌳 Tree-based Models Feature Importance:**")
        
        # Combined feature importance across all tree models
        all_importances = {}
        model_importances = {}
        
        for model_name in tree_models:
            result = model_results[model_name]
            if result.get("model") and hasattr(result["model"], 'feature_importances_'):
                feature_names = result.get("feature_names", original_feature_names)
                importances = result["model"].feature_importances_
                
                # Store individual model importance
                model_importances[model_name] = list(zip(feature_names, importances))
                
                # Aggregate for combined view
                for feature, importance in zip(feature_names, importances):
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
        
        if all_importances:
            # Calculate average importance across models
            avg_importances = {feature: np.mean(scores) for feature, scores in all_importances.items()}
            
            # Sort by average importance
            sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
            top_15_features = sorted_features[:15]
            
            # Create feature importance plot
            feature_names = [f[0] for f in top_15_features]
            importance_values = [f[1] for f in top_15_features]
            
            fig = px.bar(
                x=importance_values,
                y=feature_names,
                orientation='h',
                title="Top 15 Features - Average Importance Across Tree Models",
                labels={'x': 'Average Feature Importance', 'y': 'Features'}
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature categories analysis
            st.subheader("📊 Feature Categories Breakdown")
            
            # Categorize features
            feature_categories = {}
            for feature, importance in top_15_features:
                category = get_feature_type(feature)
                if category not in feature_categories:
                    feature_categories[category] = []
                feature_categories[category].append((feature, importance))
            
            # Display by category
            cols = st.columns(2)
            for idx, (category, features) in enumerate(feature_categories.items()):
                with cols[idx % 2]:
                    with st.expander(f"🏷️ {category} ({len(features)} features)"):
                        for feature, importance in features:
                            st.write(f"• **{feature}:** {importance:.3f}")
            
            # Individual model comparisons
            st.subheader("🔍 Individual Model Feature Rankings")
            
            for model_name in tree_models:
                if model_name in model_importances:
                    feature_importance = sorted(model_importances[model_name], key=lambda x: x[1], reverse=True)
                    top_10 = feature_importance[:10]
                    
                    with st.expander(f"🎯 {model_name} - Top 10 Features"):
                        for rank, (feature, importance) in enumerate(top_10, 1):
                            st.write(f"{rank}. **{feature}:** {importance:.3f}")
    
    # Statistical and Nixtla models
    other_models = [model for model in model_results.keys() if model not in tree_models]
    if other_models:
        st.subheader("🤖 Statistical & Foundation Models")
        
        model_info = {
            "TimeGPT": "Foundation model trained on diverse time series data. Uses transformer architecture to learn complex temporal patterns.",
            "Nixtla AutoML": "AutoML approach that automatically selects and ensembles multiple forecasting models.",
            "ETS": "Exponential smoothing model capturing error, trend, and seasonality components.",
            "Nixtla Statistical": "Statistical ensemble combining multiple traditional forecasting methods."
        }
        
        for model_name in other_models:
            if model_name in model_info:
                st.write(f"**{model_name}:** {model_info[model_name]}")
            else:
                st.write(f"**{model_name}:** Advanced forecasting model")
        
        st.info("💡 These models use internal feature selection and don't provide explicit feature importance scores.")
    
    # Feature engineering summary for context
    if use_optimized:
        st.subheader("⚙️ Feature Engineering Summary")
        with st.expander("🔧 Applied Feature Transformations"):
            st.markdown("""
            **Advanced Features Used:**
            - **Lag Features**: Historical values (1-4 weeks back)
            - **Rolling Statistics**: Moving averages, volatility measures
            - **Seasonal Features**: Holiday effects, quarterly patterns
            - **Trend Features**: Growth rates, momentum indicators  
            - **Categorical Encoding**: Company, state, program embeddings
            - **Time Features**: Week/month/quarter cyclical patterns
            
            *These optimized features enable models to capture complex temporal relationships.*
            """)
    else:
        st.subheader("⚙️ Feature Engineering Summary")
        with st.expander("🔧 Basic Features Used"):
            st.markdown("""
            **Standard Features Used:**
            - **Core Lag Features**: 1-2 week historical values
            - **Basic Time Features**: Week of year, month, quarter
            - **Simple Seasonality**: Holiday indicators
            - **Categorical Features**: Encoded company/state/program
            
            *Basic feature set for stable, interpretable forecasting.*
            """)
    
    # Feature statistics
    if X_train is not None:
        st.subheader("📈 Training Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Samples", f"{len(X_train):,}")
        with col2:
            st.metric("Features Used", f"{len(X_train.columns):,}")
        with col3:
            if y_train is not None:
                st.metric("Avg Target Value", f"{np.mean(y_train):.1f}")
            else:
                st.metric("Target", "Not Available")


def _display_future_predictions(model_results, forecast_dates, display_name):
    """Display future predictions table"""
    st.subheader(f"📋 Future Predictions - {display_name}")
    
    # Filter out models with None predictions or errors
    valid_results = {}
    for model_name, result in model_results.items():
        if result["predictions"] is not None and "error" not in result:
            valid_results[model_name] = result
    
    if not valid_results:
        st.warning("⚠️ No valid predictions available. Check API keys and model configurations.")
        return
    
    # Create predictions table
    predictions_data = {"Date": forecast_dates}
    for model_name, result in valid_results.items():
        predictions_data[model_name] = result["predictions"]
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df["Date"] = predictions_df["Date"].dt.strftime("%Y-%m-%d")
    
    st.dataframe(predictions_df, use_container_width=True, hide_index=True)
    
    # Plot predictions
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    
    for i, (model_name, result) in enumerate(valid_results.items()):
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=result["predictions"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=3)
            )
        )
    
    fig.update_layout(
        title=f"Future Weekly Predictions - {display_name}",
        xaxis_title="Date",
        yaxis_title="Predicted Values",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True) 