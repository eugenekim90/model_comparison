import polars as pl
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from feature_engineering import create_features
from data_preparation import prepare_model_data, prepare_aggregated_data, get_aggregation_level, create_display_name
from models import run_models
from metrics import calculate_metrics
from visualization import display_metrics_table, plot_metrics_comparison

def monthly_forecasting(df, selected_company, selected_state, selected_program, models_to_run, test_split_date="2024-10-01", test_end_date="2024-12-31", use_optimized=True):
    """Monthly evaluation approach: Weekly models → Aggregate weekly predictions → Monthly evaluation"""
    
    # Create display name
    display_name = create_display_name(selected_company, selected_state, selected_program)
    
    st.success(f"**Selected (Monthly):** {display_name}")
    st.info("**Approach:** Weekly models → Aggregate weekly predictions → Monthly evaluation")
    
    with st.spinner("Creating weekly features for monthly evaluation..."):
        df_features = create_features(df, optimized=use_optimized)
    
    # Determine aggregation level (same logic as weekly)
    aggregation_level = get_aggregation_level(selected_company, selected_state, selected_program)
    
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
    
    with st.spinner("Running weekly models..."):
        # Prepare data using weekly approach
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
            
        st.info(f"Weekly Training samples: {len(X_train)} | Weekly Test/Forecast periods: {len(X_test) if X_test is not None else 0}")
        
        # Run weekly models
        weekly_model_results = run_models(X_train, y_train, X_test, ts_data, models_to_run, use_optimized)
        
        if not weekly_model_results:
            st.error("All weekly models failed.")
            return
        
        # Get weekly forecast dates
        weekly_forecast_dates = pd.date_range(start=test_split_date, periods=len(X_test), freq="W")
        
        if is_future_forecast:
            # Future forecasting mode - just show predictions
            st.header("🔮 Monthly Future Forecasting Results")
            st.info("Showing future monthly predictions (no actuals available for comparison)")
            
            # Aggregate weekly predictions to monthly level
            monthly_predictions = {}
            for model_name, result in weekly_model_results.items():
                # Create dataframe with weekly predictions
                weekly_df = pd.DataFrame({
                    'date': weekly_forecast_dates,
                    'prediction': result["predictions"]
                })
                weekly_df['year_month'] = weekly_df['date'].dt.to_period('M')
                
                # Aggregate to monthly level
                monthly_df = weekly_df.groupby('year_month').agg({
                    'prediction': 'sum'
                }).reset_index()
                monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
                
                monthly_predictions[model_name] = {
                    'dates': monthly_df['date'].values,
                    'predictions': monthly_df['prediction'].values
                }
            
            # Plot future monthly predictions
            fig = _plot_future_monthly_results(monthly_predictions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction table
            _display_future_predictions_table(monthly_predictions)
            
        else:
            # Historical evaluation mode - full evaluation
            # Aggregate weekly predictions and actuals to monthly level
            monthly_results = _aggregate_to_monthly(weekly_model_results, y_test, weekly_forecast_dates)
            
            if monthly_results is None:
                st.error("Could not aggregate to monthly level.")
                return
            
            monthly_actuals = monthly_results["monthly_actuals"]
            monthly_predictions = monthly_results["monthly_predictions"]
            monthly_dates = monthly_results["monthly_dates"]
            
            # Calculate monthly metrics
            model_metrics = {}
            for model_name, monthly_pred in monthly_predictions.items():
                model_metrics[model_name] = calculate_metrics(monthly_actuals, monthly_pred)
            
            # Display monthly metrics table
            st.header("📊 Monthly Model Performance Metrics")
            st.info("Metrics calculated on monthly aggregated predictions vs monthly aggregated actuals")
            
            display_metrics_table(model_metrics, is_monthly=True)
            
            # Best monthly model
            best_model = min(model_metrics.keys(), key=lambda x: model_metrics[x]["WMAPE"])
            st.success(f"🏆 **Best Model (Monthly Evaluation):** {best_model} (WMAPE: {model_metrics[best_model]['WMAPE']:.1f}%)")
            
            # Monthly forecast plot
            st.header("📈 Monthly Evaluation Results")
            st.caption("Weekly predictions aggregated to monthly level vs true monthly actuals")
            
            fig = _plot_monthly_results(monthly_actuals, monthly_predictions, monthly_dates)
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly accuracy comparison
            st.subheader("Monthly Model Accuracy Comparison")
            
            # WMAPE comparison (primary metric)
            wmape_fig = plot_metrics_comparison(model_metrics, "WMAPE")
            st.plotly_chart(wmape_fig, use_container_width=True)
            
            # Show weekly vs monthly comparison
            _display_aggregation_info(len(y_test), len(monthly_actuals), weekly_forecast_dates, monthly_dates)

def _aggregate_to_monthly(weekly_model_results, weekly_actuals, weekly_dates):
    """Aggregate weekly predictions and actuals to monthly level"""
    
    # Create dataframe with weekly data
    weekly_df = pd.DataFrame({
        'date': weekly_dates,
        'actual': weekly_actuals
    })
    
    # Add weekly predictions
    for model_name, result in weekly_model_results.items():
        weekly_df[f'pred_{model_name}'] = result["predictions"]
    
    # Add year-month column
    weekly_df['year_month'] = weekly_df['date'].dt.to_period('M')
    
    # Aggregate to monthly level (sum all weeks within each month)
    monthly_df = weekly_df.groupby('year_month').agg({
        'actual': 'sum',
        **{f'pred_{model_name}': 'sum' for model_name in weekly_model_results.keys()}
    }).reset_index()
    
    # Convert period back to datetime for plotting
    monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
    
    if len(monthly_df) == 0:
        return None
    
    # Extract results
    monthly_actuals = monthly_df['actual'].values
    monthly_predictions = {}
    for model_name in weekly_model_results.keys():
        monthly_predictions[model_name] = monthly_df[f'pred_{model_name}'].values
    
    monthly_dates = monthly_df['date'].values
    
    return {
        "monthly_actuals": monthly_actuals,
        "monthly_predictions": monthly_predictions,
        "monthly_dates": monthly_dates
    }

def _plot_monthly_results(monthly_actuals, monthly_predictions, monthly_dates):
    """Plot monthly evaluation results"""
    
    fig = go.Figure()
    
    # Add actual monthly values
    fig.add_trace(go.Scatter(
        x=monthly_dates,
        y=monthly_actuals,
        mode='lines+markers',
        name='Actual Monthly',
        line=dict(color='black', width=3),
        marker=dict(size=10)
    ))
    
    # Add monthly model predictions (aggregated from weekly)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model_name, monthly_pred) in enumerate(monthly_predictions.items()):
        fig.add_trace(go.Scatter(
            x=monthly_dates,
            y=monthly_pred,
            mode='lines+markers',
            name=f'{model_name} (Monthly)',
            line=dict(color=colors[i % len(colors)], dash='dash'),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Monthly Evaluation: Weekly Predictions Aggregated vs Monthly Actuals",
        xaxis_title="Month",
        yaxis_title="Monthly Session Count",
        template='plotly_white',
        height=500
    )
    
    return fig

def _display_aggregation_info(weekly_count, monthly_count, weekly_dates, monthly_dates):
    """Display information about the aggregation process"""
    
    st.subheader("📅 Aggregation Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Weekly Test Periods", weekly_count)
        st.caption(f"From {weekly_dates[0].strftime('%Y-%m-%d')} to {weekly_dates[-1].strftime('%Y-%m-%d')}")
    
    with col2:
        st.metric("Monthly Test Periods", monthly_count)
        if len(monthly_dates) > 0:
            st.caption(f"From {pd.to_datetime(monthly_dates[0]).strftime('%Y-%m')} to {pd.to_datetime(monthly_dates[-1]).strftime('%Y-%m')}")
    
    st.info("""
    **Aggregation Process:**
    1. ✅ Train models on weekly data
    2. ✅ Get weekly predictions for test period
    3. ✅ Sum weekly predictions within each month
    4. ✅ Sum weekly actuals within each month  
    5. ✅ Evaluate monthly predictions vs monthly actuals
    """)

def _plot_future_monthly_results(monthly_predictions):
    """Plot future monthly predictions (no actuals)"""
    
    fig = go.Figure()
    
    # Add monthly model predictions
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model_name, pred_data) in enumerate(monthly_predictions.items()):
        fig.add_trace(go.Scatter(
            x=pred_data['dates'],
            y=pred_data['predictions'],
            mode='lines+markers',
            name=f'{model_name} (Future)',
            line=dict(color=colors[i % len(colors)]),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Future Monthly Predictions (No Actuals Available)",
        xaxis_title="Month",
        yaxis_title="Predicted Monthly Session Count",
        template='plotly_white',
        height=500
    )
    
    return fig

def _display_future_predictions_table(monthly_predictions):
    """Display future predictions in a table format"""
    
    st.subheader("📋 Future Monthly Predictions Table")
    
    # Get all dates (assume all models predict for same periods)
    first_model = list(monthly_predictions.keys())[0]
    dates = monthly_predictions[first_model]['dates']
    
    # Create table data
    table_data = []
    for i, date in enumerate(dates):
        row = {"Month": pd.to_datetime(date).strftime('%Y-%m')}
        for model_name, pred_data in monthly_predictions.items():
            row[model_name] = f"{pred_data['predictions'][i]:.0f}"
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True) 