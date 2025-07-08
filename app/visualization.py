import streamlit as st
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def perform_eda(df, display_name, aggregation_level):
    """Ultra minimal EDA - stats only, no plotting"""
    
    st.header("📊 EDA")
    st.markdown(f"**Analysis:** {display_name}")
    
    try:
        # Quick data check
        if df.height == 0:
            st.error("No data available")
            return
        
        # Get target column
        target_col = 'attended_sessions' if 'attended_sessions' in df.columns else 'session_count'
        
        if target_col not in df.columns:
            st.error(f"Column '{target_col}' not found")
            return
        
        # Option to use Nixtla analysis or basic stats
        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["Basic Stats", "Nixtla Time Series Analysis"],
            horizontal=True
        )
        
        if analysis_type == "Basic Stats":
            # Just basic stats - no plotting
            with st.spinner("Loading stats..."):
                stats = df.select([
                    pl.count(target_col).alias("count"),
                    pl.sum(target_col).alias("total"),
                    pl.mean(target_col).alias("mean"),
                    pl.min(target_col).alias("min"),
                    pl.max(target_col).alias("max")
                ]).to_pandas().iloc[0]
            
            # Show basic info only
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Records", f"{stats['count']:,}")
            with col2:
                st.metric("Total Sessions", f"{stats['total']:,.0f}")
            with col3:
                st.metric("Avg Sessions", f"{stats['mean']:.1f}")
            with col4:
                st.metric("Max Sessions", f"{stats['max']:,.0f}")
            
            # Text summary only - no plots
            st.subheader("📊 Summary")
            
            st.write(f"**Dataset:** {stats['count']:,} records")
            st.write(f"**Total Sessions:** {stats['total']:,.0f}")
            st.write(f"**Average per Week:** {stats['mean']:.1f}")
            st.write(f"**Range:** {stats['min']:.0f} to {stats['max']:.0f}")
            
            # Date range if available
            try:
                date_stats = df.select([
                    pl.min("week_start").alias("start_date"),
                    pl.max("week_start").alias("end_date")
                ]).to_pandas().iloc[0]
                
                st.write(f"**Date Range:** {date_stats['start_date']} to {date_stats['end_date']}")
            except:
                pass
        
        else:
            # Nixtla Time Series Analysis
            api_key = st.session_state.get('nixtla_api_key', '')
            if not api_key:
                st.warning("⚠️ Nixtla API key required for time series analysis. Please enter your API key in the sidebar.")
            else:
                try:
                    from nixtla import NixtlaClient
                    
                    with st.spinner("Running Nixtla time series analysis..."):
                        # Prepare data for Nixtla
                        ts_df = df.select([
                            pl.col("week_start").alias("ds"),
                            pl.col(target_col).alias("y")
                        ]).to_pandas()
                        ts_df['unique_id'] = 'series_1'
                        ts_df = ts_df[['unique_id', 'ds', 'y']]
                        
                        # Initialize client
                        client = NixtlaClient(api_key=api_key)
                        
                        # Run descriptive analysis
                        summary = client.describe(ts_df)
                        st.subheader("📊 Nixtla Time Series Summary")
                        st.dataframe(summary, use_container_width=True)
                        
                        # Plot using Nixtla
                        fig = client.plot(ts_df, engine='plotly')
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Nixtla analysis failed: {str(e)}")
                    st.info("Falling back to basic stats...")
                    # Fall back to basic analysis
                    perform_basic_stats(df, target_col)
        
        st.success("✅ EDA completed!")
        
    except Exception as e:
        st.error(f"EDA Error: {str(e)}")

def perform_basic_stats(df, target_col):
    """Helper function for basic stats"""
    stats = df.select([
        pl.count(target_col).alias("count"),
        pl.sum(target_col).alias("total"),
        pl.mean(target_col).alias("mean"),
        pl.min(target_col).alias("min"),
        pl.max(target_col).alias("max")
    ]).to_pandas().iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{stats['count']:,}")
    with col2:
        st.metric("Total Sessions", f"{stats['total']:,.0f}")
    with col3:
        st.metric("Avg Sessions", f"{stats['mean']:.1f}")
    with col4:
        st.metric("Max Sessions", f"{stats['max']:,.0f}")

def perform_comprehensive_eda(df, display_name, aggregation_level):
    """Backward compatibility - just call simple EDA"""
    return perform_eda(df, display_name, aggregation_level)

def plot_forecast_results(y_test, model_results, test_dates, title="Model Predictions vs Actual"):
    """Create forecast visualization"""
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test,
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=3)
    ))
    
    # Add model predictions
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model_name, result) in enumerate(model_results.items()):
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=result["predictions"],
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors[i % len(colors)], dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Session Count",
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_metrics_comparison(model_metrics, metric_name="WMAPE"):
    """Create metrics comparison bar chart"""
    comparison_data = []
    for model_name in model_metrics.keys():
        comparison_data.append({
            "Model": model_name,
            metric_name: model_metrics[model_name][metric_name]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comparison_df,
        x="Model",
        y=metric_name,
        title=f"{metric_name} Comparison",
        labels={metric_name: f"{metric_name} (%)" if metric_name in ["MAPE", "WMAPE"] else metric_name, 
                "Model": "Models"},
        color=metric_name,
        color_continuous_scale="RdYlBu_r"
    )
    
    return fig

def plot_feature_importance(importance_df, model_name, top_n=15):
    """Create feature importance plot with robust handling of extreme values"""
    
    # Add feature type for better visualization
    def get_feature_type(feature_name):
        if feature_name.startswith('lag_'):
            return 'Lag Features'
        elif feature_name.startswith('rolling_'):
            return 'Rolling Statistics'
        elif feature_name.endswith('_encoded') or feature_name.startswith('config_'):
            return 'Categorical Features'
        elif feature_name in ['week_of_year', 'month', 'quarter', 'is_winter', 'is_spring', 'is_summer', 'is_fall']:
            return 'Time Features'
        else:
            return 'Other Features'
    
    # Robust handling of extreme values
    if 'Importance' in importance_df.columns:
        # Replace infinite values
        importance_df = importance_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Cap extreme values at 95th percentile to prevent chart distortion
        if len(importance_df) > 3:  # Only cap if we have enough data points
            cap_value = importance_df['Importance'].quantile(0.95)
            max_reasonable = cap_value * 2  # Allow some headroom
            importance_df['Importance'] = importance_df['Importance'].clip(upper=max_reasonable)
    
    importance_df['Type'] = importance_df['Feature'].apply(get_feature_type)
    
    # Take top features after capping
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Type',
        title=f"Top {top_n} Features - {model_name}",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Check if we capped any values and add a note
    if len(importance_df) > 3:
        cap_value = importance_df['Importance'].quantile(0.95) * 2
        max_value = top_features['Importance'].max()
        if max_value >= cap_value * 0.95:  # Close to cap
            fig.add_annotation(
                text="Note: Extreme values capped for better visualization",
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                showarrow=False,
                font=dict(size=10, color="gray")
            )
    
    return fig

def plot_prediction_scatter(y_true, y_pred, model_name):
    """Create predicted vs actual scatter plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{model_name}: Predicted vs Actual",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template='plotly_white'
    )
    
    return fig

def display_metrics_table(model_metrics, is_monthly=False):
    """Display metrics in a formatted table"""
    metrics_data = []
    for model_name, metrics in model_metrics.items():
        if is_monthly:
            metrics_data.append({
                "Model": model_name,
                "MAE": f"{metrics['MAE']:.0f}",  # Monthly values are larger
                "RMSE": f"{metrics['RMSE']:.0f}",
                "MAPE": f"{metrics['MAPE']:.1f}%",
                "WMAPE": f"{metrics['WMAPE']:.1f}%"
            })
        else:
            metrics_data.append({
                "Model": model_name,
                "MAE": f"{metrics['MAE']:.2f}",
                "RMSE": f"{metrics['RMSE']:.2f}",
                "MAPE": f"{metrics['MAPE']:.1f}%",
                "WMAPE": f"{metrics['WMAPE']:.1f}%"
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    return metrics_df 