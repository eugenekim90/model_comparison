import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def perform_eda(df, display_name, aggregation_level):
    """Perform EDA on the selected data"""
    st.header("📊 Exploratory Data Analysis")
    st.markdown(f"**Analysis for:** {display_name}")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Session Count Statistics")
        session_stats = df.select("session_count").to_pandas()["session_count"].describe()
        
        stats_df = pd.DataFrame({
            "Statistic": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
            "Value": [
                f"{session_stats['count']:.0f}",
                f"{session_stats['mean']:.2f}",
                f"{session_stats['std']:.2f}",
                f"{session_stats['min']:.2f}",
                f"{session_stats['25%']:.2f}",
                f"{session_stats['50%']:.2f}",
                f"{session_stats['75%']:.2f}",
                f"{session_stats['max']:.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.subheader("🔢 Additional Statistics")
        session_data = df.select("session_count").to_pandas()["session_count"]
        
        additional_stats = pd.DataFrame({
            "Metric": ["Zero Sessions", "Non-Zero Sessions", "% Zero Sessions", "Variance", "Skewness", "Kurtosis"],
            "Value": [
                f"{(session_data == 0).sum():.0f}",
                f"{(session_data > 0).sum():.0f}",
                f"{(session_data == 0).mean() * 100:.1f}%",
                f"{session_data.var():.2f}",
                f"{session_data.skew():.2f}",
                f"{session_data.kurtosis():.2f}"
            ]
        })
        st.dataframe(additional_stats, use_container_width=True)
    
    # Time series plot
    st.subheader("📈 Session Count Over Time")
    df_pandas = df.to_pandas()
    df_pandas["week_start"] = pd.to_datetime(df_pandas["week_start"])
    df_pandas = df_pandas.sort_values("week_start")  # Ensure proper sorting
    
    fig = px.line(
        df_pandas,
        x="week_start",
        y="session_count",
        title=f"Session Count by Week - {display_name}",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Session Count",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plot
    st.subheader("📊 Session Count Distribution")
    fig_hist = px.histogram(
        df_pandas, 
        x="session_count", 
        nbins=30,
        title=f"Session Count Distribution - {display_name}"
    )
    fig_hist.update_layout(template='plotly_white')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Seasonal patterns
    st.subheader("🗓️ Seasonal Patterns")
    df_pandas['month'] = df_pandas['week_start'].dt.month
    df_pandas['quarter'] = df_pandas['week_start'].dt.quarter
    df_pandas['year'] = df_pandas['week_start'].dt.year
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly averages
        monthly_avg = df_pandas.groupby('month')['session_count'].mean().reset_index()
        fig_monthly = px.bar(
            monthly_avg, 
            x='month', 
            y='session_count',
            title="Average Sessions by Month"
        )
        fig_monthly.update_layout(template='plotly_white')
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        # Quarterly averages
        quarterly_avg = df_pandas.groupby('quarter')['session_count'].mean().reset_index()
        fig_quarterly = px.bar(
            quarterly_avg, 
            x='quarter', 
            y='session_count',
            title="Average Sessions by Quarter"
        )
        fig_quarterly.update_layout(template='plotly_white')
        st.plotly_chart(fig_quarterly, use_container_width=True)
    
    # Yearly trend
    if len(df_pandas['year'].unique()) > 1:
        st.subheader("📅 Yearly Trend")
        yearly_avg = df_pandas.groupby('year')['session_count'].mean().reset_index()
        fig_yearly = px.line(
            yearly_avg, 
            x='year', 
            y='session_count',
            title="Average Sessions by Year",
            markers=True
        )
        fig_yearly.update_layout(template='plotly_white')
        st.plotly_chart(fig_yearly, use_container_width=True)

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
    """Create feature importance plot"""
    
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
    
    importance_df['Type'] = importance_df['Feature'].apply(get_feature_type)
    
    fig = px.bar(
        importance_df.head(top_n),
        x='Importance',
        y='Feature',
        orientation='h',
        color='Type',
        title=f"Top {top_n} Features - {model_name}",
        color_discrete_sequence=px.colors.qualitative.Set3
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