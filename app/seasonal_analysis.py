import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def seasonal_analysis_dashboard(df):
    """Seasonal analysis dashboard"""
    
    st.header("🌊 Seasonal Analysis")
    st.info("Analyzing seasonal patterns, dips, and temporal behavior")
    
    # Prepare aggregated data for analysis
    df_agg = df.group_by("week_start").agg([
        pl.sum("session_count").alias("total_sessions"),
        pl.sum("subscriber_count").alias("total_subscribers"), 
        pl.sum("non_subscriber_count").alias("total_non_subscribers")
    ]).sort("week_start")
    
    # Convert to pandas for time series analysis
    ts_df = df_agg.to_pandas()
    ts_df['week_start'] = pd.to_datetime(ts_df['week_start'])
    ts_df = ts_df.set_index('week_start')
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "📊 Pattern Analysis", 
        "🔍 Dip Analysis", 
        "🎯 Features"
    ])
    
    with tab1:
        pattern_decomposition_analysis(ts_df)
    
    with tab2:
        dip_analysis(ts_df)
    
    with tab3:
        features_analysis(ts_df, df)

def pattern_decomposition_analysis(ts_df):
    """Analyze time series patterns using decomposition"""
    
    st.subheader("🔍 Pattern Decomposition Analysis")
    
    # Check data sufficiency for decomposition
    data_length = len(ts_df)
    
    # Adaptive period selection based on data length
    if data_length >= 104:  # 2 complete years
        period = 52
        period_name = "yearly"
    elif data_length >= 52:  # 1 complete year
        period = 26  # Semi-annual
        period_name = "semi-annual"
    elif data_length >= 26:  # Half year
        period = 13  # Quarterly
        period_name = "quarterly"
    elif data_length >= 12:  # Quarter
        period = 6   # Bi-monthly
        period_name = "bi-monthly"
    else:
        st.warning(f"⚠️ Insufficient data for reliable seasonal decomposition ({data_length} observations). Minimum 12 observations required.")
        return
    
    st.info(f"📊 Using {period_name} seasonality (period={period}) based on {data_length} data points")
    
    # Perform decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Handle missing values
    ts_clean = ts_df['total_sessions'].fillna(method='ffill').fillna(method='bfill')
    
    try:
        # Seasonal decomposition with adaptive period
        decomposition = seasonal_decompose(ts_clean, model='additive', period=period)
        
        # Create decomposition plot
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08
        )
        
        # Original
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['total_sessions'], 
                                name='Original', line=dict(color='blue')), row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.trend, 
                                name='Trend', line=dict(color='red')), row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.seasonal, 
                                name='Seasonal', line=dict(color='green')), row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.resid, 
                                name='Residual', line=dict(color='orange')), row=4, col=1)
        
        fig.update_layout(height=800, title=f"Time Series Decomposition ({period_name} seasonality)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except ValueError as e:
        st.error(f"❌ Decomposition failed: {str(e)}")
        st.warning(f"Data may be too short or irregular for period={period}. Showing basic analysis instead.")
    
    # Seasonal pattern analysis (always show this regardless of decomposition success)
    st.subheader("🔄 Seasonal Pattern Analysis")
    
    # Extract seasonal patterns by month
    ts_df_copy = ts_df.copy()
    ts_df_copy['month'] = ts_df_copy.index.month
    ts_df_copy['year'] = ts_df_copy.index.year
    ts_df_copy['week_of_year'] = ts_df_copy.index.isocalendar().week
    
    # Monthly aggregation
    monthly_pattern = ts_df_copy.groupby('month')['total_sessions'].agg(['mean', 'std', 'median'])
    
    # Weekly pattern within year
    weekly_pattern = ts_df_copy.groupby('week_of_year')['total_sessions'].agg(['mean', 'std', 'median'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly seasonality
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_pattern.index, y=monthly_pattern['mean'],
                                mode='lines+markers', name='Mean',
                                error_y=dict(type='data', array=monthly_pattern['std'])))
        fig.update_layout(title="Average Sessions by Month", xaxis_title="Month", yaxis_title="Sessions")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weekly seasonality
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly_pattern.index, y=weekly_pattern['mean'],
                                mode='lines+markers', name='Mean',
                                error_y=dict(type='data', array=weekly_pattern['std'])))
        fig.update_layout(title="Average Sessions by Week of Year", xaxis_title="Week", yaxis_title="Sessions")
        st.plotly_chart(fig, use_container_width=True)

def dip_analysis(ts_df):
    """Analyze significant dips in the data"""
    
    st.subheader("🔍 Dip Detection & Analysis")
    
    # Calculate percentage changes
    ts_df_copy = ts_df.copy()
    ts_df_copy['pct_change'] = ts_df_copy['total_sessions'].pct_change()
    ts_df_copy['rolling_mean'] = ts_df_copy['total_sessions'].rolling(window=4).mean()
    ts_df_copy['deviation'] = (ts_df_copy['total_sessions'] - ts_df_copy['rolling_mean']) / ts_df_copy['rolling_mean']
    
    # Identify significant dips (more than 30% below rolling average)
    dip_threshold = -0.3
    significant_dips = ts_df_copy[ts_df_copy['deviation'] < dip_threshold].copy()
    
    st.info(f"Found {len(significant_dips)} significant dips (>{abs(dip_threshold)*100}% below 4-week average)")
    
    # Plot dips
    fig = go.Figure()
    
    # Original data
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['total_sessions'],
                            mode='lines', name='Sessions', line=dict(color='blue')))
    
    # Rolling average
    fig.add_trace(go.Scatter(x=ts_df_copy.index, y=ts_df_copy['rolling_mean'],
                            mode='lines', name='4-Week Average', line=dict(color='green', dash='dash')))
    
    # Highlight dips
    if len(significant_dips) > 0:
        fig.add_trace(go.Scatter(x=significant_dips.index, y=significant_dips['total_sessions'],
                                mode='markers', name='Significant Dips', 
                                marker=dict(color='red', size=8)))
    
    fig.update_layout(title="Significant Dips Detection", xaxis_title="Date", yaxis_title="Sessions")
    st.plotly_chart(fig, use_container_width=True)
    
    if len(significant_dips) > 0:
        # Analyze dip characteristics
        st.subheader("📋 Dip Characteristics")
        
        # Add temporal features to dips
        significant_dips['month'] = significant_dips.index.month
        significant_dips['week_of_year'] = significant_dips.index.isocalendar().week
        significant_dips['year'] = significant_dips.index.year
        significant_dips['quarter'] = significant_dips.index.quarter
        
        # Dip timing analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly distribution of dips
            monthly_dips = significant_dips['month'].value_counts().sort_index()
            fig = px.bar(x=monthly_dips.index, y=monthly_dips.values, 
                        title="Dips by Month", labels={'x': 'Month', 'y': 'Number of Dips'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quarterly distribution
            quarterly_dips = significant_dips['quarter'].value_counts().sort_index()
            fig = px.bar(x=quarterly_dips.index, y=quarterly_dips.values,
                        title="Dips by Quarter", labels={'x': 'Quarter', 'y': 'Number of Dips'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Severity analysis
        st.subheader("⚡ Dip Severity Analysis")
        significant_dips['severity'] = significant_dips['deviation'].abs()
        
        # Create severity categories
        significant_dips['severity_category'] = pd.cut(significant_dips['severity'], 
                                                     bins=[0, 0.4, 0.6, 1.0], 
                                                     labels=['Moderate', 'Severe', 'Extreme'])
        
        severity_counts = significant_dips['severity_category'].value_counts()
        fig = px.pie(values=severity_counts.values, names=severity_counts.index, 
                    title="Dip Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display dip details
        st.subheader("📊 Dip Details")
        dip_details = significant_dips[['total_sessions', 'deviation', 'severity_category', 'month', 'quarter']].copy()
        dip_details['deviation'] = (dip_details['deviation'] * 100).round(1)
        dip_details.columns = ['Sessions', 'Deviation %', 'Severity', 'Month', 'Quarter']
        st.dataframe(dip_details.sort_values('Deviation %'))

def features_analysis(ts_df, original_df):
    """Create and analyze seasonal features"""
    
    st.subheader("🎯 Seasonal Features")
    
    # Create features
    features_df = create_seasonal_features(ts_df, original_df)
    
    st.success(f"Created {len(features_df.columns) - 4} seasonal features!")
    
    # Feature categories
    st.subheader("📊 Feature Categories Created")
    
    feature_categories = {
        'Seasonal Harmonics': [col for col in features_df.columns if 'seasonal_sin' in col or 'seasonal_cos' in col],
        'Dip Indicators': [col for col in features_df.columns if 'dip' in col or 'recovery' in col or 'deviation' in col],
        'Trend Features': [col for col in features_df.columns if 'trend' in col or 'momentum' in col],
        'Volatility Features': [col for col in features_df.columns if 'volatility' in col or 'stability' in col],
        'Historical Patterns': [col for col in features_df.columns if 'historical' in col or 'yoy' in col],
        'Calendar Features': [col for col in features_df.columns if any(x in col for x in ['month_end', 'quarter_end', 'year_end', 'week_of_year', 'month'])]
    }
    
    for category, features in feature_categories.items():
        if features:
            st.write(f"**{category}:** {len(features)} features")
            with st.expander(f"View {category} Features"):
                for feature in features:
                    st.write(f"• {feature}")
    
    # Feature correlation analysis
    st.subheader("🔗 Feature Correlation with Target")
    
    # Calculate correlations with target
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    target_corr = features_df[numeric_features].corrwith(features_df['total_sessions']).abs().sort_values(ascending=False)
    target_corr = target_corr[target_corr.index != 'total_sessions'].head(20)
    
    fig = px.bar(x=target_corr.values, y=target_corr.index, orientation='h',
                title="Top 20 Features by Correlation with Sessions")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def create_seasonal_features(ts_df, original_df):
    """Create seasonal features"""
    
    features_df = ts_df.copy()
    features_df = features_df.reset_index()
    
    # 1. Harmonic Features (capture multiple seasonal frequencies)
    for period in [13, 26, 52]:  # quarterly, bi-annual, annual
        for harmonic in [1, 2, 3]:  # multiple harmonics
            angle = 2 * np.pi * harmonic * np.arange(len(features_df)) / period
            features_df[f'seasonal_sin_{period}_{harmonic}'] = np.sin(angle)
            features_df[f'seasonal_cos_{period}_{harmonic}'] = np.cos(angle)
    
    # 2. Historical Dip Indicators
    features_df['rolling_mean_4w'] = features_df['total_sessions'].rolling(4).mean()
    features_df['deviation_from_trend'] = (features_df['total_sessions'] - features_df['rolling_mean_4w']) / features_df['rolling_mean_4w']
    features_df['week_of_year'] = features_df['week_start'].dt.isocalendar().week
    features_df['month'] = features_df['week_start'].dt.month
    
    # Calculate historical dip probability for each week
    dip_probs = {}
    for week in range(1, 54):
        week_data = features_df[features_df['week_of_year'] == week]
        if len(week_data) > 0:
            dip_count = len(week_data[week_data['deviation_from_trend'] < -0.2])
            total_count = len(week_data)
            dip_probs[week] = dip_count / total_count if total_count > 0 else 0
    
    features_df['historical_dip_probability'] = features_df['week_of_year'].map(dip_probs)
    
    # 3. Recovery Indicators
    features_df['sessions_lag1'] = features_df['total_sessions'].shift(1)
    features_df['sessions_lag2'] = features_df['total_sessions'].shift(2)
    features_df['recovery_momentum'] = (
        (features_df['total_sessions'] - features_df['sessions_lag1']) +
        (features_df['sessions_lag1'] - features_df['sessions_lag2'])
    ) / 2
    
    # 4. Volatility and Stability Features
    for window in [4, 8, 12]:
        features_df[f'volatility_{window}w'] = features_df['total_sessions'].rolling(window).std()
        features_df[f'stability_score_{window}w'] = 1 / (1 + features_df[f'volatility_{window}w'])
    
    # 5. Trend Features
    for window in [8, 12, 26]:
        features_df[f'trend_slope_{window}w'] = features_df['total_sessions'].rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
        )
    
    # 6. Year-over-Year Features
    features_df['yoy_growth'] = features_df['total_sessions'].pct_change(periods=52) * 100
    features_df['yoy_acceleration'] = features_df['yoy_growth'].diff()
    
    # 7. Business Calendar Features
    features_df['is_month_end'] = (features_df['week_start'].dt.day >= 25).astype(int)
    features_df['is_quarter_end'] = features_df['week_start'].dt.month.isin([3, 6, 9, 12]).astype(int)
    features_df['is_year_end'] = features_df['week_start'].dt.month.isin([11, 12]).astype(int)
    
    # 8. Holiday and Special Period Features
    features_df['is_holiday_season'] = features_df['week_start'].dt.month.isin([11, 12]).astype(int)
    features_df['is_back_to_school'] = features_df['week_start'].dt.month.isin([8, 9]).astype(int)
    features_df['is_summer_break'] = features_df['week_start'].dt.month.isin([6, 7]).astype(int)
    features_df['is_new_year'] = (features_df['week_start'].dt.month == 1).astype(int)
    
    # 9. Seasonal Strength and Phase Features
    features_df['seasonal_strength'] = features_df['total_sessions'].rolling(52).apply(
        lambda x: np.std(x) / np.mean(x) if len(x) == 52 and np.mean(x) != 0 else np.nan
    )
    
    features_df['annual_phase'] = (features_df['week_of_year'] - 1) / 51 * 2 * np.pi
    features_df['seasonal_phase_sin'] = np.sin(features_df['annual_phase'])
    features_df['seasonal_phase_cos'] = np.cos(features_df['annual_phase'])
    
    # 10. Advanced Lag Features (capture multi-year patterns)
    for lag in [26, 52, 78, 104]:  # 0.5, 1, 1.5, 2 years
        features_df[f'sessions_lag_{lag}w'] = features_df['total_sessions'].shift(lag)
        features_df[f'sessions_change_{lag}w'] = features_df['total_sessions'].pct_change(periods=lag)
    
    # Fill NaN values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(method='ffill').fillna(0)
    
    return features_df

def seasonal_evolution_analysis(ts_df):
    """Analyze how seasonal patterns evolve over time"""
    
    st.subheader("📈 Seasonal Pattern Evolution")
    
    # Prepare data with temporal features
    ts_df_copy = ts_df.copy()
    ts_df_copy['year'] = ts_df_copy.index.year
    ts_df_copy['month'] = ts_df_copy.index.month
    ts_df_copy['week_of_year'] = ts_df_copy.index.isocalendar().week
    
    # Year-over-year comparison
    yearly_patterns = {}
    for year in ts_df_copy['year'].unique():
        year_data = ts_df_copy[ts_df_copy['year'] == year]
        if len(year_data) >= 40:  # At least 40 weeks of data
            weekly_avg = year_data.groupby('week_of_year')['total_sessions'].mean()
            yearly_patterns[year] = weekly_avg
    
    # Plot yearly patterns
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    for i, (year, pattern) in enumerate(yearly_patterns.items()):
        fig.add_trace(go.Scatter(
            x=pattern.index, y=pattern.values,
            mode='lines+markers', name=str(year),
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title="Seasonal Patterns by Year",
        xaxis_title="Week of Year",
        yaxis_title="Average Sessions",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal stability analysis
    st.subheader("🔄 Seasonal Stability Analysis")
    
    if len(yearly_patterns) >= 3:
        # Calculate coefficient of variation for each week across years
        weeks_cv = {}
        for week in range(1, 53):
            week_values = []
            for year_pattern in yearly_patterns.values():
                if week in year_pattern.index:
                    week_values.append(year_pattern[week])
            
            if len(week_values) >= 3:
                cv = np.std(week_values) / np.mean(week_values) if np.mean(week_values) != 0 else 0
                weeks_cv[week] = cv
        
        if weeks_cv:
            cv_df = pd.DataFrame(list(weeks_cv.items()), columns=['week', 'cv'])
            
            fig = px.bar(cv_df, x='week', y='cv', 
                        title="Seasonal Stability (Lower = More Stable)")
            fig.update_layout(xaxis_title="Week of Year", yaxis_title="Coefficient of Variation")
            st.plotly_chart(fig, use_container_width=True)
            
            # Identify most/least stable periods
            most_stable = cv_df.nsmallest(5, 'cv')
            least_stable = cv_df.nlargest(5, 'cv')
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most Stable Weeks:**")
                st.dataframe(most_stable)
            
            with col2:
                st.write("**Least Stable Weeks:**")
                st.dataframe(least_stable)

def feature_testing_analysis(ts_df, original_df):
    """Test the effectiveness of created features"""
    
    st.subheader("🧪 Feature Effectiveness Testing")
    
    # Create features
    features_df = create_seasonal_features(ts_df, original_df)
    
    # Prepare data for testing
    feature_cols = [col for col in features_df.columns if col not in ['week_start', 'total_sessions', 'total_subscribers', 'total_non_subscribers']]
    
    # Remove highly correlated features and handle missing values
    X = features_df[feature_cols].fillna(0)
    y = features_df['total_sessions'].fillna(method='ffill')
    
    # Feature importance via mutual information
    st.subheader("📈 Feature Mutual Information")
    
    try:
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_df = pd.DataFrame({'feature': feature_cols, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
        
        fig = px.bar(mi_df.head(20), x='mi_score', y='feature', orientation='h',
                    title="Top 20 Features by Mutual Information Score")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display top features
        st.write("**Top 15 Most Important Features:**")
        st.dataframe(mi_df.head(15))
        
    except Exception as e:
        st.warning(f"Mutual information analysis failed: {str(e)}")
    
    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    
    try:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.feature_selection import SelectKBest, f_regression
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(20, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        st.info(f"Selected {len(selected_features)} best features out of {len(feature_cols)} total features")
        
        # Split data chronologically
        split_point = int(len(features_df) * 0.8)
        
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        X_train_selected = X_selected[:split_point]
        X_test_selected = X_selected[split_point:]
        
        # Test models
        models = {
            'All Features - Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test),
            'Selected Features - Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), X_train_selected, X_test_selected),
            'All Features - Linear': (LinearRegression(), X_train, X_test),
            'Selected Features - Linear': (LinearRegression(), X_train_selected, X_test_selected)
        }
        
        results = {}
        for name, (model, X_tr, X_te) in models.items():
            try:
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                results[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
        
        # Display results
        if results:
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df.round(2))
            
            # Plot performance comparison
            fig = px.bar(results_df.reset_index(), x='index', y='MAPE', 
                        title="Model Performance (MAPE - Lower is Better)")
            fig.update_layout(xaxis_title="Model", yaxis_title="MAPE (%)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for best performing model
            st.subheader("🎯 Feature Importance Analysis")
            
            if 'Selected Features - Random Forest' in results:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train_selected, y_train)
                
                importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(importance_df.head(15), x='importance', y='feature', 
                            orientation='h', title="Top 15 Feature Importances")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display top features
                st.write("**Top 10 Most Important Features:**")
                st.dataframe(importance_df.head(10))
        
    except Exception as e:
        st.error(f"Model testing failed: {str(e)}")
        st.info("This may be due to insufficient data or feature complexity.")

def anomaly_detection_analysis(ts_df):
    """Advanced anomaly detection and insights"""
    
    st.subheader("🚨 Anomaly Detection & Pattern Insights")
    
    # Statistical anomaly detection
    st.subheader("📊 Statistical Anomaly Detection")
    
    # Z-score based anomaly detection
    ts_df_copy = ts_df.copy()
    ts_df_copy['rolling_mean'] = ts_df_copy['total_sessions'].rolling(window=12).mean()
    ts_df_copy['rolling_std'] = ts_df_copy['total_sessions'].rolling(window=12).std()
    ts_df_copy['z_score'] = (ts_df_copy['total_sessions'] - ts_df_copy['rolling_mean']) / ts_df_copy['rolling_std']
    
    # Identify anomalies (Z-score > 2.5 or < -2.5)
    anomalies = ts_df_copy[ts_df_copy['z_score'].abs() > 2.5].copy()
    
    st.info(f"Detected {len(anomalies)} statistical anomalies using Z-score analysis")
    
    # Plot anomalies
    fig = go.Figure()
    
    # Normal data
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['total_sessions'],
                            mode='lines', name='Sessions', line=dict(color='blue')))
    
    # Rolling mean
    fig.add_trace(go.Scatter(x=ts_df_copy.index, y=ts_df_copy['rolling_mean'],
                            mode='lines', name='12-Week Average', line=dict(color='green', dash='dash')))
    
    # Anomalies
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['total_sessions'],
                                mode='markers', name='Anomalies', 
                                marker=dict(color='red', size=10, symbol='x')))
    
    fig.update_layout(title="Statistical Anomaly Detection", xaxis_title="Date", yaxis_title="Sessions")
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal anomaly detection
    st.subheader("🌊 Seasonal Anomaly Detection")
    
    # Create seasonal expectations
    ts_df_copy['week_of_year'] = ts_df_copy.index.isocalendar().week
    ts_df_copy['month'] = ts_df_copy.index.month
    ts_df_copy['year'] = ts_df_copy.index.year
    
    # Calculate seasonal baselines
    seasonal_baselines = ts_df_copy.groupby('week_of_year')['total_sessions'].agg(['mean', 'std'])
    
    # Merge back to calculate seasonal deviations
    ts_df_copy = ts_df_copy.merge(seasonal_baselines, left_on='week_of_year', right_index=True, suffixes=('', '_seasonal'))
    ts_df_copy['seasonal_deviation'] = (ts_df_copy['total_sessions'] - ts_df_copy['mean']) / ts_df_copy['std']
    
    # Identify seasonal anomalies
    seasonal_anomalies = ts_df_copy[ts_df_copy['seasonal_deviation'].abs() > 2.0].copy()
    
    st.info(f"Detected {len(seasonal_anomalies)} seasonal anomalies")
    
    # Pattern insights
    st.subheader("🔍 Pattern Insights & Recommendations")
    
    # Analyze patterns
    total_variance = ts_df['total_sessions'].var()
    
    # Trend analysis
    from sklearn.linear_model import LinearRegression
    X = np.arange(len(ts_df)).reshape(-1, 1)
    y = ts_df['total_sessions'].values
    
    lr = LinearRegression()
    lr.fit(X, y)
    trend_slope = lr.coef_[0]
    
    # Seasonality strength
    if len(ts_df) >= 52:
        seasonal_var = ts_df['total_sessions'].rolling(52).std().var()
        seasonality_strength = seasonal_var / total_variance if total_variance > 0 else 0
    else:
        seasonality_strength = 0
    
    # Volatility analysis
    volatility = ts_df['total_sessions'].pct_change().std() * 100
    
    # Growth analysis
    if len(ts_df) >= 52:
        yoy_growth = (ts_df['total_sessions'].iloc[-1] / ts_df['total_sessions'].iloc[-52] - 1) * 100
    else:
        yoy_growth = 0
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📈 Trend Slope", f"{trend_slope:.2f}", 
                 delta="per week" if trend_slope > 0 else "per week")
        st.metric("🌊 Seasonality Strength", f"{seasonality_strength:.3f}", 
                 delta="High" if seasonality_strength > 0.1 else "Low")
    
    with col2:
        st.metric("⚡ Volatility", f"{volatility:.1f}%", 
                 delta="High" if volatility > 20 else "Normal")
        st.metric("📊 YoY Growth", f"{yoy_growth:.1f}%", 
                 delta="Growing" if yoy_growth > 0 else "Declining")
    
    # Recommendations
    st.subheader("💡 Feature Engineering Recommendations")
    
    recommendations = []
    
    if seasonality_strength > 0.2:
        recommendations.append("🌊 **Strong Seasonality Detected**: Use harmonic features (sin/cos) for multiple periods")
    
    if volatility > 25:
        recommendations.append("⚡ **High Volatility**: Include volatility-based features and outlier detection")
    
    if len(anomalies) > len(ts_df) * 0.05:  # More than 5% anomalies
        recommendations.append("🚨 **Frequent Anomalies**: Consider robust scaling and anomaly indicators")
    
    if abs(trend_slope) > ts_df['total_sessions'].mean() * 0.01:
        recommendations.append("📈 **Strong Trend**: Include trend-based features and momentum indicators")
    
    if len(seasonal_anomalies) > 0:
        recommendations.append("🔍 **Seasonal Anomalies**: Add seasonal deviation features and adaptive baselines")
    
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("✅ **Data appears well-behaved** - standard features should work well")
    
    # Advanced insights
    st.subheader("🧠 Advanced Pattern Analysis")
    
    # Identify recurring patterns
    if len(ts_df) >= 104:  # At least 2 years of data
        # Compare year-over-year patterns
        years = ts_df_copy['year'].unique()
        if len(years) >= 2:
            year_correlations = []
            for i, year1 in enumerate(years[:-1]):
                for year2 in years[i+1:]:
                    y1_data = ts_df_copy[ts_df_copy['year'] == year1]['total_sessions']
                    y2_data = ts_df_copy[ts_df_copy['year'] == year2]['total_sessions']
                    
                    if len(y1_data) >= 40 and len(y2_data) >= 40:  # At least 40 weeks
                        corr = np.corrcoef(y1_data[:min(len(y1_data), len(y2_data))], 
                                          y2_data[:min(len(y1_data), len(y2_data))])[0, 1]
                        year_correlations.append((year1, year2, corr))
            
            if year_correlations:
                avg_correlation = np.mean([corr for _, _, corr in year_correlations])
                st.metric("🔄 Year-over-Year Pattern Consistency", f"{avg_correlation:.3f}",
                         delta="High consistency" if avg_correlation > 0.7 else "Variable patterns")
    
    # Forecast difficulty assessment
    forecast_difficulty = "Easy"
    difficulty_factors = []
    
    if seasonality_strength < 0.1:
        difficulty_factors.append("Low seasonality")
    if volatility > 30:
        difficulty_factors.append("High volatility")
    if len(anomalies) > len(ts_df) * 0.1:
        difficulty_factors.append("Frequent anomalies")
    if abs(trend_slope) < ts_df['total_sessions'].mean() * 0.005:
        difficulty_factors.append("Weak trend")
    
    if len(difficulty_factors) >= 3:
        forecast_difficulty = "Hard"
    elif len(difficulty_factors) >= 1:
        forecast_difficulty = "Medium"
    
    st.metric("🎯 Forecast Difficulty Assessment", forecast_difficulty,
             delta=f"Factors: {', '.join(difficulty_factors)}" if difficulty_factors else "Good predictability")

# Main function to run seasonal analysis
def run_seasonal_analysis(df):
    """Main function to run seasonal analysis"""
    seasonal_analysis_dashboard(df) 