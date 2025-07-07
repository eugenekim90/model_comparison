import polars as pl
import pandas as pd
import numpy as np
import streamlit as st

def create_features(df, optimized=True):
    """Create features for time series forecasting"""
    
    # Always use base features, ignore optimized parameter
    # Sort data to ensure proper lag calculations
    available_cols = df.columns
    
    # Determine sort columns based on what's available
    if all(col in available_cols for col in ["company", "us_state", "episode_session_type"]):
        sort_cols = ["company", "us_state", "episode_session_type", "week_start"]
    else:
        sort_cols = ["week_start"]
    
    df = df.sort(sort_cols)
    
    # Create basic features
    df = _add_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    
    # Only add categorical encoding if grouping columns exist
    if all(col in available_cols for col in ["company", "us_state", "episode_session_type"]):
        df = _add_categorical_encoding(df)
    
    return df

def _add_time_features(df):
    """Add basic time-based features"""
    
    df = df.with_columns([
        # Basic time features
        pl.col("week_start").dt.week().alias("week_of_year"),
        pl.col("week_start").dt.month().alias("month"),
        pl.col("week_start").dt.quarter().alias("quarter"),
        pl.col("week_start").dt.year().alias("year"),
        pl.col("week_start").dt.weekday().alias("weekday"),
        
        # Seasonal indicators (quarters)
        pl.when(pl.col("week_start").dt.month().is_in([12, 1, 2])).then(1).otherwise(0).alias("is_winter"),
        pl.when(pl.col("week_start").dt.month().is_in([3, 4, 5])).then(1).otherwise(0).alias("is_spring"),
        pl.when(pl.col("week_start").dt.month().is_in([6, 7, 8])).then(1).otherwise(0).alias("is_summer"),
        pl.when(pl.col("week_start").dt.month().is_in([9, 10, 11])).then(1).otherwise(0).alias("is_fall"),
    ])
    
    return df

def _add_lag_features(df):
    """Add lag features"""
    
    # Check what columns are available for grouping
    available_cols = df.columns
    group_cols = []
    
    # Add grouping columns if they exist (for raw data)
    if "company" in available_cols:
        group_cols.append("company")
    if "us_state" in available_cols:
        group_cols.append("us_state")
    if "episode_session_type" in available_cols:
        group_cols.append("episode_session_type")
    
    # Basic lag features (weeks)
    for lag in [1, 2, 4, 8, 12, 26, 52]:
        if group_cols:
            df = df.with_columns([
                pl.col("session_count")
                .shift(lag)
                .over(group_cols)
                .alias(f"lag_{lag}"),
            ])
        else:
            df = df.with_columns([
                pl.col("session_count")
                .shift(lag)
                .alias(f"lag_{lag}"),
            ])
    
    return df

def _add_rolling_features(df):
    """Add rolling statistics features"""
    
    # Check what columns are available for grouping
    available_cols = df.columns
    group_cols = []
    
    # Add grouping columns if they exist (for raw data)
    if "company" in available_cols:
        group_cols.append("company")
    if "us_state" in available_cols:
        group_cols.append("us_state")
    if "episode_session_type" in available_cols:
        group_cols.append("episode_session_type")
    
    # Rolling statistics (short-term)
    for window in [4, 8, 12]:
        if group_cols:
            df = df.with_columns([
                pl.col("session_count").shift(1).rolling_mean(window)
                .over(group_cols)
                .alias(f"rolling_mean_{window}"),
                
                pl.col("session_count").shift(1).rolling_std(window)
                .over(group_cols)
                .alias(f"rolling_std_{window}"),
                
                pl.col("session_count").shift(1).rolling_max(window)
                .over(group_cols)
                .alias(f"rolling_max_{window}"),
                
                pl.col("session_count").shift(1).rolling_min(window)
                .over(group_cols)
                .alias(f"rolling_min_{window}"),
            ])
        else:
            df = df.with_columns([
                pl.col("session_count").shift(1).rolling_mean(window)
                .alias(f"rolling_mean_{window}"),
                
                pl.col("session_count").shift(1).rolling_std(window)
                .alias(f"rolling_std_{window}"),
                
                pl.col("session_count").shift(1).rolling_max(window)
                .alias(f"rolling_max_{window}"),
                
                pl.col("session_count").shift(1).rolling_min(window)
                .alias(f"rolling_min_{window}"),
            ])
    
    return df

def _add_enhanced_time_features(df):
    """Add comprehensive time-based features including strong seasonal patterns"""
    
    df = df.with_columns([
        # Basic time features
        pl.col("week_start").dt.week().alias("week_of_year"),
        pl.col("week_start").dt.month().alias("month"),
        pl.col("week_start").dt.quarter().alias("quarter"),
        pl.col("week_start").dt.year().alias("year"),
        pl.col("week_start").dt.weekday().alias("weekday"),
        
        # Seasonal indicators (quarters)
        pl.when(pl.col("week_start").dt.month().is_in([12, 1, 2])).then(1).otherwise(0).alias("is_winter"),
        pl.when(pl.col("week_start").dt.month().is_in([3, 4, 5])).then(1).otherwise(0).alias("is_spring"),
        pl.when(pl.col("week_start").dt.month().is_in([6, 7, 8])).then(1).otherwise(0).alias("is_summer"),
        pl.when(pl.col("week_start").dt.month().is_in([9, 10, 11])).then(1).otherwise(0).alias("is_fall"),
        
        # Business seasonal indicators
        pl.when(pl.col("week_start").dt.month().is_in([1, 2])).then(1).otherwise(0).alias("is_q1_start"),  # New year surge
        pl.when(pl.col("week_start").dt.month().is_in([9, 10])).then(1).otherwise(0).alias("is_back_to_school"),  # Fall surge
        pl.when(pl.col("week_start").dt.month().is_in([11, 12])).then(1).otherwise(0).alias("is_holiday_season"),  # Holiday season
        pl.when(pl.col("week_start").dt.month().is_in([6, 7, 8])).then(1).otherwise(0).alias("is_summer_break"),  # Summer patterns
        
        # Year-end patterns
        pl.when(pl.col("week_start").dt.week().is_in([51, 52, 1, 2])).then(1).otherwise(0).alias("is_year_end"),
        pl.when(pl.col("week_start").dt.week().is_in([13, 14, 15, 16])).then(1).otherwise(0).alias("is_spring_break"),
        
        # Cyclical features (sine/cosine for smooth periodicity)
        (2 * np.pi * pl.col("week_start").dt.week() / 52).sin().alias("week_sin"),
        (2 * np.pi * pl.col("week_start").dt.week() / 52).cos().alias("week_cos"),
        (2 * np.pi * pl.col("week_start").dt.month() / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("week_start").dt.month() / 12).cos().alias("month_cos"),
        
        # Year progression (0 to 1 through the year)
        (pl.col("week_start").dt.week() / 52.0).alias("year_progress"),
        
        # Multi-year cycles (2-year, 3-year patterns)
        ((pl.col("week_start").dt.year() % 2)).alias("year_mod_2"),
        ((pl.col("week_start").dt.year() % 3)).alias("year_mod_3"),
        
        # Relative year (years since start of data)
        (pl.col("week_start").dt.year() - pl.col("week_start").dt.year().min()).alias("years_since_start"),
    ])
    
    return df

def _add_holiday_features(df):
    """Add holiday and special event features that affect sessions"""
    
    df = df.with_columns([
        # Major holidays (approximate weeks)
        pl.when(
            (pl.col("week_start").dt.month() == 1) & 
            (pl.col("week_start").dt.week().is_in([1, 2]))
        ).then(1).otherwise(0).alias("is_new_year"),
        
        pl.when(
            (pl.col("week_start").dt.month() == 12) & 
            (pl.col("week_start").dt.week().is_in([51, 52]))
        ).then(1).otherwise(0).alias("is_christmas"),
        
        pl.when(
            (pl.col("week_start").dt.month() == 11) & 
            (pl.col("week_start").dt.week().is_in([47, 48]))
        ).then(1).otherwise(0).alias("is_thanksgiving"),
        
        pl.when(
            (pl.col("week_start").dt.month() == 7) & 
            (pl.col("week_start").dt.week() == 27)
        ).then(1).otherwise(0).alias("is_july_4th"),
        
        # Tax season
        pl.when(
            (pl.col("week_start").dt.month().is_in([3, 4])) & 
            (pl.col("week_start").dt.week().is_in([12, 13, 14, 15, 16]))
        ).then(1).otherwise(0).alias("is_tax_season"),
        
        # Back-to-school period (important for many businesses)
        pl.when(
            (pl.col("week_start").dt.month().is_in([8, 9])) & 
            (pl.col("week_start").dt.week().is_in([34, 35, 36, 37]))
        ).then(1).otherwise(0).alias("is_back_to_school_period"),
        
        # Black Friday / Cyber Monday season
        pl.when(
            (pl.col("week_start").dt.month() == 11) & 
            (pl.col("week_start").dt.week().is_in([47, 48]))
        ).then(1).otherwise(0).alias("is_black_friday_week"),
    ])
    
    return df

def _add_trend_features(df):
    """Add trend and momentum features"""
    
    # Check what columns are available for grouping
    available_cols = df.columns
    group_cols = []
    
    # Add grouping columns if they exist (for raw data)
    if "company" in available_cols:
        group_cols.append("company")
    if "us_state" in available_cols:
        group_cols.append("us_state")
    if "episode_session_type" in available_cols:
        group_cols.append("episode_session_type")
    
    if group_cols:
        df = df.with_columns([
            # Short-term trends (4-week slopes)
            (pl.col("session_count").shift(1) - pl.col("session_count").shift(4))
            .over(group_cols)
            .alias("trend_4w"),
            
            # Medium-term trends (12-week slopes)
            (pl.col("session_count").shift(1) - pl.col("session_count").shift(12))
            .over(group_cols)
            .alias("trend_12w"),
            
            # Long-term trends (26-week slopes)
            (pl.col("session_count").shift(1) - pl.col("session_count").shift(26))
            .over(group_cols)
            .alias("trend_26w"),
            
            # Acceleration (change in trend)
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(4)) -
             (pl.col("session_count").shift(4) - pl.col("session_count").shift(8)))
            .over(group_cols)
            .alias("acceleration_4w"),
            
            # Volatility (rolling coefficient of variation)
            (pl.col("session_count").shift(1).rolling_std(8) / 
             (pl.col("session_count").shift(1).rolling_mean(8) + 1))
            .over(group_cols)
            .alias("volatility_8w"),
            
            # Momentum indicators (FIXED - properly normalized)
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(1).rolling_mean(4).over(group_cols)) / 
             (pl.col("session_count").shift(1).rolling_mean(4).over(group_cols) + 1))
            .alias("momentum_4w"),
            
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(1).rolling_mean(12).over(group_cols)) / 
             (pl.col("session_count").shift(1).rolling_mean(12).over(group_cols) + 1))
            .alias("momentum_12w"),
        ])
    else:
        df = df.with_columns([
            # Short-term trends (4-week slopes)
            (pl.col("session_count").shift(1) - pl.col("session_count").shift(4))
            .alias("trend_4w"),
            
            # Medium-term trends (12-week slopes)
            (pl.col("session_count").shift(1) - pl.col("session_count").shift(12))
            .alias("trend_12w"),
            
            # Long-term trends (26-week slopes)
            (pl.col("session_count").shift(1) - pl.col("session_count").shift(26))
            .alias("trend_26w"),
            
            # Acceleration (change in trend)
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(4)) -
             (pl.col("session_count").shift(4) - pl.col("session_count").shift(8)))
            .alias("acceleration_4w"),
            
            # Volatility (rolling coefficient of variation)
            (pl.col("session_count").shift(1).rolling_std(8) / 
             (pl.col("session_count").shift(1).rolling_mean(8) + 1))
            .alias("volatility_8w"),
            
            # Momentum indicators (FIXED - properly normalized)
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(1).rolling_mean(4)) / 
             (pl.col("session_count").shift(1).rolling_mean(4) + 1))
            .alias("momentum_4w"),
            
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(1).rolling_mean(12)) / 
             (pl.col("session_count").shift(1).rolling_mean(12) + 1))
            .alias("momentum_12w"),
        ])
    
    return df

def _add_categorical_encoding(df):
    """Add categorical feature encodings"""
    
    # Company encoding (only if column exists)
    if "company" in df.columns:
        companies = sorted(df.select("company").unique().to_pandas()["company"].tolist())
        company_mapping = {company: i for i, company in enumerate(companies)}
        df = df.with_columns([
            pl.col("company").replace(company_mapping).cast(pl.Int32).alias("company_encoded")
        ])
    
    # State encoding (only if column exists)
    if "us_state" in df.columns:
        states = sorted(df.select("us_state").unique().to_pandas()["us_state"].tolist())
        state_mapping = {state: i for i, state in enumerate(states)}
        df = df.with_columns([
            pl.col("us_state").replace(state_mapping).cast(pl.Int32).alias("state_encoded")
        ])
    
    # Program encoding (only if column exists)
    if "episode_session_type" in df.columns:
        programs = sorted(df.select("episode_session_type").unique().to_pandas()["episode_session_type"].tolist())
        program_mapping = {program: i for i, program in enumerate(programs)}
        df = df.with_columns([
            pl.col("episode_session_type").replace(program_mapping).cast(pl.Int32).alias("program_encoded")
        ])
    
    # Industry encoding
    if "final_industry_group" in df.columns:
        # Filter out null values before sorting
        industries = [x for x in df.select("final_industry_group").unique().to_pandas()["final_industry_group"].tolist() if x is not None]
        industries = sorted(industries)
        # Include null mapping
        industry_mapping = {industry: i for i, industry in enumerate(industries)}
        industry_mapping[None] = -1  # Map null to -1
        df = df.with_columns([
            pl.col("final_industry_group").replace(industry_mapping).cast(pl.Int32).alias("industry_encoded")
        ])
    
    # Configuration features (one-hot encoding)
    if "configuration" in df.columns:
        unique_configs = df.select("configuration").unique().to_pandas()["configuration"].tolist()
        for config in unique_configs:
            if config is not None:
                df = df.with_columns([
                    pl.when(pl.col("configuration") == config).then(1).otherwise(0).alias(f"config_{config}")
                ])
    
    return df

def _clean_features(df):
    """Clean and fill null values in features"""
    
    # Get all lag and rolling columns
    lag_cols = [col for col in df.columns if str(col).startswith("lag_")]
    rolling_cols = [col for col in df.columns if str(col).startswith("rolling_")]
    trend_cols = [col for col in df.columns if str(col).startswith("trend_") or 
                  str(col).startswith("acceleration_") or str(col).startswith("volatility_") or 
                  str(col).startswith("momentum_") or str(col).startswith("yoy_")]
    growth_cols = [col for col in df.columns if str(col).endswith("_growth")]
    
    # Fill nulls with appropriate values
    df = df.with_columns([
        pl.col(col).fill_null(0) for col in lag_cols + rolling_cols + trend_cols + growth_cols
    ])
    
    # Fill subscriber_ratio nulls
    if "subscriber_ratio" in df.columns:
        df = df.with_columns([
            pl.col("subscriber_ratio").fill_null(0.5)  # Neutral ratio
        ])
    
    return df

def fill_missing_dates(aggregated_df, date_df):
    """Fill missing dates in aggregated data with 0s"""
    # Fill missing weeks with 0s
    aggregated_df = date_df.join(aggregated_df, on="week_start", how="left")
    aggregated_df = aggregated_df.with_columns([
        pl.col("session_count").fill_null(0),
        pl.col("subscriber_count").fill_null(0),
        pl.col("non_subscriber_count").fill_null(0),
        pl.col("configuration").fill_null("default"),
        pl.col("final_industry_group").fill_null("default"),
        pl.col("cancellation_limit").fill_null(0)
    ])
    return aggregated_df

def prepare_aggregated_data(df, company, state, program, aggregation_level, test_start="2024-10-01"):
    """Prepare aggregated data based on aggregation level"""
    
    # Filter base data
    if company == "ALL COMPANIES":
        base_df = df
    else:
        base_df = df.filter(pl.col("company") == company)
    
    # Create complete date range from data to fill missing weeks with 0s
    min_date = base_df.select(pl.min("week_start")).item()
    max_date = base_df.select(pl.max("week_start")).item()
    
    # Create complete weekly date range
    complete_dates = pl.date_range(min_date, max_date, interval="1w", eager=True)
    date_df = pl.DataFrame({"week_start": complete_dates})
    
    if aggregation_level == "Global Total":
        # Aggregate everything across all companies
        aggregated_df = base_df.group_by("week_start").agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit")
        ])
        
    elif aggregation_level == "Global Program":
        # Aggregate by program across all companies
        aggregated_df = base_df.filter(pl.col("episode_session_type") == program).group_by("week_start").agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit")
        ])
    elif aggregation_level == "Global State":
        # Aggregate by state across all companies
        aggregated_df = base_df.filter(pl.col("us_state") == state).group_by("week_start").agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit")
        ])
    elif aggregation_level == "Global Granular":
        # Specific state-program across all companies
        aggregated_df = base_df.filter(
            (pl.col("us_state") == state) &
            (pl.col("episode_session_type") == program)
        ).group_by("week_start").agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit")
        ])
    elif aggregation_level == "Company Total":
        # Aggregate everything to company level
        aggregated_df = base_df.group_by(["week_start", "company"]).agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit")
        ])
    elif aggregation_level == "Company-State":
        # Aggregate by state (all programs)
        aggregated_df = base_df.filter(pl.col("us_state") == state).group_by(["week_start", "company", "us_state"]).agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit")
        ])
    elif aggregation_level == "Company-Program":
        # Aggregate by program (all states)
        aggregated_df = base_df.filter(pl.col("episode_session_type") == program).group_by(["week_start", "company", "episode_session_type"]).agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit")
        ])
    else:
        # Granular level
        return prepare_model_data(df, company, state, program, test_start)
    
    # Apply missing date filling to all aggregation levels except granular
    if aggregation_level != "Granular":
        aggregated_df = fill_missing_dates(aggregated_df, date_df)
    
    if len(aggregated_df) < 10:  # Reduced from 20 to 10 for aggregated data
        st.error(f"Aggregated data only has {len(aggregated_df)} rows, need at least 10")
        return None, None, None, None, None
    
    # Add aggregated features
    aggregated_df = aggregated_df.with_columns([
        (pl.col("subscriber_count") + pl.col("non_subscriber_count")).alias("total_customers"),
        (pl.col("subscriber_count") / (pl.col("subscriber_count") + pl.col("non_subscriber_count") + 1)).alias("subscriber_ratio"),
    ])
    
    # Add lag features
    aggregated_df = aggregated_df.sort("week_start")
    for lag in [1, 2, 4, 8, 12]:
        aggregated_df = aggregated_df.with_columns([
            pl.col("session_count").shift(lag).alias(f"lag_{lag}"),
        ])
    
    # Add rolling statistics
    for window in [4, 8, 12]:
        aggregated_df = aggregated_df.with_columns([
            pl.col("session_count").shift(1).rolling_mean(window).alias(f"rolling_mean_{window}"),
            pl.col("session_count").shift(1).rolling_std(window).alias(f"rolling_std_{window}"),
        ])
    
    # Fill null lag features with 0 (for cases where there's insufficient history)
    lag_cols = [f"lag_{lag}" for lag in [1, 2, 4, 8, 12]]
    rolling_cols = [f"rolling_mean_{window}" for window in [4, 8, 12]] + [f"rolling_std_{window}" for window in [4, 8, 12]]
    
    aggregated_df = aggregated_df.with_columns([
        pl.col(col).fill_null(0) for col in lag_cols + rolling_cols
    ])
    
    # Split train/test
    test_mask = aggregated_df.get_column("week_start") >= pd.to_datetime(test_start).date()
    train_df = aggregated_df.filter(~pl.Series(test_mask))
    test_df = aggregated_df.filter(pl.Series(test_mask))
    
    if len(train_df) < 8 or len(test_df) == 0:  # Reduced from 15 to 8 for aggregated data
        st.error(f"Train data: {len(train_df)} rows, Test data: {len(test_df)} rows. Need at least 8 train and 1 test.")
        return None, None, None, None, None
    
    # Feature columns (for aggregated data, we use fewer categorical features)
    feature_cols = [
        "lag_1", "lag_2", "lag_4", "lag_8", "lag_12",
        "rolling_mean_4", "rolling_mean_8", "rolling_mean_12",
        "rolling_std_4", "rolling_std_8", "rolling_std_12",
        "total_customers", "subscriber_ratio", "cancellation_limit"
    ]
    
    # For aggregated data, don't include state/program encoding since we're aggregating across them
    
    # Prepare training data
    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select("session_count").to_pandas().values.ravel()
    
    # Prepare test data
    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select("session_count").to_pandas().values.ravel()
    
    # Convert all columns to numeric (fix categorical encoding dtype issues)
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        if X_test[col].dtype == 'object':
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Clean data
    train_mask = ~X_train.isnull().any(axis=1) & ~np.isnan(y_train)
    test_mask = ~X_test.isnull().any(axis=1) & ~np.isnan(y_test)
    
    if train_mask.sum() < 5 or test_mask.sum() == 0:  # Reduced from 10 to 5 for aggregated data
        st.error(f"Clean train data: {train_mask.sum()} rows, Clean test data: {test_mask.sum()} rows. Need at least 5 train and 1 test.")
        return None, None, None, None, None
    
    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]  # Keep 0 values in target
    X_test_clean = X_test[test_mask]
    y_test_clean = y_test[test_mask]  # Keep 0 values in target
    
    # Time series data for statistical models (fill 0 for missing values)
    ts_data = train_df.select(["week_start", "session_count"]).to_pandas()
    ts_data = ts_data.set_index("week_start")["session_count"].fillna(0)
    
    return X_train_clean, y_train_clean, X_test_clean, y_test_clean, ts_data

def create_optimized_features_enhanced(df):
    """Create optimized features based on seasonal analysis insights (no leakage)"""
    
    # Check what columns are available for grouping
    available_cols = df.columns
    group_cols = []
    
    # Add grouping columns if they exist (for raw data)
    if "company" in available_cols:
        group_cols.append("company")
    if "us_state" in available_cols:
        group_cols.append("us_state")
    if "episode_session_type" in available_cols:
        group_cols.append("episode_session_type")
    
    # FIRST: Add all essential time features that optimized features depend on
    df = _add_enhanced_time_features(df)
    df = _add_holiday_features(df)
    df = _add_trend_features(df)
    
    # Only add categorical encoding if we have grouping columns
    if group_cols:
        df = _add_categorical_encoding(df)
    
    # Add basic lag features first (these are needed for optimized features)
    for lag in [1, 2, 4, 8, 12]:
        if group_cols:
            df = df.with_columns([
                pl.col("session_count")
                .shift(lag)
                .over(group_cols)
                .alias(f"lag_{lag}"),
            ])
        else:
            df = df.with_columns([
                pl.col("session_count")
                .shift(lag)
                .alias(f"lag_{lag}"),
            ])
    
    # Add year-over-year growth features (FIXED - no data leakage)
    if group_cols:
        df = df.with_columns([
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(52).over(group_cols)) /
             (pl.col("session_count").shift(52).over(group_cols) + pl.lit(1)))
            .alias("yoy_growth_52w"),
        ])
    else:
        df = df.with_columns([
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(52)) /
             (pl.col("session_count").shift(52) + pl.lit(1)))
            .alias("yoy_growth_52w"),
        ])
    
    # Holiday dip prediction features (historical probabilities - no leakage)
    holiday_dip_weeks = {
        47: 0.8, 48: 0.9, 51: 0.95, 52: 0.9, 1: 0.7  # From seasonal analysis
    }
    
    # Add enhanced lag features for long-term patterns
    critical_lags = [26, 52, 78, 104]  # 6mo, 1yr, 1.5yr, 2yr
    for lag in critical_lags:
        if group_cols:
            df = df.with_columns([
                pl.col("session_count")
                .shift(lag)
                .over(group_cols)
                .alias(f"sessions_lag_{lag}w"),
                
                ((pl.col("session_count").shift(1) - pl.col("session_count").shift(lag).over(group_cols)) /
                 (pl.col("session_count").shift(lag).over(group_cols) + pl.lit(1)))
                .alias(f"sessions_change_{lag}w"),
            ])
        else:
            df = df.with_columns([
                pl.col("session_count")
                .shift(lag)
                .alias(f"sessions_lag_{lag}w"),
                
                ((pl.col("session_count").shift(1) - pl.col("session_count").shift(lag)) /
                 (pl.col("session_count").shift(lag) + pl.lit(1)))
                .alias(f"sessions_change_{lag}w"),
            ])
    
    # Holiday and stability features (no future information)
    df = df.with_columns([
        # Holiday dip probability based on week of year only
        pl.col("week_of_year").map_elements(
            lambda x: holiday_dip_weeks.get(x, 0.1), 
            return_dtype=pl.Float64
        ).alias("holiday_dip_probability"),
        
        # Q4 intensity based on month only
        pl.when(pl.col("month").is_in([11, 12])).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("q4_intensity"),
        
        # Post-holiday recovery based on week of year only
        pl.when(pl.col("week_of_year").is_in([2, 3, 4])).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("post_holiday_recovery"),
        
        # Seasonal stability score based on week of year (from analysis)
        pl.when(pl.col("week_of_year").is_between(40, 50)).then(pl.lit(1.0))  # High stability
        .when(pl.col("week_of_year").is_between(20, 39)).then(pl.lit(0.8))    # Medium-high
        .when(pl.col("week_of_year").is_between(6, 19)).then(pl.lit(0.6))     # Medium
        .when(pl.col("week_of_year").is_between(1, 5)).then(pl.lit(0.2))      # Low (start of year)
        .when(pl.col("week_of_year").is_between(46, 52)).then(pl.lit(0.1))    # Very low (holidays)
        .otherwise(pl.lit(0.5)).alias("seasonal_stability_score"),
        
        # Holiday season gradient based on week of year only
        pl.when(pl.col("week_of_year").is_in([45, 46])).then(pl.lit(0.5))     # Pre-holiday buildup
        .when(pl.col("week_of_year").is_in([47, 48, 49, 50, 51, 52])).then(pl.lit(1.0))  # Full holiday season
        .otherwise(pl.lit(0.0)).alias("holiday_season_gradient"),
    ])
    
    # Enhanced rolling features for top windows (with proper lags)
    for window in [4, 8, 12]:
        if group_cols:
            df = df.with_columns([
                pl.col("session_count").shift(1).rolling_mean(window)
                .over(group_cols)
                .alias(f"rolling_mean_{window}w"),
                
                pl.col("session_count").shift(1).rolling_std(window)
                .over(group_cols)
                .alias(f"rolling_std_{window}w"),
                
                (pl.col("session_count").shift(1).rolling_std(window) / 
                 (pl.col("session_count").shift(1).rolling_mean(window) + pl.lit(1)))
                .over(group_cols)
                .alias(f"volatility_{window}w"),
                
                (pl.lit(1) / (pl.lit(1) + pl.col("session_count").shift(1).rolling_std(window)
                      .over(group_cols)))
                .alias(f"stability_score_{window}w"),
            ])
        else:
            df = df.with_columns([
                pl.col("session_count").shift(1).rolling_mean(window)
                .alias(f"rolling_mean_{window}w"),
                
                pl.col("session_count").shift(1).rolling_std(window)
                .alias(f"rolling_std_{window}w"),
                
                (pl.col("session_count").shift(1).rolling_std(window) / 
                 (pl.col("session_count").shift(1).rolling_mean(window) + pl.lit(1)))
                .alias(f"volatility_{window}w"),
                
                (pl.lit(1) / (pl.lit(1) + pl.col("session_count").shift(1).rolling_std(window)))
                .alias(f"stability_score_{window}w"),
            ])
    
    # Enhanced trend and momentum features (all with proper lags)
    if group_cols:
        df = df.with_columns([
            # Recovery momentum (top feature from analysis)
            (((pl.col("session_count").shift(1) - pl.col("session_count").shift(2)) +
              (pl.col("session_count").shift(2) - pl.col("session_count").shift(3))) / pl.lit(2))
            .over(group_cols)
            .alias("recovery_momentum"),
            
            # 12-week momentum (FIXED - properly normalized)
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(1).rolling_mean(12).over(group_cols)) / 
             (pl.col("session_count").shift(1).rolling_mean(12).over(group_cols) + pl.lit(1)))
            .alias("momentum_12w"),
            
            # Trend slope (12-week) - using lagged data
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(12)) / pl.lit(12))
            .over(group_cols)
            .alias("trend_slope_12w"),
            
            # Deviation from trend (important feature) - using lagged trend
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(1).rolling_mean(4)
              .over(group_cols)) /
             (pl.col("session_count").shift(1).rolling_mean(4)
              .over(group_cols) + pl.lit(1)))
            .alias("deviation_from_trend"),
        ])
    else:
        df = df.with_columns([
            # Recovery momentum (top feature from analysis)
            (((pl.col("session_count").shift(1) - pl.col("session_count").shift(2)) +
              (pl.col("session_count").shift(2) - pl.col("session_count").shift(3))) / pl.lit(2))
            .alias("recovery_momentum"),
            
            # 12-week momentum (FIXED - properly normalized)
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(1).rolling_mean(12)) / 
             (pl.col("session_count").shift(1).rolling_mean(12) + pl.lit(1)))
            .alias("momentum_12w"),
            
            # Trend slope (12-week) - using lagged data
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(12)) / pl.lit(12))
            .alias("trend_slope_12w"),
            
            # Deviation from trend (important feature) - using lagged trend
            ((pl.col("session_count").shift(1) - pl.col("session_count").shift(1).rolling_mean(4)) /
             (pl.col("session_count").shift(1).rolling_mean(4) + pl.lit(1)))
            .alias("deviation_from_trend"),
        ])
    
    # Seasonal strength (52-week rolling with lag)
    if group_cols:
        df = df.with_columns([
            (pl.col("session_count").shift(1).rolling_std(52) / 
             (pl.col("session_count").shift(1).rolling_mean(52) + pl.lit(1)))
            .over(group_cols)
            .alias("seasonal_strength"),
        ])
    else:
        df = df.with_columns([
            (pl.col("session_count").shift(1).rolling_std(52) / 
             (pl.col("session_count").shift(1).rolling_mean(52) + pl.lit(1)))
            .alias("seasonal_strength"),
        ])
    
    # Clean features
    df = _clean_features(df)
    
    return df 