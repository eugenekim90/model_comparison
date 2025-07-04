import polars as pl
import pandas as pd
import numpy as np
import streamlit as st

def prepare_model_data(df, company, state, program, test_start="2024-10-01", test_end="2024-12-31"):
    """Prepare data for a specific combination"""
    
    # Filter for specific combination
    subset = df.filter(
        (pl.col("company") == company) &
        (pl.col("us_state") == state) &
        (pl.col("episode_session_type") == program)
    )
    
    if len(subset) < 20:
        return None, None, None, None, None
    
    # Split train/test with proper date range
    test_start_date = pd.to_datetime(test_start).date()
    test_end_date = pd.to_datetime(test_end).date()
    train_mask = subset.get_column("week_start") < test_start_date
    test_mask = (subset.get_column("week_start") >= test_start_date) & (subset.get_column("week_start") <= test_end_date)
    train_df = subset.filter(pl.Series(train_mask))
    test_df = subset.filter(pl.Series(test_mask))
    
    if len(train_df) < 15 or len(test_df) == 0:
        return None, None, None, None, None
    
    # Dynamically select all feature columns (exclude metadata columns)
    subset_cols = subset.columns
    metadata_cols = {
        # Core identifiers
        "week_start", "company", "us_state", "episode_session_type",
        # Target variable
        "session_count",
        # Raw counts that shouldn't be features
        "subscriber_count", "non_subscriber_count", 
        # Config columns are added separately
        "configuration", "final_industry_group", "cancellation_limit"
    }
    
    # Get all columns that are NOT metadata (these are our features)
    potential_features = [col for col in subset_cols if col not in metadata_cols]
    
    # Add back specific metadata columns that should be features
    if "cancellation_limit" in subset_cols:
        potential_features.append("cancellation_limit")
    
    # Add configuration features if they exist
    config_cols = [col for col in subset_cols if str(col).startswith("config_")]
    potential_features.extend(config_cols)
    
    # Add industry encoding if it exists
    if "industry_encoded" in subset_cols:
        potential_features.append("industry_encoded")
    
    # Filter to only include columns that actually exist and are numeric/can be converted to numeric
    feature_cols = []
    for col in potential_features:
        if col in subset_cols:
            # Check if column can be converted to numeric
            try:
                test_val = subset.select(col).limit(1).to_pandas().iloc[0, 0]
                if pd.isna(test_val) or isinstance(test_val, (int, float, bool)):
                    feature_cols.append(col)
                elif isinstance(test_val, str):
                    # Try to convert string to numeric
                    try:
                        float(test_val)
                        feature_cols.append(col)
                    except:
                        pass
            except:
                pass
    
    # Display feature info
    st.info(f"📊 **Features Selected:** {len(feature_cols)} features")
    if len(feature_cols) > 50:
        st.info("🎯 **Using Optimized Features** - Advanced seasonal and pattern-based features detected")
    
    # Show sample of features for debugging
    if len(feature_cols) > 10:
        sample_features = feature_cols[:5] + ["..."] + feature_cols[-5:]
        st.caption(f"Features: {', '.join(sample_features)}")
    else:
        st.caption(f"Features: {', '.join(feature_cols)}")
    
    # Prepare training data
    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select("session_count").to_pandas().values.ravel()
    
    # Prepare test data
    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select("session_count").to_pandas().values.ravel()
    
    # Convert all columns to numeric
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        if X_test[col].dtype == 'object':
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Clean data - only remove rows with actual null values, keep 0 values
    train_mask = ~X_train.isnull().any(axis=1) & ~np.isnan(y_train)
    test_mask = ~X_test.isnull().any(axis=1) & ~np.isnan(y_test)
    
    if train_mask.sum() < 10 or test_mask.sum() == 0:
        return None, None, None, None, None
    
    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]
    X_test_clean = X_test[test_mask]
    y_test_clean = y_test[test_mask]
    
    # Time series data for statistical models
    ts_data = train_df.select(["week_start", "session_count"]).to_pandas()
    ts_data = ts_data.set_index("week_start")["session_count"].fillna(0)
    
    return X_train_clean, y_train_clean, X_test_clean, y_test_clean, ts_data

def prepare_aggregated_data(df, company, state, program, aggregation_level, test_start="2024-10-01", test_end="2024-12-31"):
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
            pl.first("cancellation_limit").alias("cancellation_limit"),
            pl.mean("week_of_year").alias("week_of_year"),
            pl.mean("month").alias("month"),
            pl.mean("quarter").alias("quarter"),
            pl.mean("is_winter").alias("is_winter"),
            pl.mean("is_spring").alias("is_spring"),
            pl.mean("is_summer").alias("is_summer"),
            pl.mean("is_fall").alias("is_fall")
        ])
        
    elif aggregation_level == "Global Program":
        # Aggregate by program across all companies
        aggregated_df = base_df.filter(pl.col("episode_session_type") == program).group_by("week_start").agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit"),
            pl.mean("week_of_year").alias("week_of_year"),
            pl.mean("month").alias("month"),
            pl.mean("quarter").alias("quarter"),
            pl.mean("is_winter").alias("is_winter"),
            pl.mean("is_spring").alias("is_spring"),
            pl.mean("is_summer").alias("is_summer"),
            pl.mean("is_fall").alias("is_fall")
        ])
    elif aggregation_level == "Global State":
        # Aggregate by state across all companies
        aggregated_df = base_df.filter(pl.col("us_state") == state).group_by("week_start").agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit"),
            pl.mean("week_of_year").alias("week_of_year"),
            pl.mean("month").alias("month"),
            pl.mean("quarter").alias("quarter"),
            pl.mean("is_winter").alias("is_winter"),
            pl.mean("is_spring").alias("is_spring"),
            pl.mean("is_summer").alias("is_summer"),
            pl.mean("is_fall").alias("is_fall")
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
            pl.first("cancellation_limit").alias("cancellation_limit"),
            pl.mean("week_of_year").alias("week_of_year"),
            pl.mean("month").alias("month"),
            pl.mean("quarter").alias("quarter"),
            pl.mean("is_winter").alias("is_winter"),
            pl.mean("is_spring").alias("is_spring"),
            pl.mean("is_summer").alias("is_summer"),
            pl.mean("is_fall").alias("is_fall")
        ])
    elif aggregation_level == "Company Total":
        # Aggregate everything to company level
        aggregated_df = base_df.group_by(["week_start", "company"]).agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit"),
            pl.mean("week_of_year").alias("week_of_year"),
            pl.mean("month").alias("month"),
            pl.mean("quarter").alias("quarter"),
            pl.mean("is_winter").alias("is_winter"),
            pl.mean("is_spring").alias("is_spring"),
            pl.mean("is_summer").alias("is_summer"),
            pl.mean("is_fall").alias("is_fall")
        ])
    elif aggregation_level == "Company-State":
        # Aggregate by state (all programs)
        aggregated_df = base_df.filter(pl.col("us_state") == state).group_by(["week_start", "company", "us_state"]).agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit"),
            pl.mean("week_of_year").alias("week_of_year"),
            pl.mean("month").alias("month"),
            pl.mean("quarter").alias("quarter"),
            pl.mean("is_winter").alias("is_winter"),
            pl.mean("is_spring").alias("is_spring"),
            pl.mean("is_summer").alias("is_summer"),
            pl.mean("is_fall").alias("is_fall")
        ])
    elif aggregation_level == "Company-Program":
        # Aggregate by program (all states)
        aggregated_df = base_df.filter(pl.col("episode_session_type") == program).group_by(["week_start", "company", "episode_session_type"]).agg([
            pl.sum("session_count").alias("session_count"),
            pl.sum("subscriber_count").alias("subscriber_count"),
            pl.sum("non_subscriber_count").alias("non_subscriber_count"),
            pl.first("configuration").alias("configuration"),
            pl.first("final_industry_group").alias("final_industry_group"),
            pl.first("cancellation_limit").alias("cancellation_limit"),
            pl.mean("week_of_year").alias("week_of_year"),
            pl.mean("month").alias("month"),
            pl.mean("quarter").alias("quarter"),
            pl.mean("is_winter").alias("is_winter"),
            pl.mean("is_spring").alias("is_spring"),
            pl.mean("is_summer").alias("is_summer"),
            pl.mean("is_fall").alias("is_fall")
        ])
    else:
        # Granular level
        return prepare_model_data(df, company, state, program, test_start, test_end)
    
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
    
    # For aggregated data, use the full feature engineering pipeline
    # This allows optimized features to be used for monthly forecasting too
    from feature_engineering import create_features
    
    # Check if we should use optimized features (passed via session state or detect from UI)
    import streamlit as st
    use_optimized = st.session_state.get('use_optimized', False)
    
    if use_optimized:
        # Use full feature engineering pipeline for optimized features
        aggregated_df = create_features(aggregated_df, optimized=True)
    else:
        # Use basic features only
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
    
    # Split train/test with proper date range
    test_start_date = pd.to_datetime(test_start).date()
    test_end_date = pd.to_datetime(test_end).date()
    train_mask = aggregated_df.get_column("week_start") < test_start_date
    test_mask = (aggregated_df.get_column("week_start") >= test_start_date) & (aggregated_df.get_column("week_start") <= test_end_date)
    train_df = aggregated_df.filter(pl.Series(train_mask))
    test_df = aggregated_df.filter(pl.Series(test_mask))
    
    if len(train_df) < 8 or len(test_df) == 0:  # Reduced from 15 to 8 for aggregated data
        st.error(f"Train data: {len(train_df)} rows, Test data: {len(test_df)} rows. Need at least 8 train and 1 test.")
        return None, None, None, None, None
    
    # Dynamically select feature columns for aggregated data
    available_cols = set(train_df.columns)
    metadata_cols = {
        "week_start", "session_count", "subscriber_count", "non_subscriber_count",
        "configuration", "final_industry_group", "company", "us_state", "episode_session_type"
    }
    
    # Get all columns that are NOT metadata (these are our features)
    potential_features = [col for col in available_cols if col not in metadata_cols]
    
    # Filter to only include numeric columns
    feature_cols = []
    for col in potential_features:
        if col in available_cols:
            try:
                test_val = train_df.select(col).limit(1).to_pandas().iloc[0, 0]
                if pd.isna(test_val) or isinstance(test_val, (int, float, bool)):
                    feature_cols.append(col)
            except:
                pass
    
    # Display info about aggregated features
    st.info(f"📊 **Aggregated Features:** {len(feature_cols)} features")
    
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
        pl.col("cancellation_limit").fill_null(0),
        pl.col("week_of_year").fill_null(pl.col("week_start").dt.week()),
        pl.col("month").fill_null(pl.col("week_start").dt.month()),
        pl.col("quarter").fill_null(pl.col("week_start").dt.quarter()),
        pl.col("is_winter").fill_null(pl.when(pl.col("week_start").dt.month().is_in([12, 1, 2])).then(1).otherwise(0)),
        pl.col("is_spring").fill_null(pl.when(pl.col("week_start").dt.month().is_in([3, 4, 5])).then(1).otherwise(0)),
        pl.col("is_summer").fill_null(pl.when(pl.col("week_start").dt.month().is_in([6, 7, 8])).then(1).otherwise(0)),
        pl.col("is_fall").fill_null(pl.when(pl.col("week_start").dt.month().is_in([9, 10, 11])).then(1).otherwise(0))
    ])
    return aggregated_df

def get_aggregation_level(company, state, program):
    """Determine aggregation level based on selections"""
    if company == "ALL COMPANIES" and state == "ALL STATES" and program == "ALL PROGRAMS":
        return "Global Total"
    elif company == "ALL COMPANIES" and state == "ALL STATES":
        return "Global Program"
    elif company == "ALL COMPANIES" and program == "ALL PROGRAMS":
        return "Global State"
    elif company == "ALL COMPANIES":
        return "Global Granular"
    elif state == "ALL STATES" and program == "ALL PROGRAMS":
        return "Company Total"
    elif state == "ALL STATES":
        return "Company-Program"
    elif program == "ALL PROGRAMS":
        return "Company-State"
    else:
        return "Granular"

def create_display_name(company, state, program):
    """Create display name based on selections"""
    if company == "ALL COMPANIES" and state == "ALL STATES" and program == "ALL PROGRAMS":
        return "ALL COMPANIES - ALL STATES - ALL PROGRAMS"
    elif company == "ALL COMPANIES" and state == "ALL STATES":
        return f"ALL COMPANIES - ALL STATES - {program}"
    elif company == "ALL COMPANIES" and program == "ALL PROGRAMS":
        return f"ALL COMPANIES - {state} - ALL PROGRAMS"
    elif company == "ALL COMPANIES":
        return f"ALL COMPANIES - {state} - {program}"
    elif state == "ALL STATES" and program == "ALL PROGRAMS":
        return f"{company} - ALL STATES - ALL PROGRAMS"
    elif state == "ALL STATES":
        return f"{company} - ALL STATES - {program}"
    elif program == "ALL PROGRAMS":
        return f"{company} - {state} - ALL PROGRAMS"
    else:
        return f"{company} - {state} - {program}" 