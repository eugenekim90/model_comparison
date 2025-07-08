import polars as pl
import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering functions
from feature_engineering import create_features

def prepare_model_data(df, company, state, program, test_start="2024-10-01", test_end="2024-12-31"):
    """Prepare data for a specific combination"""
    
    # Remove duplicates first
    df, duplicates_removed = remove_duplicate_columns(df)
    
    # Get hierarchical data instead of filtering
    from data_loader import get_hierarchical_data
    subset = get_hierarchical_data(df, company, state, program)
    
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
        "week_start", "company", "state", "program", "level",
        # Target variable
        "session_count", "attended_sessions",
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
    
    # Simple feature info
    st.caption(f"Features: {len(feature_cols)} selected")
    
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
    
    # Remove duplicates first
    df, duplicates_removed = remove_duplicate_columns(df)
    
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
        aggregated_df = base_df.filter(pl.col("program") == program).group_by("week_start").agg([
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
        aggregated_df = base_df.filter(pl.col("state") == state).group_by("week_start").agg([
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
            (pl.col("state") == state) &
            (pl.col("program") == program)
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
        aggregated_df = base_df.filter(pl.col("state") == state).group_by(["week_start", "company", "state"]).agg([
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
        aggregated_df = base_df.filter(pl.col("program") == program).group_by(["week_start", "company", "program"]).agg([
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
    
    # Add aggregated features (FIXED - no data leakage)
    aggregated_df = aggregated_df.with_columns([
        # Use lagged customer data (1 week ago) to avoid leakage
        (pl.col("subscriber_count").shift(1) + pl.col("non_subscriber_count").shift(1)).alias("total_customers"),
        (pl.col("subscriber_count").shift(1) / (pl.col("subscriber_count").shift(1) + pl.col("non_subscriber_count").shift(1) + 1)).alias("subscriber_ratio"),
    ])
    
    # Add features 
    aggregated_df = create_features(aggregated_df, optimized=True)
    
    # Remove duplicates after feature creation
    aggregated_df, feature_duplicates = remove_duplicate_columns(aggregated_df)
    
    # Dynamically select feature columns
    subset_cols = aggregated_df.columns
    metadata_cols = {
        "week_start", "company", "state", "program", 
        "session_count", "subscriber_count", "non_subscriber_count", 
        "configuration", "final_industry_group", "cancellation_limit"
    }
    
    # Get all columns that are NOT metadata (these are our features)
    potential_features = [col for col in subset_cols if col not in metadata_cols]
    
    # Add back specific metadata columns that should be features
    if "cancellation_limit" in subset_cols:
        potential_features.append("cancellation_limit")
    
    # Filter to only include existing numeric columns
    feature_cols = []
    for col in potential_features:
        if col in subset_cols:
            try:
                # Simple check: if it's numeric or can be converted
                test_val = aggregated_df.select(col).limit(1).to_pandas().iloc[0, 0]
                if pd.isna(test_val) or isinstance(test_val, (int, float, bool)):
                    feature_cols.append(col)
            except:
                pass
    
    # Remove status messages - just display essential info
    # st.info(f"📊 **Aggregated Features:** {len(feature_cols)} features")
    
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
    
    # Prepare data
    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select("session_count").to_pandas().values.ravel()
    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select("session_count").to_pandas().values.ravel()
    
    # Convert to numeric and clean
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        if X_test[col].dtype == 'object':
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Fill remaining NaN values with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Replace infinities with 0
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Time series data for statistical models
    ts_data = train_df.select(["week_start", "session_count"]).to_pandas()
    ts_data = ts_data.set_index("week_start")["session_count"].fillna(0)
    
    return X_train, y_train, X_test, y_test, ts_data

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

def smart_data_preparation(df, company, state, program, test_start="2024-10-01", test_end="2024-12-31"):
    """Simplified data preparation"""
    
    # Create features if they don't exist
    from feature_engineering import create_features
    
    # Check if features already exist
    has_features = any(col.startswith(('lag_', 'rolling_', 'momentum_', 'trend_')) for col in df.columns)
    
    if not has_features:
        df = create_features(df, optimized=False)
    
    # Remove duplicate columns
    df, _ = remove_duplicate_columns(df)
    
    # Use standard preparation
    return prepare_model_data(df, company, state, program, test_start, test_end)

def remove_duplicate_columns(df):
    """Comprehensive duplicate column removal for polars dataframes"""
    try:
        # Get all column names
        original_columns = df.columns
        original_count = len(original_columns)
        
        # Find unique columns while preserving order
        seen_columns = set()
        unique_columns = []
        
        for col in original_columns:
            if col not in seen_columns:
                unique_columns.append(col)
                seen_columns.add(col)
        
        # If duplicates found, recreate dataframe with unique columns
        if len(unique_columns) != original_count:
            duplicates_removed = original_count - len(unique_columns)
            # Use only unique columns
            df = df.select(unique_columns)
            return df, duplicates_removed
        else:
            return df, 0
            
    except Exception as e:
        # Try alternative approach if main method fails
        try:
            # Convert to pandas, remove duplicates, convert back
            df_pd = df.to_pandas()
            df_pd = df_pd.loc[:, ~df_pd.columns.duplicated()]
            original_count = len(df.columns)
            final_count = len(df_pd.columns)
            duplicates_removed = original_count - final_count
            
            # Convert back to polars
            df = pl.from_pandas(df_pd)
            return df, duplicates_removed
        except:
            # If all else fails, return original df
            return df, 0 