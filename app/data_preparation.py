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
    
    # Remove verbose status messages - just display essential info
    # Display feature info
    # st.info(f"📊 **Features Selected:** {len(feature_cols)} features")
    # if len(feature_cols) > 50:
    #     st.info("🎯 **Using Optimized Features** - Advanced seasonal and pattern-based features detected")
    
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

def assess_data_quality(df, company, state, program, test_start="2024-10-01", test_end="2024-12-31"):
    """Assess the quality of data for forecasting"""
    
    # Apply comprehensive duplicate column removal first
    df, duplicates_removed = remove_duplicate_columns(df)
    # Remove status message - if duplicates_removed > 0:
    #     st.warning(f"⚠️ Removed {duplicates_removed} duplicate columns from dataset")
    
    # Apply filters using hierarchical data
    from data_loader import get_hierarchical_data
    subset = get_hierarchical_data(df, company, state, program)
    
    # Calculate data metrics - ensure week_start column exists and is unique
    total_weeks = len(subset)
    if "week_start" not in subset.columns:
        # If week_start doesn't exist, try to find a date column
        date_cols = [col for col in subset.columns if "date" in col.lower() or "week" in col.lower()]
        if date_cols:
            week_start_col = date_cols[0]
            # Get min/max from that column
            try:
                date_range = subset.select([pl.min(week_start_col), pl.max(week_start_col)])
                min_date = date_range.get_column(week_start_col).min()
                max_date = date_range.get_column(week_start_col).max()
            except:
                min_date = pd.Timestamp('2020-01-01')
                max_date = pd.Timestamp('2024-12-31')
        else:
            # Fallback: create dummy dates
            min_date = pd.Timestamp('2020-01-01')
            max_date = pd.Timestamp('2024-12-31')
            week_start_col = None
    else:
        week_start_col = "week_start"
        try:
            date_range = subset.select([pl.min(week_start_col), pl.max(week_start_col)])
            min_date = date_range.get_column(week_start_col).min()
            max_date = date_range.get_column(week_start_col).max()
        except Exception as e:
            # Remove error message display - st.error(f"Error accessing week_start column: {str(e)}")
            # Try to fix by removing duplicates again
            subset, extra_duplicates = remove_duplicate_columns(subset)
            # Remove status message - if extra_duplicates > 0:
            #     st.info(f"Removed {extra_duplicates} additional duplicate columns")
            try:
                date_range = subset.select([pl.min(week_start_col), pl.max(week_start_col)])
                min_date = date_range.get_column(week_start_col).min()
                max_date = date_range.get_column(week_start_col).max()
            except:
                min_date = pd.Timestamp('2020-01-01')
                max_date = pd.Timestamp('2024-12-31')
    
    # Calculate test period
    test_start_date = pd.to_datetime(test_start).date()
    test_end_date = pd.to_datetime(test_end).date()
    
    # Use the correct date column for filtering
    if "week_start" in subset.columns:
        train_weeks = len(subset.filter(pl.col("week_start") < test_start_date))
        test_weeks = len(subset.filter(
            (pl.col("week_start") >= test_start_date) & 
            (pl.col("week_start") <= test_end_date)
        ))
    else:
        # Fallback calculations if no date column
        train_weeks = max(total_weeks - 14, 0)  # Assume last 14 weeks for test
        test_weeks = min(14, total_weeks)
    
    # Calculate data quality metrics - ensure session_count column exists
    if "session_count" in subset.columns:
        session_counts = subset.select("session_count").to_pandas()["session_count"]
        avg_sessions = session_counts.mean()
        std_sessions = session_counts.std()
        zero_weeks = (session_counts == 0).sum()
        volatility = (std_sessions / avg_sessions * 100) if avg_sessions > 0 else 0
    else:
        # Fallback if no session_count column
        avg_sessions = 100.0  # Dummy value
        volatility = 20.0     # Dummy volatility
        zero_weeks = 0
    
    # Assess data quality
    quality_issues = []
    recommendations = []
    
    # Length assessment
    if total_weeks < 20:
        quality_issues.append(f"Very short history: {total_weeks} weeks")
        recommendations.append("Use higher aggregation level (state/program/global)")
    elif total_weeks < 52:
        quality_issues.append(f"Short history: {total_weeks} weeks")
        recommendations.append("Consider ensemble with simple methods")
    
    if train_weeks < 15:
        quality_issues.append(f"Insufficient training data: {train_weeks} weeks")
        recommendations.append("Increase aggregation level or adjust test period")
    
    # Data quality assessment
    if zero_weeks > total_weeks * 0.3:
        quality_issues.append(f"High zero rate: {zero_weeks}/{total_weeks} weeks")
        recommendations.append("Consider data pooling or imputation")
    
    if volatility > 100:
        quality_issues.append(f"High volatility: {volatility:.1f}%")
        recommendations.append("Use robust models (Random Forest, Ridge)")
    
    # Determine forecasting difficulty
    if len(quality_issues) == 0:
        difficulty = "Easy"
        color = "green"
    elif len(quality_issues) <= 2:
        difficulty = "Medium"
        color = "orange"
    else:
        difficulty = "Hard"
        color = "red"
    
    return {
        "total_weeks": total_weeks,
        "train_weeks": train_weeks,
        "test_weeks": test_weeks,
        "date_range": (min_date if 'min_date' in locals() else pd.Timestamp('2020-01-01'), 
                      max_date if 'max_date' in locals() else pd.Timestamp('2024-12-31')),
        "avg_sessions": avg_sessions,
        "volatility": volatility,
        "zero_weeks": zero_weeks,
        "quality_issues": quality_issues,
        "recommendations": recommendations,
        "difficulty": difficulty,
        "color": color,
        "can_forecast": train_weeks >= 8 and test_weeks >= 1  # Minimum viable
    }

def display_data_quality_assessment(assessment):
    """Display data quality assessment in a nice format"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Weeks", assessment["total_weeks"])
        st.metric("🎯 Training Weeks", assessment["train_weeks"])
        
    with col2:
        st.metric("📈 Avg Sessions", f"{assessment['avg_sessions']:.1f}")
        st.metric("📉 Volatility", f"{assessment['volatility']:.1f}%")
        
    with col3:
        st.metric("🔍 Test Weeks", assessment["test_weeks"])
        st.metric("⚠️ Zero Weeks", assessment["zero_weeks"])
    
    # Difficulty assessment - removed per user request
    # if assessment["color"] == "green":
    #     st.success(f"✅ **Forecasting Difficulty: {assessment['difficulty']}**")
    # elif assessment["color"] == "orange":
    #     st.warning(f"⚠️ **Forecasting Difficulty: {assessment['difficulty']}**")
    # else:
    #     st.error(f"🚨 **Forecasting Difficulty: {assessment['difficulty']}**")
    
    # Issues and recommendations
    if assessment["quality_issues"]:
        st.subheader("🔍 Data Quality Issues")
        for issue in assessment["quality_issues"]:
            st.error(f"• {issue}")
    
    if assessment["recommendations"]:
        st.subheader("💡 Recommendations")
        for rec in assessment["recommendations"]:
            st.info(f"• {rec}")

def smart_data_preparation(df, company, state, program, test_start="2024-10-01", test_end="2024-12-31"):
    """Smart data preparation that handles short data series intelligently"""
    
    # First create features if they don't exist
    from feature_engineering import create_features
    
    # Check if features already exist (like lag_1, rolling_mean_4, etc.)
    has_features = any(col.startswith(('lag_', 'rolling_', 'momentum_', 'trend_')) for col in df.columns)
    
    if not has_features:
        # Create features for the data
        df = create_features(df, optimized=False)  # Use base features for smart preparation
    
    # Apply comprehensive duplicate column removal after feature creation
    df, duplicates_removed = remove_duplicate_columns(df)
    # Remove status message - if duplicates_removed > 0:
    #     st.warning(f"⚠️ Removed {duplicates_removed} duplicate columns after feature creation")
    
    # First assess data quality
    assessment = assess_data_quality(df, company, state, program, test_start, test_end)
    
    # Display assessment
    with st.expander("📊 Data Quality Assessment", expanded=True):
        display_data_quality_assessment(assessment)
    
    # If data is too short, recommend higher aggregation
    if not assessment["can_forecast"]:
        st.error("❌ **Cannot forecast with current selection**")
        # Remove verbose messages
        # st.info("**Automatic Escalation:** Trying higher aggregation levels...")
        
        # Try different aggregation levels
        if company != "ALL COMPANIES":
            # Try company-level aggregation
            # st.info("🔄 Trying company-level aggregation...")
            return prepare_aggregated_data(df, company, "ALL STATES", "ALL PROGRAMS", "Company Total", test_start, test_end)
        else:
            # Try global aggregation
            # st.info("🔄 Trying global aggregation...")
            return prepare_aggregated_data(df, company, state, program, "Global Total", test_start, test_end)
    
    # If data is short but viable, use adaptive approach
    if assessment["difficulty"] == "Hard":
        st.warning("🤖 **Using Adaptive Approach for Short Data**")
        # Use simpler feature set and ensemble approach
        return prepare_model_data_adaptive(df, company, state, program, test_start, test_end, assessment)
    
    # Use normal approach for good data
    return prepare_model_data(df, company, state, program, test_start, test_end)

def prepare_model_data_adaptive(df, company, state, program, test_start="2024-10-01", test_end="2024-12-31", assessment=None):
    """Adaptive data preparation for short/difficult series"""
    
    # Filter for specific combination
    subset = df.filter(
        (pl.col("company") == company) &
        (pl.col("state") == state) &
        (pl.col("program") == program)
    )
    
    # Split train/test
    test_start_date = pd.to_datetime(test_start).date()
    test_end_date = pd.to_datetime(test_end).date()
    train_mask = subset.get_column("week_start") < test_start_date
    test_mask = (subset.get_column("week_start") >= test_start_date) & (subset.get_column("week_start") <= test_end_date)
    train_df = subset.filter(pl.Series(train_mask))
    test_df = subset.filter(pl.Series(test_mask))
    
    # Use simplified features for short series
    if len(train_df) < 26:  # Less than 6 months
        # Use only most robust features
        feature_cols = [
            "lag_1", "lag_2", "lag_4",  # Basic lags
            "rolling_mean_4", "rolling_mean_8",  # Short-term patterns
            "week_of_year", "month", "quarter",  # Seasonality
            "is_winter", "is_spring", "is_summer", "is_fall"  # Seasonal indicators
        ]
        # Remove status message - st.info("🎯 **Using Simplified Features** for short data series")
    else:
        # Use extended but conservative features
        feature_cols = [
            "lag_1", "lag_2", "lag_4", "lag_8", "lag_12",
            "rolling_mean_4", "rolling_mean_8", "rolling_mean_12",
            "rolling_std_4", "rolling_std_8",
            "week_of_year", "month", "quarter",
            "is_winter", "is_spring", "is_summer", "is_fall",
            "momentum_4w", "trend_4w"
        ]
        # Remove status message - st.info("🎯 **Using Conservative Features** for medium-length series")
    
    # Filter features to only include those that exist
    available_features = [col for col in feature_cols if col in train_df.columns]
    
    if len(available_features) < 5:
        st.warning("⚠️ **Very few features available, using basic lag features only**")
        # Fall back to basic lag features
        basic_lags = [col for col in ["lag_1", "lag_2", "lag_4"] if col in train_df.columns]
        if len(basic_lags) > 0:
            available_features = basic_lags
        else:
            st.error("❌ No suitable features found")
            return None, None, None, None, None
    
    # Prepare data
    X_train = train_df.select(available_features).to_pandas()
    y_train = train_df.select("session_count").to_pandas().values.ravel()
    X_test = test_df.select(available_features).to_pandas()
    y_test = test_df.select("session_count").to_pandas().values.ravel()
    
    # Clean data
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Remove infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Time series data for statistical models
    ts_data = train_df.select(["week_start", "session_count"]).to_pandas()
    ts_data = ts_data.set_index("week_start")["session_count"].fillna(0)
    
    # Remove status message - st.success(f"✅ **Adaptive Preparation Complete:** {len(available_features)} features, {len(X_train)} training samples")
    
    return X_train, y_train, X_test, y_test, ts_data

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