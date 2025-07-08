import polars as pl
import pandas as pd
import numpy as np
import streamlit as st

def create_features(df, optimized=False):
    """Create simple, reliable features for time series forecasting"""
    
    # Always use simple features - ignore optimized parameter
    available_cols = df.columns
    
    # Determine sort columns based on what's available
    if all(col in available_cols for col in ["company", "state", "program"]):
        sort_cols = ["company", "state", "program", "week_start"]
    else:
        sort_cols = ["week_start"]
    
    df = df.sort(sort_cols)
    
    # Add simple time features
    df = df.with_columns([
        # Basic time features
        pl.col("week_start").dt.week().alias("week_of_year"),
        pl.col("week_start").dt.month().alias("month"),
        pl.col("week_start").dt.quarter().alias("quarter"),
        pl.col("week_start").dt.year().alias("year"),
        
        # Simple seasonal indicators
        pl.when(pl.col("week_start").dt.month().is_in([12, 1, 2])).then(1).otherwise(0).alias("is_winter"),
        pl.when(pl.col("week_start").dt.month().is_in([3, 4, 5])).then(1).otherwise(0).alias("is_spring"),
        pl.when(pl.col("week_start").dt.month().is_in([6, 7, 8])).then(1).otherwise(0).alias("is_summer"),
        pl.when(pl.col("week_start").dt.month().is_in([9, 10, 11])).then(1).otherwise(0).alias("is_fall"),
    ])
    
    # Add simple lag features
    df = _add_simple_lag_features(df)
    
    # Add simple rolling features
    df = _add_simple_rolling_features(df)
    
    # Only add categorical encoding if grouping columns exist
    if all(col in available_cols for col in ["company", "state", "program"]):
        df = _add_simple_categorical_encoding(df)
    
    return df

def _add_simple_lag_features(df):
    """Add simple lag features"""
    
    # Check what columns are available for grouping
    available_cols = df.columns
    group_cols = []
    
    # Add grouping columns if they exist (for raw data)
    if "company" in available_cols:
        group_cols.append("company")
    if "state" in available_cols:
        group_cols.append("state")
    if "program" in available_cols:
        group_cols.append("program")
    
    # Simple lag features (weeks)
    for lag in [1, 2, 4, 12, 52]:
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

def _add_simple_rolling_features(df):
    """Add simple rolling statistics features"""
    
    # Check what columns are available for grouping
    available_cols = df.columns
    group_cols = []
    
    # Add grouping columns if they exist (for raw data)
    if "company" in available_cols:
        group_cols.append("company")
    if "state" in available_cols:
        group_cols.append("state")
    if "program" in available_cols:
        group_cols.append("program")
    
    # Simple rolling statistics
    for window in [4, 12]:
        if group_cols:
            df = df.with_columns([
                pl.col("session_count").shift(1).rolling_mean(window)
                .over(group_cols)
                .alias(f"rolling_mean_{window}"),
                
                pl.col("session_count").shift(1).rolling_std(window)
                .over(group_cols)
                .alias(f"rolling_std_{window}"),
            ])
        else:
            df = df.with_columns([
                pl.col("session_count").shift(1).rolling_mean(window)
                .alias(f"rolling_mean_{window}"),
                
                pl.col("session_count").shift(1).rolling_std(window)
                .alias(f"rolling_std_{window}"),
            ])
    
    return df

def _add_simple_categorical_encoding(df):
    """Add simple categorical encoding"""
    
    # Simple label encoding for categorical variables
    try:
        # Get unique values and handle nulls
        companies = sorted([x for x in df.select("company").unique().to_pandas()["company"].tolist() if x is not None])
        states = sorted([x for x in df.select("state").unique().to_pandas()["state"].tolist() if x is not None])
        programs = sorted([x for x in df.select("program").unique().to_pandas()["program"].tolist() if x is not None])
        
        # Create simple mappings
        company_map = {comp: i for i, comp in enumerate(companies)}
        state_map = {state: i for i, state in enumerate(states)}
        program_map = {prog: i for i, prog in enumerate(programs)}
        
        # Apply mappings
        df = df.with_columns([
            pl.col("company").map_elements(lambda x: company_map.get(x, -1), return_dtype=pl.Int32).alias("company_encoded"),
            pl.col("state").map_elements(lambda x: state_map.get(x, -1), return_dtype=pl.Int32).alias("state_encoded"),
            pl.col("program").map_elements(lambda x: program_map.get(x, -1), return_dtype=pl.Int32).alias("program_encoded"),
        ])
        
    except Exception as e:
        st.warning(f"Could not create categorical encoding: {e}")
        # Add dummy encoded columns if encoding fails
        df = df.with_columns([
            pl.lit(0).alias("company_encoded"),
            pl.lit(0).alias("state_encoded"),
            pl.lit(0).alias("program_encoded"),
        ])
    
    return df 