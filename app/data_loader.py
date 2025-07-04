import streamlit as st
import polars as pl
import pandas as pd
import os

@st.cache_data
def load_data():
    """Load the sessions data"""
    try:
        # Try to load from local data directory first
        if os.path.exists("data/sessions_with_facts.parquet"):
            df = pl.read_parquet("data/sessions_with_facts.parquet")
            return df
        # Try relative path for different deployment structures
        elif os.path.exists("../data/sessions_with_facts.parquet"):
            df = pl.read_parquet("../data/sessions_with_facts.parquet")
            return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_available_options(df):
    """Get available companies, states, and programs"""
    companies = sorted(df.select("company").unique().to_pandas()["company"].tolist())
    states = sorted(df.select("us_state").unique().to_pandas()["us_state"].tolist())
    programs = sorted(df.select("episode_session_type").unique().to_pandas()["episode_session_type"].tolist())
    return companies, states, programs

def get_data_date_range(df):
    """Get the date range of the data"""
    min_date = df.select(pl.min("week_start")).to_pandas().iloc[0, 0]
    max_date = df.select(pl.max("week_start")).to_pandas().iloc[0, 0]
    return min_date, max_date

def get_simple_date_options(df):
    """Get simple date range for test split selection"""
    min_date, max_date = get_data_date_range(df)
    
    # Convert to pandas datetime for easier manipulation
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)
    
    # Calculate total weeks
    total_weeks = (max_date - min_date).days // 7
    
    # Suggest a reasonable default test start (last 12 weeks)
    default_test_weeks = min(12, max(6, total_weeks // 4))
    default_test_start = max_date - pd.Timedelta(weeks=default_test_weeks)
    
    # Calculate bounds for reasonable test splits
    min_test_start = min_date + pd.Timedelta(weeks=max(15, total_weeks // 4))  # At least 15 weeks training
    max_test_start = max_date - pd.Timedelta(weeks=4)  # At least 4 weeks testing
    
    return {
        "min_date": min_date,
        "max_date": max_date,
        "total_weeks": total_weeks,
        "default_test_start": default_test_start,
        "min_test_start": min_test_start,
        "max_test_start": max_test_start
    }

def filter_data_by_selection(df, selected_company, selected_state, selected_program):
    """Filter data based on user selection"""
    filtered_df = df
    
    if selected_company != "ALL COMPANIES":
        filtered_df = filtered_df.filter(pl.col("company") == selected_company)
    
    if selected_state != "ALL STATES":
        filtered_df = filtered_df.filter(pl.col("us_state") == selected_state)
    
    if selected_program != "ALL PROGRAMS":
        filtered_df = filtered_df.filter(pl.col("episode_session_type") == selected_program)
    
    return filtered_df 