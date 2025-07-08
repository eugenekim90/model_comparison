import streamlit as st
import polars as pl
import pandas as pd
import os
import numpy as np

@st.cache_data
def load_data():
    """Load the weekly sessions data"""
    try:
        # Try loading from multiple locations
        data_paths = [
            "data/weekly_sessions.parquet",
            "weekly_sessions.parquet",
            os.path.expanduser("~/Downloads/weekly_sessions.parquet")
        ]
        
        df = None
        for file_path in data_paths:
            if os.path.exists(file_path):
                df = pl.read_parquet(file_path)
                break
        
        if df is None:
            st.error("Could not find weekly_sessions.parquet")
            return None
        
        # Add configuration features
        df = add_configuration_features(df)
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def add_configuration_features(df):
    """Add configuration features as simple lookups"""
    
    # Create simple lookup dictionaries based on company patterns
    # These are placeholder lookups - in real implementation, these would come from a config table
    
    # Final Industry Group lookup (simple pattern-based)
    industry_lookup = {
        'healthcare': ['medic', 'health', 'care', 'hospital', 'clinic'],
        'technology': ['tech', 'soft', 'data', 'cloud', 'digital'],
        'finance': ['bank', 'finance', 'capital', 'invest', 'credit'],
        'retail': ['retail', 'shop', 'store', 'market', 'commerce'],
        'education': ['edu', 'school', 'university', 'college', 'learn'],
        'manufacturing': ['manu', 'factory', 'steel', 'auto', 'industrial'],
        'consulting': ['consult', 'advisory', 'services', 'solutions']
    }
    
    def get_industry_group(company_name):
        """Simple pattern matching for industry group"""
        if pd.isna(company_name):
            return 'other'
        
        company_lower = str(company_name).lower()
        for industry, keywords in industry_lookup.items():
            if any(keyword in company_lower for keyword in keywords):
                return industry
        return 'other'
    
    # Cancellation Limit lookup (based on company size/type)
    def get_cancellation_limit(company_name):
        """Simple lookup for cancellation limit"""
        if pd.isna(company_name):
            return 30
        
        # Simple pattern: larger companies get higher limits
        company_lower = str(company_name).lower()
        if any(word in company_lower for word in ['corp', 'inc', 'ltd', 'llc', 'enterprise']):
            return 60
        elif any(word in company_lower for word in ['group', 'company', 'systems']):
            return 45
        else:
            return 30
    
    # Configuration type lookup
    def get_configuration(company_name, program):
        """Simple lookup for configuration type"""
        if pd.isna(program) or program == 'None':
            return 'standard'
        
        program_lower = str(program).lower() if program else 'standard'
        if 'premium' in program_lower or 'enterprise' in program_lower:
            return 'premium'
        elif 'basic' in program_lower:
            return 'basic'
        else:
            return 'standard'
    
    # Add lookup columns
    df = df.with_columns([
        # Final Industry Group
        pl.col('company').map_elements(get_industry_group, return_dtype=pl.Utf8).alias('final_industry_group'),
        
        # Cancellation Limit  
        pl.col('company').map_elements(get_cancellation_limit, return_dtype=pl.Int32).alias('cancellation_limit'),
        
        # Configuration
        pl.struct(['company', 'program']).map_elements(
            lambda x: get_configuration(x['company'], x['program']), 
            return_dtype=pl.Utf8
        ).alias('configuration'),
        
        # Subscriber/Non-Subscriber counts (derived from existing metrics)
        # Using session patterns to estimate subscriber vs non-subscriber
        pl.when(pl.col('lc_ns_sessions') > 0)
        .then(pl.col('lc_ns_sessions'))
        .otherwise(0)
        .alias('non_subscriber_count'),
        
        pl.when(pl.col('attended_sessions') > pl.col('lc_ns_sessions'))
        .then(pl.col('attended_sessions') - pl.col('lc_ns_sessions'))
        .otherwise(pl.col('attended_sessions'))
        .alias('subscriber_count'),
        
        # Rename target column for consistency
        pl.col('attended_sessions').alias('session_count')
    ])
    
    return df

def get_available_options(df):
    """Get available companies, states, and programs"""
    companies = sorted(df.select("company").unique().to_pandas()["company"].dropna().tolist())
    states = sorted(df.select("state").unique().to_pandas()["state"].dropna().tolist())
    programs = sorted(df.select("program").unique().to_pandas()["program"].dropna().tolist())
    
    return companies, states, programs

def get_simple_date_options(df):
    """Get simple date range information"""
    dates = df.select("week_start").to_pandas()["week_start"]
    
    min_date = pd.to_datetime(dates.min())
    max_date = pd.to_datetime(dates.max())
    
    total_weeks = len(dates.unique())
    
    # Default test start: October 1, 2024 (as requested by user)
    default_test_start = pd.to_datetime('2024-10-01')
    
    # Make sure the default is within data bounds
    if default_test_start < min_date:
        # If October 2024 is before data starts, use 80% through data
        weeks_for_train = int(total_weeks * 0.8)
        default_test_start = min_date + pd.Timedelta(weeks=weeks_for_train)
    elif default_test_start > max_date:
        # If October 2024 is after data ends, use 80% through data
        weeks_for_train = int(total_weeks * 0.8) 
        default_test_start = min_date + pd.Timedelta(weeks=weeks_for_train)
    
    # Default test end: December 31, 2024 (as requested by user)
    default_test_end = pd.to_datetime('2024-12-31')
    
    # Make sure test end is reasonable
    if default_test_end > max_date:
        default_test_end = max_date
    elif default_test_end < default_test_start:
        default_test_end = max_date
    
    # Constraints for test period
    min_test_start = min_date + pd.Timedelta(weeks=15)  # Need at least 15 weeks for training
    max_test_start = max_date - pd.Timedelta(weeks=4)   # Need at least 4 weeks for testing
    
    # Ensure default is within valid range
    if default_test_start < min_test_start:
        default_test_start = min_test_start
    elif default_test_start > max_test_start:
        default_test_start = max_test_start
    
    return {
        'min_date': min_date,
        'max_date': max_date,
        'total_weeks': total_weeks,
        'default_test_start': default_test_start,
        'default_test_end': default_test_end,  # Add default test end
        'min_test_start': min_test_start,
        'max_test_start': max_test_start
    }

def filter_data_by_selection(df, selected_company, selected_state, selected_program):
    """Filter data based on user selection"""
    filtered_df = df
    
    if selected_company != "ALL COMPANIES":
        filtered_df = filtered_df.filter(pl.col("company") == selected_company)
    
    if selected_state != "ALL STATES":
        filtered_df = filtered_df.filter(pl.col("state") == selected_state)
    
    if selected_program != "ALL PROGRAMS":
        filtered_df = filtered_df.filter(pl.col("program") == selected_program)
    
    return filtered_df

def get_hierarchical_data(df, selected_company, selected_state, selected_program):
    """Get hierarchical data based on selection level"""
    
    # Determine the appropriate level based on selection
    if (selected_company != "ALL COMPANIES" and 
        selected_state != "ALL STATES" and 
        selected_program != "ALL PROGRAMS"):
        # Most specific level
        level_filter = "company_program_state"
    elif (selected_company != "ALL COMPANIES" and 
          selected_program != "ALL PROGRAMS"):
        # Company + Program level
        level_filter = "company_program"
    else:
        # Company level
        level_filter = "company"
    
    # Filter by level and selections
    filtered_df = df.filter(pl.col("level") == level_filter)
    
    # Apply additional filters
    if selected_company != "ALL COMPANIES":
        filtered_df = filtered_df.filter(pl.col("company") == selected_company)
    
    if selected_state != "ALL STATES":
        filtered_df = filtered_df.filter(pl.col("state") == selected_state)
    
    if selected_program != "ALL PROGRAMS":
        filtered_df = filtered_df.filter(pl.col("program") == selected_program)
    
    return filtered_df 