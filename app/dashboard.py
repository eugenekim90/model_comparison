import streamlit as st

# Set page config FIRST - before any other streamlit commands
st.set_page_config(
    page_title="Model Comparison",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import polars as pl
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import modular components
from data_loader import load_data, get_available_options, get_simple_date_options
from weekly_forecasting import weekly_forecasting
from monthly_forecasting import monthly_forecasting
from seasonal_analysis import run_seasonal_analysis

# Enable Nixtla functionality
try:
    import nixtla
    NIXTLA_MODELS_AVAILABLE = True
except ImportError:
    NIXTLA_MODELS_AVAILABLE = False

def main():
    st.title("Model Comparison")
  
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Basic data info
    st.sidebar.header("📊 Data Info")
    st.sidebar.info(f"""
    **Shape:** {len(df):,} rows × {len(df.columns)} columns
    **Companies:** {df['company'].n_unique():,} | **States:** {df['state'].n_unique():,} | **Programs:** {df['program'].n_unique():,}
    **Date Range:** {df['week_start'].min()} to {df['week_start'].max()}
    """)
    
    # Get date range information
    date_info = get_simple_date_options(df)
    
    # Sidebar selectors
    st.sidebar.header("🎯 Select Data")
    
    # Get available options
    companies, states, programs = get_available_options(df)
    selected_company = st.sidebar.selectbox("Company", ["ALL COMPANIES"] + companies)
    
    if selected_company:
        # Get available states
        if selected_company == "ALL COMPANIES":
            available_states = states
        else:
            company_df = df.filter(pl.col("company") == selected_company)
            available_states = sorted(company_df.select("state").unique().to_pandas()["state"].dropna().tolist())
        
        selected_state = st.sidebar.selectbox("State", ["ALL STATES"] + available_states)
        
        if selected_state:
            # Get available programs
            if selected_state == "ALL STATES":
                state_df = df if selected_company == "ALL COMPANIES" else df.filter(pl.col("company") == selected_company)
            else:
                state_df = df.filter(pl.col("state") == selected_state)
                if selected_company != "ALL COMPANIES":
                    state_df = state_df.filter(pl.col("company") == selected_company)
            
            available_programs = sorted(state_df.select("program").unique().to_pandas()["program"].dropna().tolist())
            selected_program = st.sidebar.selectbox("Program", ["ALL PROGRAMS"] + available_programs)
            
            if selected_program:
                # Test period selection
                st.sidebar.header("📅 Test Period")
                
                # Test dates
                test_start_date = st.sidebar.date_input(
                    "Test Start",
                    value=date_info['default_test_start'].date(),
                    min_value=date_info['min_test_start'].date(),
                    max_value=date_info['max_test_start'].date()
                )
                
                test_end_date = st.sidebar.date_input(
                    "Test End",
                    value=date_info['default_test_end'].date(),
                    min_value=test_start_date,
                    max_value=date_info['max_date'].date()
                )
                
                # Calculate split info correctly
                min_date = pd.to_datetime(date_info['min_date'])
                test_start = pd.to_datetime(test_start_date)
                test_end = pd.to_datetime(test_end_date)
                
                train_weeks = int((test_start - min_date).days / 7)
                test_weeks = int((test_end - test_start).days / 7) + 1
                
                # Display consistent test length info
                st.sidebar.info(f"**Train:** {train_weeks} weeks | **Test:** {test_weeks} weeks")
                
                # Validation
                if train_weeks < 15:
                    st.sidebar.error(f"⚠️ Training period too short: {train_weeks} weeks (minimum 15)")
                    can_run = False
                elif test_weeks < 4:
                    st.sidebar.error(f"⚠️ Testing period too short: {test_weeks} weeks (minimum 4)")
                    can_run = False
                else:
                    can_run = True
                
                # Models
                st.sidebar.header("🤖 Models")
                if NIXTLA_MODELS_AVAILABLE:
                    models_to_run = ["LightGBM", "XGBoost", "Random Forest", "ETS", "TimeGPT", "Nixtla AutoML"]
                else:
                    models_to_run = ["LightGBM", "XGBoost", "Random Forest", "ETS"]
                
                st.sidebar.write(f"Running {len(models_to_run)} models")
                
                # Initialize session state for execution tracking
                if 'forecasting_executed' not in st.session_state:
                    st.session_state.forecasting_executed = False
                
                # Run button - only enabled if validation passes
                if st.sidebar.button("🚀 Run Forecasting", type="primary", disabled=not can_run):
                    # Store parameters and mark for execution
                    st.session_state.forecasting_params = {
                        'test_split_date': test_start_date.strftime('%Y-%m-%d'),
                        'test_end_date': test_end_date.strftime('%Y-%m-%d'),
                        'models_to_run': models_to_run,
                        'use_optimized': True,
                        'selected_company': selected_company,
                        'selected_state': selected_state,
                        'selected_program': selected_program
                    }
                    st.session_state.forecasting_requested = True
                    st.session_state.forecasting_executed = False
                    st.rerun()
                
                if st.sidebar.button("🔄 Clear Results"):
                    # Clear all forecasting-related session state
                    keys_to_clear = ['forecasting_requested', 'forecasting_executed', 'forecasting_params']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                
                # Nixtla API Key
                if NIXTLA_MODELS_AVAILABLE:
                    st.sidebar.markdown("---")
                    st.sidebar.header("🔑 API Key")
                    
                    stored_key = st.session_state.get('nixtla_api_key', '')
                    api_key = st.sidebar.text_input(
                        "Nixtla API Key",
                        type="password",
                        value=stored_key,
                        placeholder="sk-...",
                        key="nixtla_api_input"
                    )
                    
                    if api_key and api_key != stored_key:
                        st.session_state['nixtla_api_key'] = api_key
                        st.sidebar.success("✅ Updated")
                    elif api_key:
                        st.session_state['nixtla_api_key'] = api_key
                        st.sidebar.success("✅ Set")
                    
                    if st.sidebar.button("🗑️ Clear Key"):
                        if 'nixtla_api_key' in st.session_state:
                            del st.session_state['nixtla_api_key']
                        st.rerun()

                # Main tabs - removed EDA
                tab1, tab2, tab3 = st.tabs(["📅 Weekly", "📆 Monthly", "🌊 Seasonal"])
                
                with tab1:
                    # Check if forecasting should be executed
                    if st.session_state.get('forecasting_requested', False) and not st.session_state.get('forecasting_executed', False):
                        params = st.session_state.get('forecasting_params', {})
                        if params:
                            with st.spinner("Running weekly forecasting..."):
                                weekly_forecasting(
                                    df, 
                                    params['selected_company'], 
                                    params['selected_state'], 
                                    params['selected_program'], 
                                    params['models_to_run'], 
                                    params['test_split_date'], 
                                    params['test_end_date'], 
                                    params['use_optimized']
                                )
                                # Mark as executed to prevent re-running
                                st.session_state.forecasting_executed = True
                                st.success("✅ Weekly forecasting completed!")
                    elif st.session_state.get('forecasting_executed', False):
                        # Show cached results
                        params = st.session_state.get('forecasting_params', {})
                        if params:
                            weekly_forecasting(
                                df, 
                                params['selected_company'], 
                                params['selected_state'], 
                                params['selected_program'], 
                                params['models_to_run'], 
                                params['test_split_date'], 
                                params['test_end_date'], 
                                params['use_optimized']
                            )
                    else:
                        st.info("Select options and click 'Run Forecasting' to start weekly analysis")
                
                with tab2:
                    # Check if forecasting should be executed for monthly
                    if st.session_state.get('forecasting_requested', False):
                        params = st.session_state.get('forecasting_params', {})
                        if params:
                            # Only run monthly if weekly is completed or we're in monthly tab
                            if st.session_state.get('forecasting_executed', False):
                                with st.spinner("Running monthly analysis..."):
                                    monthly_forecasting(
                                        df, 
                                        params['selected_company'], 
                                        params['selected_state'], 
                                        params['selected_program'], 
                                        params['models_to_run'], 
                                        params['test_split_date'], 
                                        params['test_end_date'], 
                                        params['use_optimized']
                                    )
                            else:
                                # If weekly hasn't run yet, run it first for monthly
                                with st.spinner("Running models for monthly analysis..."):
                                    monthly_forecasting(
                                        df, 
                                        params['selected_company'], 
                                        params['selected_state'], 
                                        params['selected_program'], 
                                        params['models_to_run'], 
                                        params['test_split_date'], 
                                        params['test_end_date'], 
                                        params['use_optimized']
                                    )
                                    st.session_state.forecasting_executed = True
                    else:
                        st.info("Select options and click 'Run Forecasting' to start monthly analysis")
                
                with tab3:
                    # Filter data
                    filtered_df = df
                    if selected_company != "ALL COMPANIES":
                        filtered_df = filtered_df.filter(pl.col("company") == selected_company)
                    if selected_state != "ALL STATES":
                        filtered_df = filtered_df.filter(pl.col("state") == selected_state)
                    if selected_program != "ALL PROGRAMS":
                        filtered_df = filtered_df.filter(pl.col("program") == selected_program)
                    
                    run_seasonal_analysis(filtered_df)


if __name__ == "__main__":
    main()
