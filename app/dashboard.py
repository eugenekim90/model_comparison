import streamlit as st
import polars as pl
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import modular components
from data_loader import load_data, get_available_options, get_simple_date_options
from weekly_forecasting import weekly_forecasting
from monthly_forecasting import monthly_forecasting
from seasonal_analysis import run_seasonal_analysis

# Set page config
st.set_page_config(
    page_title="Model Comparison",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Model Comparison")
  
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Get simple date range information
    date_info = get_simple_date_options(df)
    
    # Unified sidebar selectors
    st.sidebar.header("🎯 Select Combination")
    
    # Get available options
    companies, states, programs = get_available_options(df)
    selected_company = st.sidebar.selectbox("Select Company", ["ALL COMPANIES"] + companies)
    
    if selected_company:
        if selected_company == "ALL COMPANIES":
            company_df = df
            available_states = states
        else:
            company_df = df.filter(pl.col("company") == selected_company)
            available_states = sorted(company_df.select("us_state").unique().to_pandas()["us_state"].tolist())
        
        selected_state = st.sidebar.selectbox("Select State", ["ALL STATES"] + available_states)
        
        if selected_state:
            if selected_state == "ALL STATES":
                state_df = company_df
                available_programs = sorted(state_df.select("episode_session_type").unique().to_pandas()["episode_session_type"].tolist())
            else:
                state_df = company_df.filter(pl.col("us_state") == selected_state)
                available_programs = sorted(state_df.select("episode_session_type").unique().to_pandas()["episode_session_type"].tolist())
            
            selected_program = st.sidebar.selectbox("Select Program", ["ALL PROGRAMS"] + available_programs)
            
            if selected_program:
                # Simple Test Split Date Selector
                st.sidebar.header("📅 Test Period")
                
                # Data overview
                st.sidebar.info(f"""
                **Data Available:**
                • **From:** {date_info['min_date'].strftime('%Y-%m-%d')}
                • **To:** {date_info['max_date'].strftime('%Y-%m-%d')}
                • **Total:** {date_info['total_weeks']} weeks
                """)
                
                # Test start date
                test_start_date = st.sidebar.date_input(
                    "Test Start Date",
                    value=date_info['default_test_start'].date(),
                    min_value=date_info['min_test_start'].date(),
                    max_value=date_info['max_test_start'].date(),
                    help="When to start testing (everything before this is training data)"
                )
                
                # Test end date
                test_end_date = st.sidebar.date_input(
                    "Test End Date",
                    value=date_info['max_date'].date(),
                    min_value=test_start_date,
                    max_value=date_info['max_date'].date(),
                    help="When to end testing (usually the last date in your data)"
                )
                
                # Calculate and display split info
                train_weeks = (pd.to_datetime(test_start_date) - date_info['min_date']).days // 7
                test_weeks = (pd.to_datetime(test_end_date) - pd.to_datetime(test_start_date)).days // 7 + 1
                
                # Validation
                if train_weeks < 15:
                    st.sidebar.error(f"⚠️ Training period too short: {train_weeks} weeks (need at least 15)")
                elif test_weeks < 4:
                    st.sidebar.error(f"⚠️ Test period too short: {test_weeks} weeks (need at least 4)")
                else:
                    st.sidebar.success(f"""
                    **Split Configuration:**
                    • **Training:** {train_weeks} weeks ({date_info['min_date'].strftime('%Y-%m-%d')} to {(pd.to_datetime(test_start_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')})
                    • **Testing:** {test_weeks} weeks ({test_start_date} to {test_end_date})
                    """)
                
                # Model selection
                st.sidebar.header("🤖 Select Models")
                models_to_run = []
                
                if st.sidebar.checkbox("Linear Regression", value=True):
                    models_to_run.append("Linear Regression")
                    
                if st.sidebar.checkbox("LightGBM", value=True):
                    models_to_run.append("LightGBM")
                if st.sidebar.checkbox("XGBoost", value=True):
                    models_to_run.append("XGBoost")
                if st.sidebar.checkbox("Random Forest", value=True):
                    models_to_run.append("Random Forest")
                if st.sidebar.checkbox("ETS", value=True):
                    models_to_run.append("ETS")
                
                # Feature optimization toggle
                use_optimized = st.sidebar.checkbox("🎯 Use Optimized Features", value=True, 
                                                   help="Use enhanced features based on seasonal analysis")
                
                # Run button
                can_run = train_weeks >= 15 and test_weeks >= 4
                
                if can_run:
                    if st.sidebar.button("🚀 Run Forecasting", type="primary"):
                        # Store the run state
                        st.session_state.run_forecasting = True
                        st.session_state.test_split_date = test_start_date.strftime('%Y-%m-%d')
                        st.session_state.test_end_date = test_end_date.strftime('%Y-%m-%d')
                        st.session_state.split_info = f"Train: {train_weeks} weeks → Test: {test_weeks} weeks"
                        st.session_state.models_to_run = models_to_run
                        st.session_state.use_optimized = use_optimized
                else:
                    st.sidebar.button("🚀 Run Forecasting", type="primary", disabled=True)
                    st.sidebar.caption("Fix validation errors above to enable")
                
                # Create tabs
                tab1, tab2, tab3 = st.tabs(["📅 Weekly Forecasting", "📆 Monthly Forecasting", "🌊 Seasonal Analysis"])
                
                with tab1:
                    st.header("Weekly Level Forecasting & Evaluation")
                    
                    if st.session_state.get('run_forecasting', False):
                        # Get parameters from session state
                        test_split_date = st.session_state.get('test_split_date', '2024-10-01')
                        test_end_date = st.session_state.get('test_end_date', '2024-12-31')
                        split_info = st.session_state.get('split_info', 'Default split')
                        session_models = st.session_state.get('models_to_run', models_to_run)
                        session_optimized = st.session_state.get('use_optimized', True)
                        
                        st.info(f"**Using Split:** {split_info}")
                        if session_optimized:
                            st.success("🎯 **Using Optimized Features**")
                        weekly_forecasting(df, selected_company, selected_state, selected_program, session_models, test_split_date, test_end_date, session_optimized)
                    else:
                        st.info("👆 **Setup Steps:**")
                        st.markdown("""
                        1. Select **company, state, program**
                        2. Choose **test start/end dates**
                        3. Select **models** to run
                        4. Click **'Run Forecasting'**
                        """)
                
                with tab2:
                    st.header("Monthly Level Forecasting & Evaluation")
                    st.info("**Approach:** Weekly models → Aggregate weekly predictions → Monthly evaluation")
                    
                    if st.session_state.get('run_forecasting', False):
                        # Get parameters from session state
                        test_split_date = st.session_state.get('test_split_date', '2024-10-01')
                        test_end_date = st.session_state.get('test_end_date', '2024-12-31')
                        split_info = st.session_state.get('split_info', 'Default split')
                        session_models = st.session_state.get('models_to_run', models_to_run)
                        session_optimized = st.session_state.get('use_optimized', True)
                        
                        st.info(f"**Using Split:** {split_info}")
                        if session_optimized:
                            st.success("🎯 **Using Optimized Features**")
                        monthly_forecasting(df, selected_company, selected_state, selected_program, session_models, test_split_date, test_end_date, session_optimized)
                    else:
                        st.info("👆 **Setup Steps:**")
                        st.markdown("""
                        1. Select **company, state, program**
                        2. Choose **test start/end dates**
                        3. Select **models** to run  
                        4. Click **'Run Forecasting'**
                        
                        **Monthly Process:**
                        - Train models on weekly data
                        - Generate weekly predictions
                        - Aggregate predictions to monthly level
                        - Compare vs actual monthly totals
                        """)
                
                with tab3:
                    st.header("Advanced Seasonal Pattern Analysis")
                    st.info("**Deep dive into seasonal patterns, dips, and advanced feature engineering**")
                    
                    # Filter data based on selections
                    filtered_df = df
                    if selected_company != "ALL COMPANIES":
                        filtered_df = filtered_df.filter(pl.col("company") == selected_company)
                    if selected_state != "ALL STATES":
                        filtered_df = filtered_df.filter(pl.col("us_state") == selected_state)
                    if selected_program != "ALL PROGRAMS":
                        filtered_df = filtered_df.filter(pl.col("episode_session_type") == selected_program)
                    
                    # Run seasonal analysis
                    run_seasonal_analysis(filtered_df)

if __name__ == "__main__":
    main()
