# Forecasting Dashboard

A comprehensive time series forecasting dashboard for session count analysis and prediction.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run app/dashboard.py
```

3. Open your browser to `http://localhost:8501`

## Usage

1. **Select Company**: Choose from available companies
2. **Select State**: Choose specific state or "ALL STATES" for aggregation
3. **Select Program**: Choose specific program or "ALL PROGRAMS" for aggregation
4. **Choose Models**: Select which models to run
5. **Run Forecasting**: Click to start analysis

## Folder Structure

```
forecasting_dashboard/
├── app/
│   └── dashboard.py          # Main Streamlit dashboard
├── data/
│   └── sessions_with_facts.parquet  # Main dataset
├── requirements.txt          # Python dependencies
├── run_dashboard.py         # Python run script
├── run_dashboard.bat        # Windows batch file
└── README.md                # This file
```

## Data

The dashboard uses session count data with the following structure:
- Weekly time series data
- Multiple companies, states, and programs
- Features include session counts, subscriber counts, and metadata

## Models

- **LightGBM**: Gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **Random Forest**: Ensemble of decision trees
- **ETS**: Exponential smoothing state space model

## Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error 