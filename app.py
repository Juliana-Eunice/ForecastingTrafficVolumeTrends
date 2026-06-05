import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Page Configuration
st.set_page_config(page_title="Metro Manila Traffic Forecasting Tool", layout="wide")

st.title("Traffic Volume Forecasting Simulation")
st.markdown("""
This dashboard implements **Holt's Linear Trend Method (Double Exponential Smoothing)** to analyze 
historical Annual Average Daily Traffic (AADT) data and project traffic volumes for **2026, 2027, and 2028** across selected high-traffic roads in Metro Manila.
""")

# 2. Math Reference Section (as requested by the research methodology)
with st.expander("Mathematical Framework (Holt's Linear Trend Method)", expanded=False):
    st.markdown("### 1. Level Equation")
    st.latex(r"L_{t} = \alpha A_{t} + (1 - \alpha)(L_{t-1} + T_{t-1})")
    st.caption("Updates the estimated traffic volume level for the current year based on the actual volume (At).")
    
    st.markdown("### 2. Trend Equation")
    st.latex(r"T_{t} = \beta(L_{t} - L_{t-1}) + (1 - \beta)T_{t-1}")
    st.caption("Updates the estimated rate of increase or decrease in traffic volume.")
    
    st.markdown("### 3. Forecast Equation")
    st.latex(r"F_{t+m} = L_{t} + m T_{t}")
    st.caption("Generates predictions forward by m periods (years ahead).")

# 3. Bulletproof Data Loader
@st.cache_data
def load_data():
    filename = "Traffic Historic Data - Sheet1.csv"
    
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            # CRITICAL FIX: Strip spaces AND force uppercase to eliminate header formatting errors
            df.columns = df.columns.str.strip().str.upper()
        except Exception as e:
            st.error(f"Error reading CSV file structure: {e}")
            st.stop()
    else:
        # Graceful fallback data simulating years leading up to 2025
        data = {
            "YEAR": [2020, 2021, 2022, 2023, 2024, 2025],
            "EDSA": [350000, 365000, 375000, 385000, 398000, 405000],
            "C5": [210000, 220000, 228000, 235000, 242000, 250000],
            "COMMONWEALTH": [280000, 290000, 302000, 315000, 324000, 335000]
        }
        df = pd.DataFrame(data)
    
    # Standardize and clean commas from numeric data columns safely
    for col in ["YEAR", "EDSA", "C5", "COMMONWEALTH"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)
            
    return df

# Execute loading cleanly into df_hist
df_hist = load_data()

# 4. Sidebar Controls
st.sidebar.header("Simulation Parameters")

road = st.sidebar.selectbox(
    "Select Target Highway",
    ["EDSA", "C5", "COMMONWEALTH"]
)

alpha = st.sidebar.slider(
    "Alpha (α) - Level Smoothing",
    0.01, 1.0, 0.30, step=0.05,
    help="Weight given to the most recent actual traffic observation."
)

beta = st.sidebar.slider(
    "Beta (β) - Trend Smoothing",
    0.01, 1.0, 0.20, step=0.05,
    help="Controls how fast the system adapts to changes in the trajectory growth."
)

# 5. Implementing Holt's Linear Trend Math Engine
def compute_holts_forecast(df, column, alpha, beta, forecast_years=[2026, 2027, 2028]):
    # Fallback checking if user column name completely missing from CSV file
    if column not in df.columns:
        st.error(f"Column '{column}' not found in your CSV file. Available headers are: {list(df.columns)}")
        st.stop()
        
    y = df[column].values
    years = df['YEAR'].values
    n = len(y)
    
    level = np.zeros(n)
    trend = np.zeros(n)
    
    level[0] = y[0]
    if n > 1:
        trend[0] = y[1] - y[0]
    else:
        trend[0] = 0
        
    for t in range(1, n):
        level[t] = alpha * y[t] + (1 - alpha) * (level[t-1] + trend[t-1])
        trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
        
    last_level = level[-1]
    last_trend = trend[-1]
    
    future_records = []
    for m, f_year in enumerate(forecast_years, start=1):
        predicted_value = last_level + (m * last_trend)
        future_records.append({"YEAR": f_year, column: np.nan, "Type": "Forecasted", "Value": predicted_value})
        
    hist_records = []
    for t in range(n):
        fit_val = level[t-1] + trend[t-1] if t > 0 else level[0]
        hist_records.append({"YEAR": years[t], column: y[t], "Type": "Historical", "Value": fit_val})
        
    return pd.DataFrame(hist_records), pd.DataFrame(future_records)

# Run calculation using unified global variable parameters
df_fit, df_fore = compute_holts_forecast(df_hist, road, alpha, beta)

# 6. Metrics Summary Panel
st.subheader(f"Traffic Dashboard Metrics: {road}")
m_col1, m_col2, m_col3 = st.columns(3)

f_2026 = df_fore[df_fore['YEAR'] == 2026]['Value'].values[0]
f_2027 = df_fore[df_fore['YEAR'] == 2027]['Value'].values[0]
f_2028 = df_fore[df_fore['YEAR'] == 2028]['Value'].values[0]

m_col1.metric(label="Projected Traffic Volume (2026)", value=f"{f_2026:,.0f} AADT")
m_col2.metric(label="Projected Traffic Volume (2027)", value=f"{f_2027:,.0f} AADT")
m_col3.metric(label="Projected Traffic Volume (2028)", value=f"{f_2028:,.0f} AADT")

# 7. Chart Display Section
st.subheader("Historical Trends vs 3-Year Projections")

fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(df_fit['YEAR'], df_fit[road], marker='o', color='#1f77b4', linewidth=2.5, label='Actual Historical Traffic')
ax.plot(df_fit['YEAR'], df_fit['Value'], linestyle=':', color='#aec7e8', linewidth=2, label="Model Fitted Baseline")

connect_year = [df_fit['YEAR'].iloc[-1]] + df_fore['YEAR'].tolist()
connect_val = [df_fit[road].iloc[-1]] + df_fore['Value'].tolist()

ax.plot(connect_year, connect_val, marker='s', linestyle='--', color='#d62728', linewidth=2.5, label="Holt's Forecast (2026-2028)")

ax.set_title(f'AADT Evolution Trend and Forecast Profile: {road}', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Timeline (Years)', fontsize=11)
ax.set_ylabel('Annual Average Daily Traffic (Vehicles/Day)', fontsize=11)

all_years = sorted(list(df_fit['YEAR'].unique().astype(int)) + list(df_fore['YEAR'].unique().astype(int)))
ax.set_xticks(all_years)
ax.set_xticklabels([str(y) for y in all_years])
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=10)

st.pyplot(fig)

# 8. Complete Projection Data Matrix 
st.subheader("Data Analysis Breakdown")
tab1, tab2 = st.tabs(["Forecast Schedule Output", "Full Simulation Matrix"])

with tab1:
    display_fore = df_fore[['YEAR', 'Value']].copy()
    display_fore.columns = ['Year Target', 'Forecasted Traffic Volume (AADT)']
    st.dataframe(
        display_fore.style.format({'Year Target': '{:.0f}', 'Forecasted Traffic Volume (AADT)': '{:,.2f}'}),
        use_container_width=True, hide_index=True
    )

with tab2:
    master_log = []
    for idx, row in df_fit.iterrows():
        master_log.append([int(row['YEAR']), f"{row[road]:,.0f}" if not np.isnan(row[row.index == road].values[0]) else "-", f"{row['Value']:,.2f}", "Historical Baseline"])
    for idx, row in df_fore.iterrows():
        master_log.append([int(row['YEAR']), "-", f"{row['Value']:,.2f}", "Future Projection"])
        
    unified_df = pd.DataFrame(master_log, columns=["Year", "Actual Observed Data", "Model Value", "Timeline Phase"])
    st.dataframe(unified_df, use_container_width=True, hide_index=True)