import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Traffic Data Analytics", layout="wide")

st.title("Historical Traffic Data Analysis")
st.markdown("""
This dashboard visualizes historical traffic volumes across major roads and applies 
**Simple Exponential Smoothing** to highlight long-term trends.
""")


st.latex(r"F_t = \alpha A_{t-1} + (1 - \alpha)F_{t-1}")
st.caption("Where $alpha$ represents the smoothing factor, $A_{t-1}$ is the actual traffic from the previous year, and $F$ represents the forecast.")


@st.cache_data
def load_data():
    
    df = pd.read_csv("Traffic Historic Data - Sheet1.csv")
    
    for col in ["EDSA", "C5", "COMMONWEALTH"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()


st.sidebar.header("Dashboard Controls")

road = st.sidebar.selectbox(
    "Choose Road",
    ["EDSA", "C5", "COMMONWEALTH"]
)

alpha = st.sidebar.slider(
    "Alpha Value",
    0.1,
    1.0,
    0.3
)


df['Smoothed Traffic'] = df[road].ewm(alpha=alpha, adjust=False).mean()


st.subheader(r"Metrics Summary for " + road)
col1, col2, col3 = st.columns(3)

latest_actual = df[road].iloc[-1]
latest_smoothed = df['Smoothed Traffic'].iloc[-1]
avg_traffic = df[road].mean()

col1.metric(label=f"Latest Year Actual ({int(df['YEAR'].iloc[-1])})", value=f"{latest_actual:,.0f}")
col2.metric(label=f"Latest Smoothed Estimate", value=f"{latest_smoothed:,.0f}")
col3.metric(label="Historical Average Traffic", value=f"{avg_traffic:,.0f}")


st.subheader("Traffic Trend Visualization")

fig, ax = plt.subplots(figsize=(12, 6))


ax.plot(df['YEAR'], df[road], marker='o', linewidth=2.5, color='blue', label=f'Actual Traffic on {road}')


ax.plot(
    df['YEAR'], 
    df['Smoothed Traffic'], 
    marker='s', 
    linestyle='--', 
    linewidth=2, 
    color='orange', 
    label=fr'$F_t = {alpha} \cdot A_{{t-1}} + (1-{alpha})F_{{t-1}}$'
)


ax.set_title(f'Traffic Volume Trend over Years for {road}', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Traffic Volume (Number of Vehicles)', fontsize=12)
ax.set_xticks(df['YEAR'])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=11)

st.pyplot(fig)


with st.expander("Historical Data Table"):
    
    formatted_df = df.copy()
    for col in [road, 'Smoothed Traffic']:
        formatted_df[col] = formatted_df[col].map('{:,.2f}'.format)
    st.dataframe(formatted_df[['YEAR', road, 'Smoothed Traffic']].reset_index(drop=True), use_container_width=True)
