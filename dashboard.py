import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# --- Page config ---
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Apply custom style ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #121212;
        color: #f1f1f1;
    }
    h1, h2, h3 {
        color: #00b4d8;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .stSlider > div[data-baseweb="slider"] > div {
        background: linear-gradient(90deg, #00b4d8 0%, #48cae4 100%);
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(to right, #00b4d8, #48cae4);
        color: #000;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: box-shadow 0.3s ease-in-out;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        box-shadow: 0 0 12px #00b4d8aa;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Controls")
    forecast_period = st.slider("Forecast Days", 30, 180, 90)
    st.markdown("---")
    st.caption("Built with Facebook Prophet & Streamlit")

# --- Title Block ---
st.markdown('<div class="glass-card"><h1>📊 AI Sales Forecasting Dashboard</h1><p>Predict future trends based on historical sales using Prophet.</p></div>', unsafe_allow_html=True)

# --- Load and prep data ---
df = pd.read_csv("Superstore.csv", encoding="latin1")
df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.groupby('Order Date')['Sales'].sum().reset_index()
df.columns = ['ds', 'y']

# --- Train model ---
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=forecast_period)
forecast = model.predict(future)

# --- Forecast Plot ---
st.subheader(f"📅 Forecast for Next {forecast_period} Days")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1, use_container_width=True)

# --- Trend and seasonality ---
st.subheader("📉 Trend & Seasonality")
fig2 = plot_components_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

# --- Download forecast ---
st.subheader("📁 Export Forecast CSV")
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Forecast", csv, "forecast.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.caption("🚀 Internship Project by Anand | ML Track – Future Interns")
