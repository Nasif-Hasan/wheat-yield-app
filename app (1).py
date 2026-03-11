import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🌾 Wheat Yield Predictor", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .main { background-color: #0f1a0f; color: #e8f5e9; }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #c8e6c9; border-left: 4px solid #388e3c; padding-left: 0.7rem; margin: 1.5rem 0 1rem 0; }
    .predict-result { background: linear-gradient(135deg, #1b5e20, #2e7d32); border-radius: 16px; padding: 1.5rem 2rem; text-align: center; margin-top: 1rem; }
    .predict-result h1 { color: #f1f8e9; font-size: 3rem; margin: 0; }
    .predict-result p { color: #dcedc8; font-size: 1rem; margin: 0.3rem 0 0 0; }
    div[data-testid="stMetric"] { background: #1b2e1b; border-radius: 10px; padding: 0.8rem; border: 1px solid #2e7d32; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train():
    df = pd.read_excel('data_manuscript_morales_villalobos.xlsx', sheet_name='output_all_farms_wheat')
    features = ['mean soil depth', 'irrigation', 'rain (inc. Fallow)', 'N applied', 'Tmax', 'Tmin', 'Rs']
    df_clean = df[features + ['yield kg/ha']].dropna()
    X = df_clean[features]
    y = df_clean['yield kg/ha']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return df_clean, model, scaler, X_test, y_test, y_pred, features

df, model, scaler, X_test, y_test, y_pred, features = load_and_train()
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.markdown("# 🌾 Wheat Yield Predictor")
st.markdown("Machine Learning model trained on real farm data to predict wheat yield (kg/ha)")

tab1, tab2, tab3 = st.tabs(["📊 Dataset Summary", "📈 Actual vs Predicted", "🔮 Predict Yield"])

with tab1:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Features Used", len(features))
    col3.metric("Avg Yield", f"{df['yield kg/ha'].mean():,.0f} kg/ha")
    col4.metric("Max Yield", f"{df['yield kg/ha'].max():,.0f} kg/ha")
    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)
    st.markdown('<div class="section-title">Yield Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f1a0f')
    ax.set_facecolor('#1b2e1b')
    ax.hist(df['yield kg/ha'], bins=30, color='#66bb6a', edgecolor='#2e7d32', linewidth=0.6)
    ax.set_xlabel('Yield (kg/ha)', color='#c8e6c9')
    ax.set_ylabel('Count', color='#c8e6c9')
    ax.set_title('Distribution of Wheat Yield', color='#a5d6a7', fontsize=13)
    ax.tick_params(colors='#81c784')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e7d32')
    st.pyplot(fig)

with tab2:
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("R² Score", f"{r2:.4f}")
    col_m2.metric("RMSE", f"{rmse:,.1f} kg/ha")
    st.markdown('<div class="section-title">Actual vs Predicted Scatter</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor='#0f1a0f')
    ax2.set_facecolor('#1b2e1b')
    ax2.scatter(y_test, y_pred, alpha=0.6, color='#66bb6a', edgecolors='#2e7d32', linewidth=0.5, s=60)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax2.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect Prediction')
    ax2.set_xlabel('Actual Yield (kg/ha)', color='#c8e6c9')
    ax2.set_ylabel('Predicted Yield (kg/ha)', color='#c8e6c9')
    ax2.set_title(f'Actual vs Predicted  |  R² = {r2:.4f}  |  RMSE = {rmse:.1f}', color='#a5d6a7', fontsize=12)
    ax2.tick_params(colors='#81c784')
    ax2.legend(facecolor='#1b2e1b', labelcolor='#c8e6c9')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#2e7d32')
    st.pyplot(fig2)
    st.markdown('<div class="section-title">Sample Predictions Table</div>', unsafe_allow_html=True)
    results = pd.DataFrame({
        'Actual (kg/ha)': y_test.values[:15].round(1),
        'Predicted (kg/ha)': y_pred[:15].round(1),
        'Error': (y_test.values[:15] - y_pred[:15]).round(1)
    })
    st.dataframe(results, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Enter Farm Conditions</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        soil_depth = st.slider("Mean Soil Depth (cm)",      float(df['mean soil depth'].min()),        float(df['mean soil depth'].max()),        float(df['mean soil depth'].mean()))
        irrigation = st.slider("Irrigation (mm)",            float(df['irrigation'].min()),             float(df['irrigation'].max()),             float(df['irrigation'].mean()))
        rain       = st.slider("Rain inc. Fallow (mm)",      float(df['rain (inc. Fallow)'].min()),     float(df['rain (inc. Fallow)'].max()),     float(df['rain (inc. Fallow)'].mean()))
        n_applied  = st.slider("N Applied (kg/ha)",          float(df['N applied'].min()),              float(df['N applied'].max()),              float(df['N applied'].mean()))
    with col_b:
        tmax = st.slider("Max Temperature (°C)",             float(df['Tmax'].min()),                   float(df['Tmax'].max()),                   float(df['Tmax'].mean()))
        tmin = st.slider("Min Temperature (°C)",             float(df['Tmin'].min()),                   float(df['Tmin'].max()),                   float(df['Tmin'].mean()))
        rs   = st.slider("Solar Radiation Rs (MJ/m²)",       float(df['Rs'].min()),                     float(df['Rs'].max()),                     float(df['Rs'].mean()))
    if st.button("🔮 Predict Yield", use_container_width=True):
        input_data   = np.array([[soil_depth, irrigation, rain, n_applied, tmax, tmin, rs]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        st.markdown(f"""
        <div class="predict-result">
            <p>Estimated Wheat Yield</p>
            <h1>{prediction:,.0f} <span style="font-size:1.5rem">kg/ha</span></h1>
            <p>Based on Linear Regression model trained on {len(df)} farm records</p>
        </div>
        """, unsafe_allow_html=True)
