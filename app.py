import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Wheat Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ─── Color Palette ────────────────────────────────────────────────────────────
# From colorhunt: #005F02  #427A43  #C0B87A  #F2E3BB
# Light mode: warm cream bg, green accents
# Dark mode:  deep green bg, gold accents

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Sora', sans-serif !important;
}

/* ── LIGHT MODE ── */
@media (prefers-color-scheme: light) {
    :root {
        --bg:      #F2E3BB;
        --card:    #ffffff;
        --border:  #C0B87A;
        --accent:  #005F02;
        --accent2: #427A43;
        --text:    #1a1a0a;
        --subtext: #427A43;
        --plot-bg: #FDF6E3;
    }
}

/* ── DARK MODE ── */
@media (prefers-color-scheme: dark) {
    :root {
        --bg:      #071407;
        --card:    #0d2a0d;
        --border:  #427A43;
        --accent:  #C0B87A;
        --accent2: #F2E3BB;
        --text:    #F2E3BB;
        --subtext: #C0B87A;
        --plot-bg: #0d2a0d;
    }
}

.stApp {
    background-color: var(--bg) !important;
}

/* Hero */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: var(--subtext);
    margin-bottom: 2rem;
}

/* Section title */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text);
    border-left: 5px solid var(--accent);
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="stMetric"] label {
    color: var(--subtext) !important;
    font-size: 0.8rem !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-weight: 700 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--subtext) !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
}

/* Sliders */
div[data-testid="stSlider"] label {
    color: var(--text) !important;
    font-weight: 500 !important;
}

/* Button */
div[data-testid="stButton"] button {
    background-color: var(--accent) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s ease;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.85 !important;
}

/* Predict result */
.predict-result {
    background: linear-gradient(135deg, #005F02, #427A43);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.2rem;
    border: 2px solid #C0B87A;
}
.predict-result .label { color: #C0B87A; font-size: 1rem; margin-bottom: 0.3rem; }
.predict-result .value { color: #F2E3BB; font-size: 3.5rem; font-weight: 700; line-height: 1; }
.predict-result .unit  { color: #C0B87A; font-size: 1.2rem; }
.predict-result .note  { color: #c8e6b0; font-size: 0.85rem; margin-top: 0.6rem; }
</style>
""", unsafe_allow_html=True)


# ─── Load & Train ─────────────────────────────────────────────────────────────
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


# ─── Plot style helper ────────────────────────────────────────────────────────
def style_plot_light(fig, ax, title=""):
    fig.patch.set_facecolor('#FDF6E3')
    ax.set_facecolor('#FDF6E3')
    ax.set_title(title, color='#005F02', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#427A43')
    ax.xaxis.label.set_color('#427A43')
    ax.yaxis.label.set_color('#427A43')
    for spine in ax.spines.values():
        spine.set_edgecolor('#C0B87A')


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌾 Wheat Yield Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Machine Learning · Linear Regression · 7 Features · Real Farm Data</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Dataset Summary", "📈 Actual vs Predicted", "🔮 Predict Yield"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Dataset Summary
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows",    f"{len(df):,}")
    col2.metric("Features Used", f"{len(features)}")
    col3.metric("Avg Yield",     f"{df['yield kg/ha'].mean():,.0f} kg/ha")
    col4.metric("Max Yield",     f"{df['yield kg/ha'].max():,.0f} kg/ha")

    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown('<div class="section-title">Yield Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    style_plot_light(fig, ax, "Distribution of Wheat Yield (kg/ha)")
    ax.hist(df['yield kg/ha'], bins=30, color='#427A43', edgecolor='#005F02', linewidth=0.7, alpha=0.85)
    ax.set_xlabel('Yield (kg/ha)')
    ax.set_ylabel('Count')
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Actual vs Predicted
# ══════════════════════════════════════════════════════════════
with tab2:
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("R² Score", f"{r2:.4f}",          help="1.0 = perfect prediction")
    col_m2.metric("RMSE",     f"{rmse:,.1f} kg/ha",  help="Average prediction error")

    st.markdown('<div class="section-title">Actual vs Predicted Scatter</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    style_plot_light(fig2, ax2, f"Actual vs Predicted  |  R² = {r2:.4f}  |  RMSE = {rmse:.1f} kg/ha")
    ax2.scatter(y_test, y_pred, alpha=0.65, color='#427A43', edgecolors='#005F02', linewidth=0.6, s=65)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax2.plot([mn, mx], [mn, mx], color='#C0B87A', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Yield (kg/ha)')
    ax2.set_ylabel('Predicted Yield (kg/ha)')
    ax2.legend(facecolor='#FDF6E3', labelcolor='#005F02', edgecolor='#C0B87A')
    st.pyplot(fig2)

    st.markdown('<div class="section-title">Sample Predictions Table</div>', unsafe_allow_html=True)
    results = pd.DataFrame({
        'Actual (kg/ha)':    y_test.values[:15].round(1),
        'Predicted (kg/ha)': y_pred[:15].round(1),
        'Error':             (y_test.values[:15] - y_pred[:15]).round(1)
    })
    st.dataframe(results, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Predict Yield
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Enter Farm Conditions</div>', unsafe_allow_html=True)
    st.caption("Adjust the sliders to match your farm's conditions and get an instant yield prediction.")

    col_a, col_b = st.columns(2)
    with col_a:
        soil_depth = st.slider("Mean Soil Depth (cm)",    float(df['mean soil depth'].min()),      float(df['mean soil depth'].max()),      float(df['mean soil depth'].mean()))
        irrigation = st.slider("Irrigation (mm)",          float(df['irrigation'].min()),           float(df['irrigation'].max()),           float(df['irrigation'].mean()))
        rain       = st.slider("Rain inc. Fallow (mm)",    float(df['rain (inc. Fallow)'].min()),   float(df['rain (inc. Fallow)'].max()),   float(df['rain (inc. Fallow)'].mean()))
        n_applied  = st.slider("N Applied (kg/ha)",        float(df['N applied'].min()),            float(df['N applied'].max()),            float(df['N applied'].mean()))
    with col_b:
        tmax = st.slider("Max Temperature (°C)",           float(df['Tmax'].min()),                 float(df['Tmax'].max()),                 float(df['Tmax'].mean()))
        tmin = st.slider("Min Temperature (°C)",           float(df['Tmin'].min()),                 float(df['Tmin'].max()),                 float(df['Tmin'].mean()))
        rs   = st.slider("Solar Radiation Rs (MJ/m²)",     float(df['Rs'].min()),                   float(df['Rs'].max()),                   float(df['Rs'].mean()))

    if st.button("🔮 Predict Yield", use_container_width=True):
        input_scaled = scaler.transform(np.array([[soil_depth, irrigation, rain, n_applied, tmax, tmin, rs]]))
        prediction   = model.predict(input_scaled)[0]
        st.markdown(f"""
        <div class="predict-result">
            <div class="label">Estimated Wheat Yield</div>
            <div class="value">{prediction:,.0f} <span class="unit">kg/ha</span></div>
            <div class="note">Based on Linear Regression · Trained on {len(df):,} farm records</div>
        </div>
        """, unsafe_allow_html=True)
