"""
Rainfall AI - Professional Weather Analytics Dashboard
Streamlit web application for machine-learning based rainfall prediction.
Uses the pre-trained and serialized Random Forest model bundle.
"""
import os
import streamlit as st
import pandas as pd
from src.predict import load_model, predict_single, EXPECTED_FEATURES

# ==============================================================================
# APP CONFIGURATION  ←  Edit all display text and metrics here
# ==============================================================================
APP_CONFIG = {
    # ── Brand ──────────────────────────────────────────────────────────────────
    "brand_name":    "Rainfall AI",
    "brand_icon":    "🌧️",
    "brand_tagline": "Weather Intelligence & Rainfall Prediction",
    "page_title":    "Rainfall AI • Weather Intelligence Dashboard",

    # ── Verified model metrics (from 74-sample holdout test set) ───────────────
    "metrics": [
        {"label": "Test Accuracy",  "value": "81.08%"},
        {"label": "Rain Recall",    "value": "96.00%"},
        {"label": "Test F1 Score",  "value": "0.8727"},
        {"label": "5-Fold CV F1",   "value": "0.8790"},
    ],
    "metrics_note": "Measured on the project's 74-sample stratified holdout test set.",

    # ── Info tabs ──────────────────────────────────────────────────────────────
    "info_tabs": [
        {
            "tab":  "⚙️  How It Works",
            "steps": [
                ("1", "Surface Data Input",
                 "User provides 7 meteorological observations: pressure, dew point, humidity, "
                 "cloud cover, sunshine hours, wind direction, and wind speed."),
                ("2", "Preprocessing",
                 "Inputs are validated and passed through the stored median imputer fitted "
                 "exclusively on training data — no test-set leakage."),
                ("3", "Model Inference",
                 "The tuned Random Forest (150 estimators) aggregates votes across all trees "
                 "and returns a binary prediction plus a calibrated probability score."),
                ("4", "Result Display",
                 "Rain probability, model confidence, and a visual gauge are rendered "
                 "immediately in the dashboard without any page reload."),
            ],
        },
        {
            "tab": "🧠  Model Details",
            "models": [
                {
                    "name":   "Logistic Regression (Baseline)",
                    "badge":  "BASELINE",
                    "color":  "#6366f1",
                    "detail": "Linear classifier with L2 regularisation. SMOTE oversampling applied to address class imbalance. "
                              "Achieved 83.78% accuracy and F1 0.8776 on holdout — retained as the reproducibility baseline.",
                },
                {
                    "name":   "Decision Tree",
                    "badge":  "COMPARISON",
                    "color":  "#f59e0b",
                    "detail": "Single tree pruned to max_depth=3. Interpretable but high-variance; used as a complexity benchmark.",
                },
                {
                    "name":   "Tuned Random Forest",
                    "badge":  "SELECTED",
                    "color":  "#10b981",
                    "detail": "150 bagged estimators, max_depth=6, min_samples_leaf=2. "
                              "Selected for highest CV F1 (0.8790) and 96% rain-event recall on the holdout set.",
                },
            ],
        },
        {
            "tab": "📊  Feature Importance",
            "note": "Gini impurity importance from the production Random Forest. "
                    "Describes predictive utility — does not imply causation.",
            "features": [
                {"name": "Cloud Cover",    "pct": 28.1},
                {"name": "Sunshine",       "pct": 27.0},
                {"name": "Humidity",       "pct": 13.9},
                {"name": "Wind Speed",     "pct": 10.3},
                {"name": "Dew Point",      "pct": 8.2},
                {"name": "Pressure",       "pct": 7.9},
                {"name": "Wind Direction", "pct": 4.7},
            ],
        },
    ],
}
# ==============================================================================

# Set page configuration with wide layout for modern SaaS dashboard feel
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["brand_icon"],
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Design System CSS
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Theme Overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header & Navigation Cleanup */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Top Dashboard Header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.75rem;
    }
    
    .header-left {
        display: flex;
        flex-direction: column;
    }
    
    .brand-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .brand-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 0.15rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
    }
    
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* Professional Card Styling */
    .saas-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    
    .card-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.25rem;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    
    .card-desc {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 1.25rem;
    }

    /* Result Card Styles */
    .result-card-empty {
        background: #111827;
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .result-card-rain {
        background: radial-gradient(circle at top right, rgba(37, 99, 235, 0.15), transparent 70%), #111827;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 14px;
        padding: 1.6rem;
        box-shadow: 0 8px 30px rgba(37, 99, 235, 0.15);
    }
    
    .result-card-norain {
        background: radial-gradient(circle at top right, rgba(245, 158, 11, 0.12), transparent 70%), #111827;
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 14px;
        padding: 1.6rem;
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.1);
    }
    
    .result-headline {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .metric-row:last-child {
        border-bottom: none;
    }
    
    .metric-label-text {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .metric-val-text {
        font-size: 1.1rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #f8fafc;
    }

    /* Custom Progress Bar / Meter */
    .gauge-container {
        margin: 1.25rem 0 0.5rem 0;
    }
    
    .gauge-track {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        overflow: hidden;
        position: relative;
    }
    
    .gauge-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Performance Metric Cards */
    .perf-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .perf-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        transition: border-color 0.2s ease;
    }
    
    .perf-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .perf-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    
    .perf-value {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #f1f5f9;
    }

    /* Primary CTA Button */
    .stButton>button {
        width: 100%;
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.25rem !important;
        border-radius: 8px !important;
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
        margin-top: 0.5rem;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px);
    }

    /* Input overrides for dark mode consistency */
    .stNumberInput input, .stSlider {
        color: #f1f5f9;
    }
    
    div[data-baseweb="input"] {
        background-color: #0d131f !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1320;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Footer Styling */
    .dashboard-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #475569;
        font-size: 0.8rem;
    }
    
    /* Responsive tweaks */
    @media (max-width: 768px) {
        .dashboard-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.75rem;
        }
        .perf-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .dashboard-footer {
            flex-direction: column;
            gap: 0.5rem;
            text-align: center;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_model():
    """
    Loads and caches the serialized model bundle to ensure efficient inference
    without reloading from disk on every widget interaction.
    """
    model_path = "models/rainfall_prediction_model.pkl"
    if not os.path.exists(model_path):
        if os.path.exists("../models/rainfall_prediction_model.pkl"):
            model_path = "../models/rainfall_prediction_model.pkl"
        elif os.path.exists("rainfall_prediction_model.pkl"):
            model_path = "rainfall_prediction_model.pkl"
    return load_model(model_path)


# --- Minimalist, Professional Sidebar ---
with st.sidebar:
    st.markdown('<div class="card-label">SYSTEM ARCHITECTURE</div>', unsafe_allow_html=True)
    st.markdown("### ABOUT THE MODEL")
    st.markdown("""
    **Model:** Tuned Random Forest  
    **Features:** 7 weather variables  
    **Validation:** Stratified 5-Fold CV  
    """)
    st.markdown("---")
    st.markdown("### ABOUT THIS PROJECT")
    st.markdown("""
    A machine-learning system that predicts rainfall from surface weather observations.

    Built with empirical evaluation, stratified holdout validation, and train-only median imputation.
    """)
    st.caption("Release: v1.0 • Random Forest Production Pipeline")


# --- Compact Professional Header ---
st.markdown(f"""
<div class="dashboard-header">
    <div class="header-left">
        <div class="brand-title">{APP_CONFIG['brand_icon']} {APP_CONFIG['brand_name']}</div>
        <div class="brand-subtitle">{APP_CONFIG['brand_tagline']}</div>
    </div>
    <div>
        <span class="status-badge"><span class="status-dot"></span>MODEL READY</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Load Pre-trained Model Bundle ---
try:
    model_bundle = get_cached_model()
    model_loaded = True
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.info("Please verify that `models/rainfall_prediction_model.pkl` exists or run `python main.py` to regenerate the pipeline.")
    model_loaded = False


# --- Main Dashboard Section ---
if model_loaded:
    # 2-column layout: Large Left section for inputs, Compact Right section for prediction
    col_input, col_pred = st.columns([1.5, 1.0], gap="large")

    with col_input:
        st.markdown("""
        <div class="card-label">INPUT PARAMETERS</div>
        <div class="card-title">WEATHER CONDITIONS</div>
        <div class="card-desc">Enter current surface weather observations.</div>
        """, unsafe_allow_html=True)

        # 2-column clean grid for inputs
        in_col1, in_col2 = st.columns(2)

        with in_col1:
            pressure = st.number_input(
                "Pressure (hPa)",
                min_value=950.0,
                max_value=1050.0,
                value=1012.5,
                step=0.1
            )
            dewpoint = st.number_input(
                "Dew Point (°C)",
                min_value=-10.0,
                max_value=40.0,
                value=22.1,
                step=0.1
            )
            sunshine = st.number_input(
                "Sunshine (hours)",
                min_value=0.0,
                max_value=16.0,
                value=3.5,
                step=0.1
            )
            winddirection = st.number_input(
                "Wind Direction (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=70.0,
                step=1.0
            )

        with in_col2:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )
            cloud = st.number_input(
                "Cloud Cover (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )
            windspeed = st.number_input(
                "Wind Speed (km/h)",
                min_value=0.0,
                max_value=120.0,
                value=20.4,
                step=0.5
            )

        predict_clicked = st.button("Predict Rainfall", use_container_width=True)

    with col_pred:
        st.markdown("""
        <div class="card-label">INFERENCE ENGINE</div>
        <div class="card-title">PREDICTION</div>
        <div class="card-desc">Real-time classification result & confidence.</div>
        """, unsafe_allow_html=True)

        if predict_clicked:
            input_data = {
                'pressure': pressure,
                'dewpoint': dewpoint,
                'humidity': humidity,
                'cloud': cloud,
                'sunshine': sunshine,
                'winddirection': winddirection,
                'windspeed': windspeed
            }

            try:
                # Use existing predict logic from src/predict.py
                result = predict_single(model_bundle, input_data)
                rain_pred = result['prediction']
                prob = result['probability']
                prob_pct = f"{prob * 100:.2f}%"
                
                # Model confidence: probability of the predicted outcome
                confidence = prob if rain_pred == 1 else (1.0 - prob)
                conf_pct = f"{confidence * 100:.2f}%"

                if rain_pred == 1:
                    gauge_color = "#3b82f6"
                    st.markdown(f"""
                    <div class="result-card-rain">
                        <div class="result-headline" style="color: #60a5fa;">
                            <span>🌧️</span> Rain Expected
                        </div>
                        <div class="metric-row">
                            <span class="metric-label-text">Rain Probability</span>
                            <span class="metric-val-text" style="color: #60a5fa;">{prob_pct}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label-text">Model Confidence</span>
                            <span class="metric-val-text">{conf_pct}</span>
                        </div>
                        <div class="gauge-container">
                            <div class="gauge-track">
                                <div class="gauge-fill" style="width: {prob * 100}%; background-color: {gauge_color};"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    gauge_color = "#f59e0b"
                    st.markdown(f"""
                    <div class="result-card-norain">
                        <div class="result-headline" style="color: #fbbf24;">
                            <span>☀️</span> No Rain Expected
                        </div>
                        <div class="metric-row">
                            <span class="metric-label-text">Rain Probability</span>
                            <span class="metric-val-text" style="color: #fbbf24;">{prob_pct}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label-text">Model Confidence</span>
                            <span class="metric-val-text">{conf_pct}</span>
                        </div>
                        <div class="gauge-container">
                            <div class="gauge-track">
                                <div class="gauge-fill" style="width: {prob * 100}%; background-color: {gauge_color};"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as err:
                st.error(f"Inference error: {err}")
        else:
            st.markdown("""
            <div class="result-card-empty">
                <div style="font-size: 2rem; margin-bottom: 0.75rem; opacity: 0.6;">⚡</div>
                <div style="font-weight: 600; color: #94a3b8; font-size: 0.95rem; margin-bottom: 0.35rem;">Awaiting Input</div>
                <div style="color: #475569; font-size: 0.8rem; max-width: 200px; line-height: 1.4;">
                    Enter weather conditions and click "Predict Rainfall"
                </div>
            </div>
            """, unsafe_allow_html=True)


    # ── Benchmark Metrics (auto-rendered from APP_CONFIG) ─────────────────────
    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card-label">BENCHMARK METRICS</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem;">MODEL PERFORMANCE</div>
    """, unsafe_allow_html=True)

    cards_html = '<div class="perf-grid">'
    for m in APP_CONFIG["metrics"]:
        cards_html += f"""
        <div class="perf-card">
            <div class="perf-label">{m['label']}</div>
            <div class="perf-value">{m['value']}</div>
        </div>"""
    cards_html += f'</div><div style="font-size:0.78rem;color:#475569;margin-top:0.4rem;">{APP_CONFIG["metrics_note"]}</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


    # ── Professional Documentation Tabs ───────────────────────────────────────
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card-label">PROJECT REFERENCE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:700;color:#f8fafc;margin-bottom:1rem;">DOCUMENTATION</div>',
        unsafe_allow_html=True
    )

    cfg_tabs = APP_CONFIG["info_tabs"]
    tabs = st.tabs([t["tab"] for t in cfg_tabs])

    # Tab 0 — How It Works: numbered step cards
    with tabs[0]:
        st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
        for num, title, desc in cfg_tabs[0]["steps"]:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:1.1rem;">
                <div style="min-width:32px;height:32px;border-radius:50%;
                            background:#1e3a5f;border:1px solid #2563eb;
                            display:flex;align-items:center;justify-content:center;
                            font-size:0.8rem;font-weight:700;color:#60a5fa;flex-shrink:0;">{num}</div>
                <div>
                    <div style="font-size:0.9rem;font-weight:700;color:#e2e8f0;
                                margin-bottom:0.2rem;">{title}</div>
                    <div style="font-size:0.83rem;color:#64748b;line-height:1.55;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tab 1 — Model Details: color-coded model cards with badges
    with tabs[1]:
        st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
        for mdl in cfg_tabs[1]["models"]:
            badge_bg  = mdl['color'] + '22'
            sel_border = f"border-left:3px solid {mdl['color']};" if mdl['badge'] == 'SELECTED' else ''
            st.markdown(f"""
            <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);
                        border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.75rem;
                        {sel_border}">
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
                    <span style="font-size:0.9rem;font-weight:700;color:#f1f5f9;">{mdl['name']}</span>
                    <span style="font-size:0.68rem;font-weight:700;letter-spacing:0.07em;
                                padding:0.2rem 0.55rem;border-radius:9999px;
                                background:{badge_bg};color:{mdl['color']};
                                border:1px solid {mdl['color']}40;">{mdl['badge']}</span>
                </div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.55;">{mdl['detail']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Tab 2 — Feature Importance: animated horizontal bar chart
    with tabs[2]:
        st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.82rem;color:#64748b;margin-bottom:1rem;">{cfg_tabs[2]["note"]}</div>',
            unsafe_allow_html=True
        )
        max_pct = max(f["pct"] for f in cfg_tabs[2]["features"])
        for feat in cfg_tabs[2]["features"]:
            bar_width  = (feat["pct"] / max_pct) * 100
            bar_color  = "#3b82f6" if feat["pct"] >= 20 else "#334155"
            text_color = "#60a5fa" if feat["pct"] >= 20 else "#94a3b8"
            st.markdown(f"""
            <div style="margin-bottom:0.7rem;">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:0.3rem;">
                    <span style="font-size:0.83rem;font-weight:600;color:{text_color};">{feat['name']}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
                                font-weight:700;color:{text_color};">{feat['pct']}%</span>
                </div>
                <div style="height:6px;background:rgba(255,255,255,0.06);
                            border-radius:9999px;overflow:hidden;">
                    <div style="height:100%;width:{bar_width}%;background:{bar_color};
                                border-radius:9999px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


    # ── Dashboard Footer ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="dashboard-footer">
        <div>Rainfall AI &nbsp;&bull;&nbsp; Machine Learning Weather Prediction</div>
        <div>Python &nbsp;&bull;&nbsp; Scikit-learn &nbsp;&bull;&nbsp; Streamlit</div>
    </div>
    """, unsafe_allow_html=True)
