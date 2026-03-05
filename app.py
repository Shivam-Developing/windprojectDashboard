# ============================================================
# FROST Dashboard  |  app.py
# Inference logic is 1-to-1 faithful to final_inference_wind.ipynb
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import plotly.express as px
import math
from datetime import datetime

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="FROST – Wind Turbine Icing Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0d1117 0%,#161b22 100%);}
h1,h2,h3 {color:#58a6ff;}
.risk-high   {border-left:4px solid #f85149;background:#2d1519;padding:18px;border-radius:8px;}
.risk-medium {border-left:4px solid #d29922;background:#2d2008;padding:18px;border-radius:8px;}
.risk-low    {border-left:4px solid #3fb950;background:#0d1f14;padding:18px;border-radius:8px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# STEP 0 – Load model & scaler  (exact from inference nb)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_artefacts():
    """
    Mirrors:
        scaler = joblib.load("/content/correct_scaler.pkl")
        model  = tf.saved_model.load('/content/content/ice_rf_savedmodel')
    Place  correct_scaler.pkl  and  ice_rf_savedmodel/  next to app.py in the repo.
    """
    scaler = joblib.load("correct_scaler.pkl")
    model  = tf.saved_model.load("ice_rf_savedmodel")
    return scaler, model

# ── Graceful fallback so UI still renders without model files ──
MODEL_READY = False
try:
    _scaler, _model = load_artefacts()
    MODEL_READY = True
except Exception as _e:
    _load_error = str(_e)

# ═══════════════════════════════════════════════════════════
# STEP 1 – Fetch from Open-Meteo  (exact from inference nb)
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=900)
def get_open_meteo_6h_forecast(lat: float, lon: float):
    """
    Exact replica of get_open_meteo_6h_forecast() in the notebook:
      - minutely_15: wind_speed_10m, apparent_temperature, wind_direction_10m
      - wind_speed_unit: kmh, temperature_unit: celsius
      - idx = list(range(0,4)) + [4,5] + list(range(6,10))  → 10 rows
    Returns raw (N,3): wind_kmh | apparent_temp_C | wind_dir_deg
    Also returns the raw time-stamps and temperature_2m for display.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "minutely_15":      "wind_speed_10m,apparent_temperature,wind_direction_10m,temperature_2m",
        "forecast_days":    1,
        "timezone":         "auto",
        "wind_speed_unit":  "kmh",
        "temperature_unit": "celsius",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    m = j["minutely_15"]

    winds = np.array(m["wind_speed_10m"],       dtype=float)
    feels = np.array(m["apparent_temperature"],  dtype=float)
    wdirs = np.array(m["wind_direction_10m"],    dtype=float)
    temps = np.array(m["temperature_2m"],        dtype=float)
    times = m["time"]

    # Exact slot selection from notebook
    idx = list(range(0, 4)) + [4, 5] + list(range(6, 10))  # 10 rows
    raw = np.column_stack((winds[idx], feels[idx], wdirs[idx]))
    return raw, [times[i] for i in idx], temps[idx]

# ═══════════════════════════════════════════════════════════
# STEP 2 – Build 5-feature matrix  (exact from inference nb)
# ═══════════════════════════════════════════════════════════
def to_feature_matrix(raw: np.ndarray) -> np.ndarray:
    """
    Exact replica of to_feature_matrix() in the notebook.
    raw: (N,3)  →  wind_kmh | apparent_temp_C | wind_dir_deg
    Output column order (must match training CSV):
        Wind_speed_ms | Ambient_temperature_C | direc_sin | direc_cos | Wind_chill_factor
    """
    wind_kmh = raw[:, 0]
    temp_c   = raw[:, 1]
    wdir_deg = raw[:, 2]

    # 1. Wind speed m/s
    wind_ms = wind_kmh / 3.6

    # 2. Cyclical direction encoding
    rad      = np.deg2rad(wdir_deg)
    direc_sin = np.sin(rad)
    direc_cos = np.cos(rad)

    # 3. Wind-chill °C  (paper formula)
    wind_mph = wind_kmh / 1.60934
    temp_f   = temp_c * 1.8 + 32.0
    valid    = (wind_mph >= 3.0) & (temp_f <= 50.0)
    wcf      = np.empty_like(temp_f)
    wcf[valid] = (
        35.74
        + 0.6215  * temp_f[valid]
        - 35.75   * (wind_mph[valid] ** 0.16)
        + 0.4275  * temp_f[valid] * (wind_mph[valid] ** 0.16)
    )
    wcf[~valid] = temp_f[~valid]   # outside formula range → use apparent temp
    wcf = (wcf - 32) * 5 / 9       # back to °C

    # 4. Column order must match the CSV
    return np.column_stack((wind_ms, temp_c, direc_sin, direc_cos, wcf))

# ═══════════════════════════════════════════════════════════
# STEP 3 – Inference  (exact from inference nb)
# ═══════════════════════════════════════════════════════════
FEATURE_NAMES = [
    "Wind_speed_ms",
    "Ambient_temperature_C",
    "direc_sin",
    "direc_cos",
    "Wind_chill_factor",
]

def predict_ice(matrix: np.ndarray, scaler, model):
    """
    Exact replica of predict_ice() in the notebook:
      1. scaler.transform(matrix)
      2. Build per-feature tf.constant tensors
      3. model.signatures['serving_default'](**inputs)['output_1']
      4. threshold 0.5
    """
    scaled = scaler.transform(matrix).astype("float32")
    inputs = {name: tf.constant(scaled[:, i]) for i, name in enumerate(FEATURE_NAMES)}
    probs  = model.signatures["serving_default"](**inputs)["output_1"]
    probs_arr = probs.numpy().ravel()
    preds = (probs_arr > 0.5).astype(int)
    return preds, probs_arr, scaled

# ── Helpers ──────────────────────────────────────────────────
def risk_label(prob):
    if prob >= 0.65: return "🔴 HIGH",   "risk-high"
    if prob >= 0.35: return "🟡 MEDIUM", "risk-medium"
    return "🟢 LOW",    "risk-low"

LOCATIONS = {
    "Greenland Ice Cap":   (72.57,  -38.57),
    "Canada – Baffin":     (70.47,  -68.59),
    "Norway – Tromsø":     (69.65,   18.96),
    "Iceland – Reykjavik": (64.14,  -21.94),
    "Custom":              (None,    None),
}

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("❄️ FROST")
    st.caption("Forecasting Real-time Operational Safety for Turbine Icing")

    if MODEL_READY:
        st.success("✅ Model & Scaler Loaded")
    else:
        st.warning("⚠️ Model files not found.\nUpload `correct_scaler.pkl` & `ice_rf_savedmodel/` to repo root.")

    st.divider()
    loc_choice = st.selectbox("📍 Location", list(LOCATIONS.keys()))
    if loc_choice == "Custom":
        lat = st.number_input("Latitude",  value=65.0,  format="%.4f")
        lon = st.number_input("Longitude", value=-18.0, format="%.4f")
    else:
        lat, lon = LOCATIONS[loc_choice]

    threshold = st.slider("⚠️ Alert Threshold (P_ice)", 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.markdown("**Model**")
    st.markdown("- Random Forest (TF-DF SavedModel)")
    st.markdown("- Scaler: `correct_scaler.pkl` (joblib)")
    st.markdown("- Signature: `serving_default → output_1`")
    st.markdown("- F1 = **0.98** | AUC = **0.9985**")
    fetch_btn = st.button("🔄 Fetch & Predict", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("# ❄️ FROST – Wind Turbine Icing Risk Dashboard")
st.caption(
    f"**{loc_choice}** | Lat `{lat}` Lon `{lon}` | "
    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"
)
st.divider()

tab1, tab2, tab3 = st.tabs(["📡 6-Hour Live Forecast", "🔮 Single Prediction", "📊 Model & Features"])

# ═══════════════════════════════════════════════════════════
#  TAB 1 – Live 6-hour forecast (primary tab)
# ═══════════════════════════════════════════════════════════
with tab1:
    if fetch_btn or "results" in st.session_state:
        if fetch_btn:
            if not MODEL_READY:
                st.error("Model files not loaded. Add `correct_scaler.pkl` and `ice_rf_savedmodel/` to the repo.")
                st.stop()
            with st.spinner("Fetching from Open-Meteo & running inference…"):
                try:
                    raw, times, amb_temps = get_open_meteo_6h_forecast(lat, lon)
                    feat   = to_feature_matrix(raw)
                    preds, probs, scaled = predict_ice(feat, _scaler, _model)
                    st.session_state["results"] = {
                        "raw": raw, "times": times, "amb_temps": amb_temps,
                        "feat": feat, "preds": preds, "probs": probs,
                        "scaled": scaled, "lat": lat, "lon": lon,
                    }
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.stop()

        res = st.session_state["results"]
        raw, times, amb_temps = res["raw"], res["times"], res["amb_temps"]
        feat, preds, probs, scaled = res["feat"], res["preds"], res["probs"], res["scaled"]

        # ── KPI row ─────────────────────────────────────────
        ice_count = int(preds.sum())
        avg_prob  = float(probs.mean())
        lbl, css  = risk_label(avg_prob)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("⏱️ Time Slots",     len(preds))
        k2.metric("🧊 Icing Slots",    ice_count,
                  delta=f"{ice_count/len(preds)*100:.0f}% of horizon",
                  delta_color="inverse")
        k3.metric("📊 Avg P(icing)",   f"{avg_prob*100:.1f}%")
        k4.metric("🌡️ Min Amb. Temp",  f"{amb_temps.min():.1f} °C")

        st.markdown(f'<div class="{css}"><h3>{lbl} — Overall Risk</h3>'
                    f'<p>Average icing probability: <strong>{avg_prob*100:.1f}%</strong> '
                    f'across {len(preds)} slots</p></div>', unsafe_allow_html=True)
        st.write("")

        # ── Probability timeline ─────────────────────────────
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=list(range(1, len(probs)+1)), y=probs,
            fill="tozeroy", fillcolor="rgba(88,166,255,0.15)",
            line=dict(color="#58a6ff", width=2.5),
            mode="lines+markers",
            marker=dict(color=["#f85149" if p >= threshold else "#3fb950" for p in probs],
                        size=9),
            name="P(icing)",
            text=[f"Slot {i+1}<br>Time: {t}<br>P={p:.3f}<br>{'🧊 ICE' if p>=threshold else '✅ No Ice'}"
                  for i,(t,p) in enumerate(zip(times,probs))],
            hoverinfo="text",
        ))
        fig_prob.add_hline(y=threshold, line_dash="dash", line_color="#f85149",
                           annotation_text=f"Alert threshold {threshold:.2f}")
        fig_prob.update_layout(
            title="Slot-wise Icing Probability (6-Hour Horizon)",
            xaxis_title="Time Slot", yaxis_title="P(icing)",
            yaxis=dict(range=[0, 1.05]),
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font_color="#c9d1d9", height=360,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # ── Temp + Wind ──────────────────────────────────────
        slots = list(range(1, len(times)+1))
        fig_tw = go.Figure()
        fig_tw.add_trace(go.Scatter(
            x=slots, y=amb_temps, name="Ambient Temp (°C)",
            line=dict(color="#79c0ff", width=2),
        ))
        fig_tw.add_trace(go.Scatter(
            x=slots, y=feat[:, 4], name="Wind Chill (°C)",
            line=dict(color="#d2a8ff", dash="dot", width=2),
        ))
        fig_tw.add_trace(go.Bar(
            x=slots, y=feat[:, 0], name="Wind Speed (m/s)",
            marker_color="rgba(63,185,80,0.45)", yaxis="y2",
        ))
        fig_tw.update_layout(
            title="Ambient Temp / Wind Chill / Wind Speed per Slot",
            xaxis_title="Time Slot",
            yaxis=dict(title="Temperature (°C)", color="#79c0ff"),
            yaxis2=dict(title="Wind Speed (m/s)", overlaying="y",
                        side="right", color="#3fb950"),
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font_color="#c9d1d9", height=380,
            legend=dict(x=0, y=1.12, orientation="h"),
        )
        st.plotly_chart(fig_tw, use_container_width=True)

        # ── Slot-wise table ──────────────────────────────────
        st.subheader("📋 Slot-wise Prediction Table")
        tbl = pd.DataFrame({
            "Slot":          list(range(1, len(times)+1)),
            "Time":          times,
            "Wind (km/h)":   raw[:, 0].round(1),
            "App.Temp (°C)": raw[:, 1].round(1),
            "Dir (°)":       raw[:, 2].round(1),
            "Wind (m/s)":    feat[:, 0].round(2),
            "Amb.Temp (°C)": amb_temps.round(1),
            "dir_sin":       feat[:, 2].round(4),
            "dir_cos":       feat[:, 3].round(4),
            "WCF (°C)":      feat[:, 4].round(2),
            "P(icing)":      probs.round(4),
            "Prediction":    ["🧊 ICE" if p else "✅ No Ice" for p in preds],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # ── Raw API matrix (like DEBUG output in notebook) ───
        with st.expander("🔍 DEBUG – RAW Open-Meteo matrix (km/h, °C, deg)"):
            st.text("[DEBUG] RAW from Open-Meteo (km/h, °C, deg):")
            st.dataframe(
                pd.DataFrame(raw, columns=["wind_kmh","apparent_temp_C","wind_dir_deg"]),
                use_container_width=True,
            )
            st.text("[DEBUG] Scaled matrix (after scaler.transform):")
            st.dataframe(
                pd.DataFrame(scaled, columns=FEATURE_NAMES),
                use_container_width=True,
            )
            st.text("[DEBUG] Probabilities (model.signatures['serving_default']['output_1']):")
            st.write(probs)
            st.text("Final ice predictions (0=no, 1=yes):")
            st.write(preds)

        # ── Wind rose ────────────────────────────────────────
        rose_df = pd.DataFrame({
            "wind_dir": raw[:, 2],
            "wind_ms":  feat[:, 0],
            "P(icing)": probs,
        })
        fig_rose = px.bar_polar(
            rose_df, r="wind_ms", theta="wind_dir",
            color="P(icing)",
            color_continuous_scale=["#3fb950","#d29922","#f85149"],
            title="Wind Rose — colored by icing probability",
        )
        fig_rose.update_layout(
            paper_bgcolor="#161b22", font_color="#c9d1d9", height=420,
        )
        st.plotly_chart(fig_rose, use_container_width=True)

    else:
        st.info("👈 Select a location and press **Fetch & Predict** in the sidebar.")
        st.markdown("""
        **Inference pipeline (exact notebook flow):**
        1. `get_open_meteo_6h_forecast(lat, lon)` → RAW (10×3): wind_kmh, apparent_temp, wind_dir
        2. `to_feature_matrix(raw)` → 5-col matrix: Wind_speed_ms, Ambient_temperature_C, direc_sin, direc_cos, Wind_chill_factor
        3. `scaler.transform(matrix)` using `correct_scaler.pkl`
        4. `model.signatures['serving_default'](**inputs)['output_1']` → probabilities
        5. Threshold 0.5 → binary predictions
        """)

# ═══════════════════════════════════════════════════════════
#  TAB 2 – Single manual prediction
# ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Manual Single-Slot Prediction")
    st.caption("Builds the exact same 5-feature vector as `to_feature_matrix()`, then runs through the model.")

    c1, c2, c3 = st.columns(3)
    with c1:
        wind_kmh_in = st.number_input("💨 Wind Speed (km/h)", value=32.0, step=1.0, min_value=0.0)
        app_temp_in = st.number_input("🌡️ Apparent Temp (°C)", value=-12.0, step=0.5)
    with c2:
        wdir_in     = st.number_input("🧭 Wind Direction (°)", value=229.0, step=1.0,
                                      min_value=0.0, max_value=360.0)

    # Compute features (same as to_feature_matrix)
    wind_ms_s  = wind_kmh_in / 3.6
    rad_s      = math.radians(wdir_in)
    dir_sin_s  = math.sin(rad_s)
    dir_cos_s  = math.cos(rad_s)
    wind_mph_s = wind_kmh_in / 1.60934
    temp_f_s   = app_temp_in * 1.8 + 32.0
    if wind_mph_s >= 3.0 and temp_f_s <= 50.0:
        wcf_f = (35.74 + 0.6215*temp_f_s
                 - 35.75*(wind_mph_s**0.16)
                 + 0.4275*temp_f_s*(wind_mph_s**0.16))
    else:
        wcf_f = temp_f_s
    wcf_s = (wcf_f - 32) * 5/9

    mat = np.array([[wind_ms_s, app_temp_in, dir_sin_s, dir_cos_s, wcf_s]])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Wind (m/s)",   f"{wind_ms_s:.2f}")
    m2.metric("Temp (°C)",    f"{app_temp_in:.1f}")
    m3.metric("dir_sin",      f"{dir_sin_s:.4f}")
    m4.metric("dir_cos",      f"{dir_cos_s:.4f}")
    m5.metric("WCF (°C)",     f"{wcf_s:.2f}")

    if MODEL_READY:
        sp, sp_arr, sp_sc = predict_ice(mat, _scaler, _model)
        prob_s = float(sp_arr[0])
        pred_s = int(sp[0])
        lbl_s, css_s = risk_label(prob_s)

        st.markdown(f"""
        <div class="{css_s}">
          <h3>{lbl_s} — P(icing) = {prob_s*100:.1f}%</h3>
          <p>Prediction: <strong>{"🧊 ICE DETECTED" if pred_s else "✅ No Icing"}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_s * 100,
            title={"text": "Icing Probability (%)"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#58a6ff"},
                "steps": [
                    {"range": [0,  35],  "color": "#0d1f14"},
                    {"range": [35, 65],  "color": "#2d2008"},
                    {"range": [65, 100], "color": "#2d1519"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.75, "value": threshold*100},
            },
        ))
        fig_g.update_layout(paper_bgcolor="#161b22", font_color="#c9d1d9", height=300)
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.warning("Load model files to get predictions. Feature engineering shown above is correct.")

# ═══════════════════════════════════════════════════════════
#  TAB 3 – Model info
# ═══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Classification Performance")
    perf = pd.DataFrame({
        "Class":     ["No Icing (0)", "Icing (1)", "Macro Avg", "—"],
        "Precision": [1.00, 0.97, 0.98, "—"],
        "Recall":    [0.97, 1.00, 0.98, "—"],
        "F1-Score":  [0.98, 0.98, 0.98, "—"],
        "Support":   [9221, 9059, "—", "—"],
        "Extra":     ["", "", f"Accuracy 0.98", f"ROC-AUC 0.9985"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    st.subheader("Feature Importances (Permutation – INV_MEAN_MIN_DEPTH)")
    fi = pd.DataFrame({
        "Feature":    ["Rolling Avg Temp (°C)", "Rolling Avg Wind (m/s)",
                       "Ambient Temp (°C)", "direc_sin", "direc_cos",
                       "Wind Speed (m/s)", "Wind Chill Factor (°C)"],
        "Importance": [0.3835, 0.2224, 0.2208, 0.1658, 0.1470, 0.1410, 0.0990],
    }).sort_values("Importance")
    fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                    title="Top Features — Random Forest (TF-DF)")
    fig_fi.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                         font_color="#c9d1d9", height=360, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("🌍 Inference Sites (from paper)")
    sites = pd.DataFrame([
        {"Location": "Greenland Ice Cap",   "Latitude": 72.57, "Longitude": -38.57},
        {"Location": "Canada – Baffin",     "Latitude": 70.47, "Longitude": -68.59},
        {"Location": "Norway – Tromsø",     "Latitude": 69.65, "Longitude":  18.96},
        {"Location": "Iceland – Reykjavik", "Latitude": 64.14, "Longitude": -21.94},
    ])
    fig_map = px.scatter_geo(sites, lat="Latitude", lon="Longitude", text="Location",
                             projection="natural earth", title="Paper Evaluation Sites")
    fig_map.update_traces(textposition="top center", marker=dict(size=12, color="#58a6ff"))
    fig_map.update_layout(
        paper_bgcolor="#161b22", font_color="#c9d1d9",
        geo=dict(bgcolor="#0d1117", showland=True, landcolor="#21262d",
                 showocean=True, oceancolor="#0d1117"),
        height=420,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("⚙️ Feature Engineering Formulas")
    st.markdown("**Wind Chill Factor (paper Eq. 4):**")
    st.latex(r"WCF = 35.74 + 0.6215\,T_F - 35.75\,V^{0.16} + 0.4275\,T_F\,V^{0.16}")
    st.markdown("**Cyclical Wind Direction Encoding (paper Eq. 3):**")
    st.latex(r"\theta_{sin}=\sin\!\left(\frac{\theta\pi}{180}\right),\quad \theta_{cos}=\cos\!\left(\frac{\theta\pi}{180}\right)")
    st.markdown("**Random Forest Probability (paper Eq. 5):**")
    st.latex(r"P(y{=}1\mid x)=\frac{1}{K}\sum_{k=1}^{K}h_k(x)")
