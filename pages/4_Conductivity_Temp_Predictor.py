from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from auth_db import fetch_history_records, require_login
from xps_ml_predictor import (
    DEFAULT_TRAINING_CSV,
    FEATURE_COLS,
    extract_xps_ratios_from_summary,
    fit_conductivity_and_temp_models,
    load_training_dataset,
    predict_required_temp_from_target_sigma,
    predict_sigma_temp_curve,
)

st.set_page_config(
    page_title="Conductivity & Temp Predictor · XPS",
    page_icon="◈",
    layout="wide",
)
current_user = require_login()

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

st.caption("HOME / CONDUCTIVITY & CARBONIZATION TEMP PREDICTOR")
st.title("◈ Conductivity & Carbonization Temp Predictor")
st.caption(
    f"Signed in as `{current_user['email']}` · "
    "Random Forest prediction linked with XPS C1s / N1s deconvolution ratios"
)

st.info(
    "**Information**\n"
    "- **XPS Deconvolution Link:** Select saved C1s / N1s history records to map peak `Contribution (%)` "
    "(converted to `0.0–1.0` fractions) onto `C=C`, `C-C/C-N`, `C=O`, `O-C=O`, `pi-pi*`, and N1s species.\n"
    "- **Calibration & Inverse Temp:** Evaluates predicted electrical conductivity (`S/m`) across carbonization "
    "temperatures (`5 °C/min, 1 h hold` reference) and estimates the minimum temperature for a target conductivity."
)

# 기본 피처 값(Colab 테스트 벡터 기반)
DEFAULT_FEATURE_VALUES = {
    "C=C": 0.80,
    "C-C/C-N": 0.10,
    "C=O": 0.03,
    "O-C=O": 0.00,
    "pi-pi*": 0.01,
    "Pyridinic N": 0.01,
    "NH2": 0.00,
    "Pyrrolic N": 0.02,
    "graphitic N": 0.04,
    "Oxidized N": 0.03,
    "ID_IG": 1.50,
    "C_at": 0.95,
    "N_at": 0.01,
    "O_at": 0.03,
    "Cu_at": 0.02,
    "Crystallinity": 0.50,
}

with st.sidebar:
    st.header("1. Training Dataset")
    uploaded_train_csv = st.file_uploader(
        "Upload Training CSV (carbon_data_Temp.csv)",
        type=["csv"],
        key="ml_train_csv",
    )
    if uploaded_train_csv is None and not DEFAULT_TRAINING_CSV.exists():
        st.caption(f"Expected default file: `{DEFAULT_TRAINING_CSV}`")

    n_estimators = st.slider("Random Forest Trees", min_value=50, max_value=1000, value=300, step=50)

    st.header("2. Temperature & Target σ")
    base_temp = st.number_input("Base Carbonization Temp (°C)", value=1000.0, min_value=200.0, max_value=2500.0, step=25.0)
    sigma_req = st.number_input("Target Conductivity σ_req (S/m)", value=38.0, min_value=0.0001, step=5.0, format="%.4f")

    st.subheader("σ–T Sweep Range")
    t_min = st.number_input("Sweep Start Temp (°C)", value=800, min_value=200, max_value=2000, step=25)
    t_max = st.number_input("Sweep End Temp (°C)", value=1600, min_value=400, max_value=2500, step=25)
    t_step = st.number_input("Sweep Step (°C)", value=25, min_value=5, max_value=100, step=5)


# 1) 저장된 XPS Deconvolution History에서 피크 비율 불러오기
history_records = fetch_history_records(current_user)
record_options = {"Manual / Default Values": None}
for idx, rec in enumerate(history_records):
    label = f"[{rec.get('element', '?')}] {rec.get('record_name', f'record_{idx}')} (R²={rec.get('r2_score', '-')})"
    record_options[label] = rec

st.subheader("1. Link XPS Deconvolution Results")
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    selected_c1s_label = st.selectbox(
        "C1s Deconvolution Record",
        options=list(record_options.keys()),
        index=0,
    )
with col_sel2:
    selected_n1s_label = st.selectbox(
        "N1s Deconvolution Record",
        options=list(record_options.keys()),
        index=0,
    )

mapped_ratios: dict[str, float] = {}
for lbl in (selected_c1s_label, selected_n1s_label):
    rec = record_options.get(lbl)
    if rec:
        mapped_ratios.update(extract_xps_ratios_from_summary(rec.get("fitted_summary")))

if mapped_ratios:
    st.caption("Mapped peak ratios from selected XPS deconvolution history: " + ", ".join(f"`{k}: {v:.3f}`" for k, v in mapped_ratios.items()))
else:
    st.caption("No XPS history record linked yet; using default/manual feature values below.")


# 2) 16개 입력 피처 편집 테이블
st.subheader("2. Input Features (16 Parameters)")
initial_table_rows = []
for feat in FEATURE_COLS:
    val = mapped_ratios.get(feat, DEFAULT_FEATURE_VALUES.get(feat, 0.0))
    initial_table_rows.append({"Feature": feat, "Value": float(val)})

feature_editor_df = st.data_editor(
    pd.DataFrame(initial_table_rows),
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    key="ml_feature_editor",
)
st.caption(
    "※ Footnote: XPS C1s/N1s peak contributions are mapped as fractions (`0.0–1.0`). "
    "Structural/compositional values (`ID_IG`, `C_at`, `N_at`, `O_at`, `Cu_at`, `Crystallinity`) can be adjusted directly."
)

# 3) 모델 학습 및 예측 실행
if st.button("Run Conductivity & Temperature Prediction", type="primary"):
    try:
        train_df, _ = load_training_dataset(uploaded_train_csv)
        model_bundle = fit_conductivity_and_temp_models(
            train_df,
            n_estimators=int(n_estimators),
            random_state=42,
        )

        if model_bundle["n_samples"] < 15:
            st.warning(
                f"Training dataset currently has **{model_bundle['n_samples']}** clean rows. "
                "With small sample sizes, Random Forest predictions may overfit training points; "
                "add more experimental rows to `carbon_data_Temp.csv` for broader generalization."
            )

        feat_map = dict(DEFAULT_FEATURE_VALUES)
        for _, r in feature_editor_df.iterrows():
            fname = str(r["Feature"]).strip()
            if fname in feat_map:
                try:
                    feat_map[fname] = float(r["Value"])
                except (TypeError, ValueError):
                    pass

        x_16 = np.array([feat_map[col] for col in FEATURE_COLS], dtype=np.float32)

        # 기준 온도에서의 전기전도도 예측
        x_at_base_t = np.hstack((x_16.reshape(1, -1), np.array([[float(base_temp)]], dtype=np.float32)))
        pred_sigma_base = float(model_bundle["rf_cond"].predict(x_at_base_t)[0])

        # σ–T Calibration Curve & 최소 탄화온도(T_min)
        sweep_res = predict_sigma_temp_curve(
            model_bundle["rf_cond"],
            x_16,
            t_min=int(t_min),
            t_max=int(t_max),
            t_step=int(t_step),
            sigma_req=float(sigma_req),
        )

        # 목표 전기전도도 기반 탄화온도 회귀 예측
        pred_temp_inverse = predict_required_temp_from_target_sigma(
            model_bundle["rf_temp"],
            x_16,
            target_sigma=float(sigma_req),
        )

        st.divider()
        st.subheader("3. Prediction Summary")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label=f"Predicted Conductivity @ {base_temp:.0f} °C",
                value=f"{pred_sigma_base:.4f} S/m",
            )
        with m2:
            tmin_str = f"{sweep_res['T_min_required']:.0f} °C" if sweep_res["T_min_required"] is not None else "Not reached"
            st.metric(
                label=f"Min Temp for σ ≥ {sigma_req:g} S/m (Sweep)",
                value=tmin_str,
            )
        with m3:
            st.metric(
                label=f"Estimated Temp for σ = {sigma_req:g} S/m (RF)",
                value=f"{pred_temp_inverse:.1f} °C",
            )

        st.caption(
            f"Training R² (in-sample, n={model_bundle['n_samples']}): "
            f"Conductivity = `{model_bundle['r2_cond']:.4f}` · Carbonization Temp = `{model_bundle['r2_temp']:.4f}`"
        )

        # 4) σ–T Calibration Curve & Feature Importance 시각화
        c_left, c_right = st.columns([1.25, 1.0])

        with c_left:
            st.markdown("**σ–T Calibration Curve**")
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.plot(sweep_res["T_values"], sweep_res["sigma_preds"], "o-", color="#1f4e78", lw=1.8, markersize=4, label="Predicted σ")
            ax.axhline(float(sigma_req), color="#d9534f", linestyle="--", lw=1.3, label=f"Target σ = {sigma_req:g} S/m")
            if sweep_res["T_min_required"] is not None:
                ax.axvline(sweep_res["T_min_required"], color="#2e7d32", linestyle=":", lw=1.4, label=f"T_min = {sweep_res['T_min_required']:.0f} °C")
            ax.set_yscale("log")
            ax.set_xlabel("Carbonization Temperature (°C)")
            ax.set_ylabel("Predicted Conductivity (S/m)")
            ax.set_title("Conductivity vs. Carbonization Temperature")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", frameon=False, fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        with c_right:
            st.markdown("**Feature Importance (Conductivity Model)**")
            imp_df = model_bundle["importance_df"]
            top_imp = imp_df.head(10)
            fig_imp, ax_imp = plt.subplots(figsize=(6.5, 4.5))
            y_pos = np.arange(len(top_imp))
            ax_imp.barh(y_pos, top_imp["Importance (Conductivity Model)"], color="#4a6fa5", height=0.6)
            ax_imp.set_yticks(y_pos)
            ax_imp.set_yticklabels(top_imp["Feature"], fontsize=8.5)
            ax_imp.invert_yaxis()
            ax_imp.set_xlabel("Importance", fontsize=8.5)
            ax_imp.set_title("Top 10 Important Features")
            st.pyplot(fig_imp)
            plt.close(fig_imp)

        with st.expander("Full Feature Importance Table"):
            st.dataframe(model_bundle["importance_df"], use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
