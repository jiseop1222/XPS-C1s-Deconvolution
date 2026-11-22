import io
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from lmfit import Model, Parameters

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from auth_db import require_login, save_history_record

st.set_page_config(page_title="XPS C1s Deconvolution", layout="wide")
current_user = require_login()

st.title("🔬 XPS C1s Auto Deconvolution Web App")
st.caption(f"로그인 계정: {current_user['email']} · Pseudo-Voigt (lmfit) 피팅 기반 (.txt, .csv, .dat 지원)")

st.info(
    "**Information**\n"
    "- **(1) Data Format:** `Binding Energy (eV)`, `Intensity (a.u.)` 두 열로 구성된 **`.txt`, `.csv`, `.dat` 파일**을 지원합니다 "
    "(쉼표 또는 공백/탭 구분, 헤더나 주석 행은 자동 제외).\n"
    "- **(2) 원본 데이터 피팅 vs Auto Align:** `Auto Align 활성화`를 켜면 지정 범위 내 최대 피크를 기준 피크(`284.4 eV`)로 이동 보정한 뒤 피팅하고, "
    "끄면 축 이동 없이 입력 데이터 기준으로 바로 피팅합니다."
)


def sanitize_prefix(name: str) -> str:
    replacements = {
        "/": "_", "=": "_", " ": "_", "–": "_", "-": "_",
        "π": "pi", "*": "star", "(": "", ")": "", "+": "plus"
    }
    res = name
    for k, v in replacements.items():
        res = res.replace(k, v)
    return res.strip("_") + "_"


def auto_align(x: np.ndarray, y: np.ndarray, target_peak: float = 284.4, search_min: float = 283.0, search_max: float = 290.0):
    mask = (x >= search_min) & (x <= search_max)
    if np.any(mask):
        peak_pos = x[mask][np.argmax(y[mask])]
        shift = target_peak - peak_pos
        return x + shift, shift, peak_pos
    return x, 0.0, None


def remove_background(x: np.ndarray, y: np.ndarray, mode: str, bg_start: float, bg_end: float):
    if mode == "None":
        return y.copy(), np.zeros_like(y)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    lo, hi = min(bg_start, bg_end), max(bg_start, bg_end)
    idx_lo = int(np.argmin(np.abs(x_sorted - lo)))
    idx_hi = int(np.argmin(np.abs(x_sorted - hi)))
    if idx_lo > idx_hi:
        idx_lo, idx_hi = idx_hi, idx_lo
    if idx_hi == idx_lo:
        idx_hi = min(idx_lo + 1, len(x_sorted) - 1)

    bg_sorted = np.zeros_like(y_sorted)

    if mode == "Linear":
        x0, x1 = x_sorted[idx_lo], x_sorted[idx_hi]
        y0, y1 = y_sorted[idx_lo], y_sorted[idx_hi]
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        bg_sorted = y0 + slope * (x_sorted - x0)

    elif mode == "Shirley (Iterative)":
        seg_x = x_sorted[idx_lo:idx_hi + 1]
        seg_y = y_sorted[idx_lo:idx_hi + 1]
        y_left = seg_y[0]
        y_right = seg_y[-1]
        bg_seg = np.linspace(y_left, y_right, len(seg_y))

        for _ in range(30):
            rem = np.maximum(seg_y - bg_seg, 0.0)
            cum_area = np.zeros_like(rem)
            if len(seg_x) > 1:
                dx = np.diff(seg_x)
                cum_area[1:] = np.cumsum(0.5 * (rem[:-1] + rem[1:]) * dx)
            total_area = cum_area[-1]
            if total_area > 0:
                new_bg = y_right + (y_left - y_right) * (1.0 - cum_area / total_area)
            else:
                new_bg = np.linspace(y_left, y_right, len(seg_y))
            if np.max(np.abs(new_bg - bg_seg)) < 1e-6:
                bg_seg = new_bg
                break
            bg_seg = new_bg

        bg_sorted[:idx_lo] = y_left
        bg_sorted[idx_lo:idx_hi + 1] = bg_seg
        bg_sorted[idx_hi + 1:] = y_right

    bg_restored = np.empty_like(bg_sorted)
    bg_restored[order] = bg_sorted
    return y - bg_restored, bg_restored


def pseudo_voigt(x, amplitude, center, sigma, fraction_lorentzian):
    g = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    l = amplitude * (1.0 / (1.0 + ((x - center) / sigma) ** 2))
    return (1.0 - fraction_lorentzian) * g + fraction_lorentzian * l


with st.sidebar:
    st.header("1. XPS 데이터 업로드")
    uploaded_file = st.file_uploader("Binding Energy, Intensity 파일 (.txt, .csv, .dat)", type=["txt", "csv", "dat"])

    st.header("2. Peak Alignment (보정)")
    enable_align = st.checkbox("Auto Align 활성화", value=True)
    target_peak = st.number_input("기준 피크 위치 (eV)", value=284.4, step=0.1)
    search_min = st.number_input("탐색 시작 (eV)", value=283.0, step=0.1)
    search_max = st.number_input("탐색 종료 (eV)", value=290.0, step=0.1)

    st.header("3. Background 제거")
    bg_mode = st.selectbox("Background 방식", ["None", "Linear", "Shirley (Iterative)"], index=0)
    bg_start = st.number_input("BG 시작 에너지 (eV)", value=292.0, step=0.5)
    bg_end = st.number_input("BG 종료 에너지 (eV)", value=281.0, step=0.5)

    st.header("4. 피팅 공통 옵션")
    init_amp = st.number_input("(1) 초기 Amplitude", value=1000.0, min_value=1.0, step=100.0)
    st.caption(
        "※ 각주 (1) 초기 Amplitude: 피팅 시작 시 각 피크 높이의 초기값(하한 0)입니다. "
        "입력 스펙트럼 Intensity 크기에 맞춰 설정하면 수렴 안정성이 높아집니다."
    )

    init_sigma = st.number_input("(2) 초기 Sigma (eV)", value=0.6, min_value=0.05, step=0.05)
    fix_sigma = st.checkbox("Sigma 고정", value=False)
    st.caption(
        "※ 각주 (2) 초기 Sigma (eV): 피크 폭 스케일 파라미터(0.05~5.0 eV)입니다. "
        "'Sigma 고정'을 켜면 입력값으로 고정하고, 끄면 피팅 과정에서 자유롭게 최적화됩니다."
    )

    init_lorentz = st.slider("(3) 초기 Lorentzian ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    st.caption(
        "※ 각주 (3) 초기 Lorentzian ratio: Pseudo-Voigt 함수에서 Lorentzian 성분 혼합 비율"
        "(0.0=완전 Gaussian, 1.0=완전 Lorentzian)의 초기값입니다."
    )


st.subheader("결합 에너지(Binding Energy) 피크 목록 설정")
default_peaks_df = pd.DataFrame([
    {"Peak Name": "C-C sp2", "Center (eV)": 284.40, "Tolerance (±eV)": 0.30},
    {"Peak Name": "C–N/C–C sp3", "Center (eV)": 286.00, "Tolerance (±eV)": 0.30},
    {"Peak Name": "C=O", "Center (eV)": 287.30, "Tolerance (±eV)": 0.30},
    {"Peak Name": "O–C=O", "Center (eV)": 288.80, "Tolerance (±eV)": 0.30},
    {"Peak Name": "π–π*", "Center (eV)": 290.38, "Tolerance (±eV)": 0.30},
])

peaks_df = st.data_editor(
    default_peaks_df,
    num_rows="dynamic",
    use_container_width=True,
    key="peaks_editor"
)
st.caption("※ 각주: 피크 중심 이동 허용 범위(Tolerance)는 **0.2 ~ 0.3 eV** 설정을 권장합니다.")

if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = [ln.strip().replace(",", " ") for ln in raw_bytes.splitlines() if ln.strip()]
        numeric_rows = []
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 2:
                try:
                    numeric_rows.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        if len(numeric_rows) < 5:
            st.error("유효한 숫자 데이터(Binding Energy, Intensity)를 충분히 읽지 못했습니다.")
            st.stop()

        data = np.array(numeric_rows)
        be_raw = data[:, 0]
        int_raw = data[:, 1]

        if enable_align:
            be_aligned, shift_val, detected_pos = auto_align(be_raw, int_raw, target_peak, search_min, search_max)
            if detected_pos is not None:
                st.info(f"Auto Align 적용: {detected_pos:.2f} eV 피크 감지 → {shift_val:+.2f} eV 보정 완료")
            else:
                st.warning("지정 범위 내 피크를 찾지 못해 정렬을 건너뛰었습니다.")
        else:
            be_aligned = be_raw.copy()

        corrected_intensity, bg_curve = remove_background(be_aligned, int_raw, bg_mode, bg_start, bg_end)

        valid_peaks = peaks_df.dropna(subset=["Peak Name", "Center (eV)"])
        valid_peaks = valid_peaks[valid_peaks["Peak Name"].astype(str).str.strip() != ""]

        if len(valid_peaks) == 0:
            st.warning("최소 1개 이상의 피크를 입력해 주세요.")
            st.stop()

        if st.button("🚀 Deconvolution 실행", type="primary"):
            models = []
            params = Parameters()
            peak_entries = []

            for _, row in valid_peaks.iterrows():
                name = str(row["Peak Name"]).strip()
                center_val = float(row["Center (eV)"])
                tol_val = float(row["Tolerance (±eV)"]) if pd.notna(row["Tolerance (±eV)"]) else 0.3
                prefix = sanitize_prefix(name)

                model = Model(pseudo_voigt, prefix=prefix)
                models.append(model)

                params.add(f"{prefix}amplitude", value=init_amp, min=0.0)
                params.add(f"{prefix}center", value=center_val, min=center_val - tol_val, max=center_val + tol_val)
                params.add(f"{prefix}sigma", value=init_sigma, min=0.05, max=5.0, vary=not fix_sigma)
                params.add(f"{prefix}fraction_lorentzian", value=init_lorentz, min=0.0, max=1.0)
                peak_entries.append((name, prefix))

            full_model = models[0]
            for m in models[1:]:
                full_model = full_model + m

            result = full_model.fit(corrected_intensity, params, x=be_aligned)

            ss_res = np.sum((corrected_intensity - result.best_fit) ** 2)
            ss_tot = np.sum((corrected_intensity - np.mean(corrected_intensity)) ** 2)
            r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            st.divider()
            st.subheader(f"Deconvolution 결과 (R² = {r2_score:.5f})")

            amps = {name: result.params[f"{prefix}amplitude"].value for name, prefix in peak_entries}
            total_amp = sum(amps.values())

            summary_rows = []
            comp_curves = {}
            for name, prefix in peak_entries:
                amp = result.params[f"{prefix}amplitude"].value
                cen = result.params[f"{prefix}center"].value
                sig = result.params[f"{prefix}sigma"].value
                frac = result.params[f"{prefix}fraction_lorentzian"].value
                rel_pct = (amp / total_amp * 100.0) if total_amp > 0 else 0.0

                if rel_pct > 10:
                    status = "Present (>10%)"
                elif rel_pct > 3:
                    status = "Weak (3–10%)"
                else:
                    status = "Negligible (≤3%)"

                comp = pseudo_voigt(be_aligned, amp, cen, sig, frac)
                comp_curves[name] = comp

                summary_rows.append({
                    "Peak Name": name,
                    "Fitted Center (eV)": round(cen, 3),
                    "Sigma (eV)": round(sig, 3),
                    "Lorentzian Frac": round(frac, 3),
                    "Amplitude": round(amp, 2),
                    "Contribution (%)": round(rel_pct, 2),
                    "Status": status
                })

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(be_aligned, corrected_intensity, "r.", markersize=3, label="Corrected Data")
            ax.plot(be_aligned, result.best_fit, "k-", lw=2.0, label="Total Fit")

            for name, comp in comp_curves.items():
                pct = summary_df.loc[summary_df["Peak Name"] == name, "Contribution (%)"].values[0]
                ax.plot(be_aligned, comp, "--", lw=1.3, label=f"{name} ({pct:.1f}%)")

            ax.set_xlabel("Binding Energy (eV)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title("XPS C1s Deconvolution")
            ax.invert_xaxis()
            ax.legend(loc="best", fontsize=8)
            st.pyplot(fig)

            export_cols = [be_aligned, int_raw, corrected_intensity, result.best_fit] + [comp_curves[n] for n, _ in peak_entries]
            export_headers = ["BindingEnergy(eV)", "RawIntensity(a.u.)", "CorrectedIntensity(a.u.)", "TotalFit(a.u.)"] + [
                f"{sanitize_prefix(n).strip('_')}" for n, _ in peak_entries
            ]
            stacked = np.column_stack(export_cols)
            txt_buf = io.StringIO()
            np.savetxt(txt_buf, stacked, fmt="%.4f", header="  ".join(export_headers))

            csv_df = pd.DataFrame(stacked, columns=export_headers)
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8-sig")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 전체 피팅 결과 다운로드 (.txt)",
                    data=txt_buf.getvalue().encode("utf-8"),
                    file_name="xps_c1s_deconvolution_result.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="📥 전체 피팅 결과 다운로드 (.csv)",
                    data=csv_bytes,
                    file_name="xps_c1s_deconvolution_result.csv",
                    mime="text/csv"
                )

            with st.expander("lmfit 상세 피팅 리포트 보기"):
                st.text(result.fit_report(min_correl=0.5))

            peak_list = [
                {
                    "Peak Name": str(row["Peak Name"]).strip(),
                    "Center (eV)": float(row["Center (eV)"]),
                    "Tolerance (±eV)": float(row["Tolerance (±eV)"]) if pd.notna(row["Tolerance (±eV)"]) else 0.3,
                }
                for _, row in valid_peaks.iterrows()
            ]

            saved_name = save_history_record(
                owner_email=current_user["email"],
                element="C1s",
                uploaded_filename=uploaded_file.name,
                bg_mode=bg_mode,
                lorentzian_ratio=init_lorentz,
                payload={
                    "element": "C1s",
                    "uploaded_filename": uploaded_file.name,
                    "r2_score": round(float(r2_score), 6),
                    "peak_alignment": {
                        "enabled": bool(enable_align),
                        "target_peak_eV": float(target_peak),
                        "range_eV": [float(search_min), float(search_max)],
                    },
                    "background": {
                        "method": bg_mode,
                        "range_eV": [float(bg_start), float(bg_end)],
                    },
                    "fitting_options": {
                        "initial_amplitude": float(init_amp),
                        "initial_sigma_eV": float(init_sigma),
                        "fix_sigma": bool(fix_sigma),
                        "initial_lorentzian_ratio": float(init_lorentz),
                    },
                    "binding_energy_peaks": peak_list,
                    "fitted_summary": summary_df.to_dict(orient="records"),
                    "deconvolved_data": csv_df.to_dict(orient="records"),
                },
            )
            st.success(f"영구 저장 완료 ({current_user['email']}): `{saved_name}`")

    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {e}")
else:
    st.info("왼쪽 사이드바에서 XPS 데이터 파일(`.txt`, `.csv`, `.dat`)을 업로드해 주세요.")
