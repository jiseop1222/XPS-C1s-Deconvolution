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

st.set_page_config(page_title="C1s · XPS Deconvolution", page_icon="◈", layout="wide")
current_user = require_login()

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

st.caption("HOME / C1s DECONVOLUTION")
st.title("◈ XPS C1s Deconvolution")
st.caption(f"Signed in as `{current_user['email']}` · Pseudo-Voigt (`lmfit`)")

st.info(
    "**Information**\n"
    "- **(1) Data Format:** Upload two-column (`Binding Energy (eV)`, `Intensity (a.u.)`) **`.txt`, `.csv`, or `.dat`** files "
    "(comma, space, or tab separated; non-numeric header/comment lines are ignored).\n"
    "- **(2) Raw Data Fitting vs. Auto Align:** Enabling **Auto Align** shifts the spectrum so the maximum peak within the search "
    "window aligns to the reference position (`284.4 eV`) before fitting; disabling it fits the input spectrum without energy shift."
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
    st.header("1. Spectrum Upload")
    uploaded_file = st.file_uploader("Binding Energy, Intensity (.txt, .csv, .dat)", type=["txt", "csv", "dat"])

    st.header("2. Peak Alignment")
    enable_align = st.checkbox("Enable Auto Align", value=True)
    target_peak = st.number_input("Reference Peak (eV)", value=284.4, step=0.1)
    search_min = st.number_input("Search Start (eV)", value=283.0, step=0.1)
    search_max = st.number_input("Search End (eV)", value=290.0, step=0.1)

    st.header("3. Background")
    bg_mode = st.selectbox("Background Method", ["None", "Linear", "Shirley (Iterative)"], index=0)
    bg_start = st.number_input("BG Start (eV)", value=292.0, step=0.5)
    bg_end = st.number_input("BG End (eV)", value=281.0, step=0.5)

    st.header("4. Fitting Options")
    init_amp = st.number_input("(1) Initial Amplitude", value=1000.0, min_value=1.0, step=100.0)
    st.caption(
        "※ Footnote (1) Initial Amplitude: Starting peak height (lower bound 0) for non-linear least-squares fitting; "
        "matching your spectrum intensity scale improves convergence."
    )

    init_sigma = st.number_input("(2) Initial Sigma (eV)", value=0.6, min_value=0.05, step=0.05)
    fix_sigma = st.checkbox("Fix Sigma", value=False)
    st.caption(
        "※ Footnote (2) Initial Sigma (eV): Peak width scale parameter (0.05–5.0 eV). "
        "Check 'Fix Sigma' to keep it constant or uncheck to optimize during fitting."
    )

    init_lorentz = st.slider("(3) Initial Lorentzian Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    st.caption(
        "※ Footnote (3) Initial Lorentzian Ratio: Initial Lorentzian fraction in Pseudo-Voigt "
        "(0.0 = pure Gaussian, 1.0 = pure Lorentzian)."
    )


st.subheader("Binding Energy Peaks")
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
st.caption("※ Footnote: Recommended peak center **Tolerance** is **0.2 – 0.3 eV**.")

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
            st.error("Insufficient numeric data rows (Binding Energy, Intensity).")
            st.stop()

        data = np.array(numeric_rows)
        be_raw = data[:, 0]
        int_raw = data[:, 1]

        if enable_align:
            be_aligned, shift_val, detected_pos = auto_align(be_raw, int_raw, target_peak, search_min, search_max)
            if detected_pos is not None:
                st.caption(f"Auto Align: detected `{detected_pos:.2f} eV` → shifted `{shift_val:+.2f} eV`")
            else:
                st.warning("No peak found in the specified search range; skipped alignment.")
        else:
            be_aligned = be_raw.copy()

        corrected_intensity, bg_curve = remove_background(be_aligned, int_raw, bg_mode, bg_start, bg_end)

        valid_peaks = peaks_df.dropna(subset=["Peak Name", "Center (eV)"])
        valid_peaks = valid_peaks[valid_peaks["Peak Name"].astype(str).str.strip() != ""]

        if len(valid_peaks) == 0:
            st.warning("Please specify at least one peak.")
            st.stop()

        if st.button("Run Deconvolution", type="primary"):
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
            st.subheader(f"Fit Result · R² = {r2_score:.5f}")

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

            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.plot(be_aligned, corrected_intensity, "o", color="#d9534f", markersize=2.5, alpha=0.85, label="Corrected Data")
            ax.plot(be_aligned, result.best_fit, "-", color="#111111", lw=1.8, label="Total Fit")

            for name, comp in comp_curves.items():
                pct = summary_df.loc[summary_df["Peak Name"] == name, "Contribution (%)"].values[0]
                ax.plot(be_aligned, comp, "--", lw=1.2, label=f"{name} ({pct:.1f}%)")

            ax.set_xlabel("Binding Energy (eV)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title("XPS C1s Deconvolution")
            ax.invert_xaxis()
            ax.legend(loc="best", frameon=False, fontsize=8)
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
                    label="Export (.txt)",
                    data=txt_buf.getvalue().encode("utf-8"),
                    file_name="xps_c1s_deconvolution_result.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="Export (.csv)",
                    data=csv_bytes,
                    file_name="xps_c1s_deconvolution_result.csv",
                    mime="text/csv"
                )

            with st.expander("lmfit Fit Report"):
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
            st.success(f"Saved to history: `{saved_name}`")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a C1s spectrum file (`.txt`, `.csv`, `.dat`) in the sidebar.")
