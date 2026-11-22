import io
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from auth_db import delete_history_record, fetch_history_records, require_login

st.set_page_config(page_title="History · XPS Deconvolution", page_icon="◈", layout="wide")
current_user = require_login()
is_host = current_user.get("role") == "host"

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

st.caption("HOME / DATA HISTORY")
st.title("◈ Data History")
if is_host:
    st.caption(f"Host view (`{current_user['email']}`) · All user records")
else:
    st.caption(f"User view (`{current_user['email']}`) · Own records only")

records = fetch_history_records(current_user)

if not records:
    st.info("No history records available.")
    st.stop()

available_elements = sorted({r.get("element", "Unknown") for r in records})
available_owners = sorted({r.get("owner_email", "") for r in records if r.get("owner_email")})

f1, f2 = st.columns(2)
with f1:
    selected_elements = st.multiselect(
        "Element Filter",
        options=available_elements,
        default=available_elements,
    )
with f2:
    if is_host:
        selected_owners = st.multiselect(
            "Owner Filter (Host Only)",
            options=available_owners,
            default=available_owners,
        )
    else:
        selected_owners = [current_user["email"]]

filtered = [
    r for r in records
    if r.get("element", "Unknown") in selected_elements
    and r.get("owner_email") in selected_owners
]


def _sanitize_peak_name(name: str) -> str:
    replacements = {
        "/": "_", "=": "_", " ": "_", "–": "_", "-": "_",
        "π": "pi", "*": "star", "(": "", ")": "", "+": "plus",
    }
    res = str(name)
    for k, v in replacements.items():
        res = res.replace(k, v)
    return res.strip("_")


def _render_record_visuals(item: dict, element: str) -> None:
    spec_df = pd.DataFrame(item.get("deconvolved_data", []))
    fitted_summary = item.get("fitted_summary", [])
    peaks_list = item.get("binding_energy_peaks", [])

    ratio_df = pd.DataFrame()
    if fitted_summary:
        ratio_df = pd.DataFrame(fitted_summary)
    elif peaks_list:
        ratio_df = pd.DataFrame(peaks_list)

    col_plot, col_ratio = st.columns([1.45, 1.0])

    with col_plot:
        st.markdown("**Deconvolution Plot**")
        if not spec_df.empty and "BindingEnergy(eV)" in spec_df.columns:
            be = pd.to_numeric(spec_df["BindingEnergy(eV)"], errors="coerce").to_numpy()
            fig, ax = plt.subplots(figsize=(8.5, 4.5))

            if "CorrectedIntensity(a.u.)" in spec_df.columns:
                corr = pd.to_numeric(spec_df["CorrectedIntensity(a.u.)"], errors="coerce").to_numpy()
                ax.plot(be, corr, "o", color="#d9534f", markersize=2.2, alpha=0.85, label="Corrected Data")
            elif "RawIntensity(a.u.)" in spec_df.columns:
                raw = pd.to_numeric(spec_df["RawIntensity(a.u.)"], errors="coerce").to_numpy()
                ax.plot(be, raw, "o", color="#d9534f", markersize=2.2, alpha=0.85, label="Raw Data")

            if "TotalFit(a.u.)" in spec_df.columns:
                tot_fit = pd.to_numeric(spec_df["TotalFit(a.u.)"], errors="coerce").to_numpy()
                ax.plot(be, tot_fit, "-", color="#111111", lw=1.6, label="Total Fit")

            pct_map = {}
            if not ratio_df.empty and "Peak Name" in ratio_df.columns and "Contribution (%)" in ratio_df.columns:
                for _, r in ratio_df.iterrows():
                    try:
                        pct_map[str(r["Peak Name"])] = float(r["Contribution (%)"])
                    except (TypeError, ValueError):
                        continue

            reserved_cols = {
                "BindingEnergy(eV)",
                "RawIntensity(a.u.)",
                "CorrectedIntensity(a.u.)",
                "TotalFit(a.u.)",
            }
            for col in spec_df.columns:
                if col not in reserved_cols:
                    comp_y = pd.to_numeric(spec_df[col], errors="coerce").to_numpy()
                    matched_label = col
                    for p_name, p_pct in pct_map.items():
                        if _sanitize_peak_name(p_name) == col:
                            matched_label = f"{p_name} ({p_pct:.1f}%)"
                            break
                    ax.plot(be, comp_y, "--", lw=1.15, label=matched_label)

            ax.set_xlabel("Binding Energy (eV)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"XPS {element} Deconvolution")
            ax.invert_xaxis()
            ax.legend(loc="best", frameon=False, fontsize=7.5)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("No curve data available for plotting.")

    with col_ratio:
        st.markdown("**Binding Energy Peak Ratios**")
        if not ratio_df.empty:
            display_cols = [
                c for c in ["Peak Name", "Fitted Center (eV)", "Center (eV)", "Contribution (%)", "Status"]
                if c in ratio_df.columns
            ]
            st.dataframe(
                ratio_df[display_cols] if display_cols else ratio_df,
                use_container_width=True,
                hide_index=True,
            )

            if "Peak Name" in ratio_df.columns and "Contribution (%)" in ratio_df.columns:
                bar_df = ratio_df[["Peak Name", "Contribution (%)"]].copy()
                bar_df["Contribution (%)"] = pd.to_numeric(bar_df["Contribution (%)"], errors="coerce").fillna(0.0)
                if bar_df["Contribution (%)"].sum() > 0:
                    fig_bar, ax_bar = plt.subplots(figsize=(6.5, 2.8))
                    y_pos = np.arange(len(bar_df))
                    bars = ax_bar.barh(y_pos, bar_df["Contribution (%)"], color="#4a6fa5", height=0.55)
                    ax_bar.set_yticks(y_pos)
                    ax_bar.set_yticklabels(bar_df["Peak Name"], fontsize=8)
                    ax_bar.invert_yaxis()
                    ax_bar.set_xlabel("Contribution (%)", fontsize=8)
                    ax_bar.set_xlim(0, max(100.0, float(bar_df["Contribution (%)"].max()) * 1.18))
                    for bar, pct in zip(bars, bar_df["Contribution (%)"]):
                        ax_bar.text(
                            bar.get_width() + 1.0,
                            bar.get_y() + bar.get_height() / 2,
                            f"{pct:.1f}%",
                            va="center",
                            fontsize=7.5,
                        )
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)
        else:
            st.caption("No peak ratio data available.")


for idx, item in enumerate(filtered):
    base_stem = item.get("record_name", f"record_{idx}")
    element = item.get("element", "")
    owner = item.get("owner_email", "")
    r2_val = item.get("r2_score")
    saved_at = item.get("saved_at", "")

    header_parts = [base_stem, element]
    if is_host and owner:
        header_parts.append(owner)
    if isinstance(r2_val, (int, float)):
        header_parts.append(f"R²={r2_val:.4f}")
    if saved_at:
        header_parts.append(saved_at)

    with st.expander("  ·  ".join(header_parts), expanded=False):
        st.write(f"**Source File:** `{item.get('uploaded_filename', '')}`")
        if is_host:
            st.write(f"**Owner:** `{owner}`")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Peak Alignment**")
            align_info = item.get("peak_alignment", {})
            st.write(f"- Enabled: `{align_info.get('enabled')}`")
            st.write(f"- Reference Peak: `{align_info.get('target_peak_eV')} eV`")
            rng = align_info.get("range_eV", [None, None])
            st.write(f"- Search Range: `{rng[0]} – {rng[1]} eV`")
        with c2:
            st.markdown("**Background**")
            bg_info = item.get("background", {})
            st.write(f"- Method: `{bg_info.get('method')}`")
            bg_rng = bg_info.get("range_eV", [None, None])
            st.write(f"- Range: `{bg_rng[0]} – {bg_rng[1]} eV`")
        with c3:
            st.markdown("**Fitting Options**")
            fit_info = item.get("fitting_options", {})
            st.write(f"- Initial Amplitude: `{fit_info.get('initial_amplitude')}`")
            st.write(f"- Initial Sigma: `{fit_info.get('initial_sigma_eV')} eV` (Fixed: `{fit_info.get('fix_sigma')}`)")
            st.write(f"- Initial Lorentzian Ratio: `{fit_info.get('initial_lorentzian_ratio')}`")

        st.divider()
        _render_record_visuals(item, element)

        spec_df = pd.DataFrame(item.get("deconvolved_data", []))
        peaks_list = item.get("binding_energy_peaks", [])
        export_obj = {k: v for k, v in item.items() if not k.startswith("_")}
        json_bytes = json.dumps(export_obj, ensure_ascii=False, indent=2).encode("utf-8")
        csv_bytes = spec_df.to_csv(index=False).encode("utf-8-sig")

        txt_buf = io.StringIO()
        txt_buf.write(f"Record Name: {base_stem}\n")
        txt_buf.write(f"Owner: {owner}\n")
        txt_buf.write(f"Element: {element}\n")
        txt_buf.write(f"Uploaded File: {item.get('uploaded_filename', '')}\n")
        txt_buf.write(f"Peak Alignment: {item.get('peak_alignment')}\n")
        txt_buf.write(f"Background: {item.get('background')}\n")
        txt_buf.write(f"Fitting Options: {item.get('fitting_options')}\n")
        txt_buf.write(f"Binding Energy Peaks: {json.dumps(peaks_list, ensure_ascii=False)}\n\n")
        if not spec_df.empty:
            spec_df.to_csv(txt_buf, sep="\t", index=False)
        txt_bytes = txt_buf.getvalue().encode("utf-8")

        d1, d2, d3, d4 = st.columns([1, 1, 1, 0.8])
        with d1:
            st.download_button(
                label=".json",
                data=json_bytes,
                file_name=f"{base_stem}.json",
                mime="application/json",
                key=f"dl_json_{idx}",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                label=".csv",
                data=csv_bytes,
                file_name=f"{base_stem}.csv",
                mime="text/csv",
                key=f"dl_csv_{idx}",
                use_container_width=True,
            )
        with d3:
            st.download_button(
                label=".txt",
                data=txt_bytes,
                file_name=f"{base_stem}.txt",
                mime="text/plain",
                key=f"dl_txt_{idx}",
                use_container_width=True,
            )
        with d4:
            if st.button("Delete", key=f"del_{idx}", use_container_width=True):
                delete_history_record(item["_db_id"], current_user)
                st.rerun()
