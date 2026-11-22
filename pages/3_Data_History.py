import io
import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from auth_db import delete_history_record, fetch_history_records, require_login

st.set_page_config(page_title="History · XPS Deconvolution", page_icon="◈", layout="wide")
current_user = require_login()
is_host = current_user.get("role") == "host"

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

        st.markdown("**Peaks**")
        peaks_list = item.get("binding_energy_peaks", [])
        if peaks_list:
            st.dataframe(pd.DataFrame(peaks_list), use_container_width=True, hide_index=True)

        st.markdown("**Deconvolved Data Preview (Top 10 Rows)**")
        spec_df = pd.DataFrame(item.get("deconvolved_data", []))
        if not spec_df.empty:
            st.dataframe(spec_df.head(10), use_container_width=True, hide_index=True)

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
