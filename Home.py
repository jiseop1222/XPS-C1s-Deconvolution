from pathlib import Path
import streamlit as st
from auth_db import authenticate_user, get_host_email, init_db, register_user

st.set_page_config(
    page_title="Home · XPS Deconvolution",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2.5rem;}
    .minimal-card {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .minimal-tag {
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        opacity: 0.65;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

st.markdown('<div class="minimal-tag">HOME · SPECTROSCOPY WORKSPACE</div>', unsafe_allow_html=True)
st.title("◈ XPS Deconvolution")
st.caption("Pseudo-Voigt (`lmfit`) peak fitting and persistent archive.")

user = st.session_state.get("auth_user")

if not user:
    tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            if submitted:
                ok, user_info, msg = authenticate_user(email, password)
                if ok and user_info:
                    st.session_state["auth_user"] = user_info
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.caption(f"Host email configured via `HOST_EMAIL` (`{get_host_email()}`).")

    with tab_signup:
        with st.form("signup_form"):
            new_email = st.text_input("Email")
            new_pw = st.text_input("Password (min. 6 chars)", type="password")
            new_pw2 = st.text_input("Confirm Password", type="password")
            reg_submitted = st.form_submit_button("Create Account", use_container_width=True)
            if reg_submitted:
                if new_pw != new_pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(new_email, new_pw)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
    st.stop()

top_left, top_right = st.columns([5, 1])
with top_left:
    role_label = "HOST · All Records" if user.get("role") == "host" else "USER · Own Records"
    st.caption(f"Account: `{user['email']}`  ·  Role: `{role_label}`")
with top_right:
    if st.button("Sign Out", use_container_width=True):
        st.session_state.pop("auth_user", None)
        st.rerun()

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="minimal-card"><div class="minimal-tag">01 · CARBON</div>'
        '<strong>C1s Deconvolution</strong><br><span style="opacity:0.75;font-size:0.9rem;">'
        'Peak alignment, background removal, and auto-save.</span></div>',
        unsafe_allow_html=True,
    )
    if Path("pages/1_C1s_Deconvolution.py").exists():
        st.page_link("pages/1_C1s_Deconvolution.py", label="Open C1s →", use_container_width=True)

with col2:
    st.markdown(
        '<div class="minimal-card"><div class="minimal-tag">02 · NITROGEN</div>'
        '<strong>N1s Deconvolution</strong><br><span style="opacity:0.75;font-size:0.9rem;">'
        'Peak alignment, background removal, and auto-save.</span></div>',
        unsafe_allow_html=True,
    )
    if Path("pages/2_N1s_Deconvolution.py").exists():
        st.page_link("pages/2_N1s_Deconvolution.py", label="Open N1s →", use_container_width=True)

with col3:
    st.markdown(
        '<div class="minimal-card"><div class="minimal-tag">03 · ARCHIVE</div>'
        '<strong>Data History</strong><br><span style="opacity:0.75;font-size:0.9rem;">'
        'Inspect saved runs and export JSON, CSV, or TXT.</span></div>',
        unsafe_allow_html=True,
    )
    if Path("pages/3_Data_History.py").exists():
        st.page_link("pages/3_Data_History.py", label="Open History →", use_container_width=True)

st.markdown("---")
col4, _ = st.columns([1.2, 2])
with col4:
    st.markdown(
        '<div class="minimal-card"><div class="minimal-tag">04 · PREDICTION</div>'
        '<strong>Conductivity &amp; Temp Predictor</strong><br><span style="opacity:0.75;font-size:0.9rem;">'
        'Predict conductivity and required carbonization temperature from XPS ratios.</span></div>',
        unsafe_allow_html=True,
    )
    if Path("pages/4_Conductivity_Temp_Predictor.py").exists():
        st.page_link(
            "pages/4_Conductivity_Temp_Predictor.py",
            label="Open Predictor →",
            use_container_width=True,
        )
