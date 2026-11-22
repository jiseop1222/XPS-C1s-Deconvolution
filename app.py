from pathlib import Path
import streamlit as st
from auth_db import authenticate_user, get_host_email, init_db, register_user

st.set_page_config(
    page_title="XPS Deconvolution Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

st.title("🔬 XPS Deconvolution Studio")
st.caption("Pseudo-Voigt (`lmfit`) 기반 XPS 피크 분리 및 영구 이력 관리 웹 앱")

user = st.session_state.get("auth_user")

if not user:
    tab_login, tab_signup = st.tabs(["🔑 로그인", "📝 회원가입"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("이메일")
            password = st.text_input("비밀번호", type="password")
            submitted = st.form_submit_button("로그인", type="primary", use_container_width=True)
            if submitted:
                ok, user_info, msg = authenticate_user(email, password)
                if ok and user_info:
                    st.session_state["auth_user"] = user_info
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

       
    with tab_signup:
        with st.form("signup_form"):
            new_email = st.text_input("가입할 이메일")
            new_pw = st.text_input("비밀번호 (6자 이상)", type="password")
            new_pw2 = st.text_input("비밀번호 확인", type="password")
            reg_submitted = st.form_submit_button("회원가입", use_container_width=True)
            if reg_submitted:
                if new_pw != new_pw2:
                    st.error("비밀번호 확인이 일치하지 않습니다.")
                else:
                    ok, msg = register_user(new_email, new_pw)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
    st.stop()

# 로그인 완료 상태
top_left, top_right = st.columns([4, 1])
with top_left:
    role_badge = "👑 HOST (전체 기록 열람 가능)" if user.get("role") == "host" else "👤 USER (내 기록 열람)"
    st.success(f"로그인 계정: **{user['email']}**  |  권한: **{role_badge}**")
with top_right:
    if st.button("로그아웃", use_container_width=True):
        st.session_state.pop("auth_user", None)
        st.rerun()

st.markdown("원하는 분석 페이지 또는 History 페이지를 선택해 주세요.")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 1. C1s Deconvolution\n탄소(C1s) 스펙트럼 피팅 및 자동 저장")
    if Path("pages/1_C1s_Deconvolution.py").exists():
        st.page_link("pages/1_C1s_Deconvolution.py", label="👉 1 C1s Deconvolution 열기", use_container_width=True)

with col2:
    st.info("### 2. N1s Deconvolution\n질소(N1s) 스펙트럼 피팅 및 자동 저장")
    if Path("pages/2_N1s_Deconvolution.py").exists():
        st.page_link("pages/2_N1s_Deconvolution.py", label="👉 2 N1s Deconvolution 열기", use_container_width=True)

with col3:
    st.info("### 3. Data History\n영구 저장 기록 조회 및 다운로드")
    if Path("pages/3_Data_History.py").exists():
        st.page_link("pages/3_Data_History.py", label="👉 3 Data History 열기", use_container_width=True)
