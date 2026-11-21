import streamlit as st

st.set_page_config(
    page_title="XPS Deconvolution Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 XPS Deconvolution Studio")
st.caption("Pseudo-Voigt (`lmfit`) 기반 XPS 피크 분리 웹 앱")

st.markdown("원하는 분석 페이지를 선택해 주세요.")

col1, col2 = st.columns(2)

with col1:
    st.info("### 1. C1s Deconvolution\n탄소(C1s) 스펙트럼 피팅 및 결과 다운로드")
    st.page_link("pages/1_C1s_Deconvolution.py", label="👉 1 C1s Deconvolution 열기", use_container_width=True)

with col2:
    st.info("### 2. N1s Deconvolution\n질소(N1s) 스펙트럼 피팅 및 결과 다운로드")
    st.page_link("pages/2_N1s_Deconvolution.py", label="👉 2 N1s Deconvolution 열기", use_container_width=True)
