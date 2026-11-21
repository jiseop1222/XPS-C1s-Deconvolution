import streamlit as st

st.set_page_config(
    page_title="XPS Deconvolution Studio",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 XPS Deconvolution Studio")
st.caption("Pseudo-Voigt (`lmfit`) 기반 XPS 코어 레벨 피크 분리 및 분석 웹 앱")

st.markdown(
    """
    왼쪽 사이드바 메뉴에서 분석하려는 원소 페이지를 선택해 주세요.

    * **1 C1s Deconvolution:** 탄소(C1s) 결합 상태 피크 분리, Auto Align, Background 제거(`None`, `Linear`, `Shirley`), $R^2$ 및 기여도(%) 산출, `.txt`/`.csv` 결과 다운로드
    * **2 N1s Deconvolution:** 질소(N1s) 결합 상태(`Pyridinic N`, `NH2`, `Pyrrolic N`, `Graphitic N`, `Oxidized N` 등) 피크 분리 및 결과 다운로드[cite: 4]
    """
)

col1, col2 = st.columns(2)

with col1:
    st.info("### 📌 C1s Deconvolution\n사이드바에서 **1 C1s Deconvolution**을 선택해 C1s 스펙트럼 데이터를 업로드하고 피팅을 진행하세요.")

with col2:
    st.info("### 📌 N1s Deconvolution\n사이드바에서 **2 N1s Deconvolution**을 선택해 N1s 스펙트럼 데이터를 업로드하고 피팅을 진행하세요[cite: 4].")

st.divider()

st.subheader("기본 사용 순서")
st.markdown(
    """
    1. **데이터 업로드:** `Binding Energy`, `Intensity`가 포함된 `.txt`, `.csv`, `.dat` 파일을 업로드합니다.
    2. **전처리 설정:** 필요에 따라 **Auto Align**과 **Background 제거** 방식을 선택합니다.
    3. **피크 및 파라미터 조정:** 기준 피크 위치, `Tolerance (±eV)`, `Sigma` 고정 여부, `Lorentzian` 비율을 설정합니다.
    4. **실행 및 저장:** **Deconvolution 실행**을 누른 뒤 $R^2$, 피크 기여도, 그래프를 확인하고 `.txt` 또는 `.csv` 파일로 내려받습니다.
    """
)
