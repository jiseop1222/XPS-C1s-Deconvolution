from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

FEATURE_COLS = [
    "C=C",
    "C-C/C-N",
    "C=O",
    "O-C=O",
    "pi-pi*",
    "Pyridinic N",
    "NH2",
    "Pyrrolic N",
    "graphitic N",
    "Oxidized N",
    "ID_IG",
    "C_at",
    "N_at",
    "O_at",
    "Cu_at",
    "Crystallinity",
]

DEFAULT_TRAINING_CSV = Path(__file__).resolve().parent / "data" / "carbon_data_Temp.csv"

# XPS C1s / N1s 피크명 변형(공백, 대소문자, 슬래시, 하이픈 등)을 모델 피처로 매핑
PEAK_ALIAS_MAP = {
    "ccsp2": "C=C",
    "c=c": "C=C",
    "cncnsp3": "C-C/C-N",
    "c-c/c-n": "C-C/C-N",
    "c-n/c-csp3": "C-C/C-N",
    "c=o": "C=O",
    "o-c=o": "O-C=O",
    "oc=o": "O-C=O",
    "pi-pi*": "pi-pi*",
    "pipi*": "pi-pi*",
    "pyridinicn": "Pyridinic N",
    "pyridinic": "Pyridinic N",
    "nh2": "NH2",
    "pyrrolicn": "Pyrrolic N",
    "pyrrolic": "Pyrrolic N",
    "graphiticn": "graphitic N",
    "graphitic": "graphitic N",
    "oxidizedn": "Oxidized N",
    "oxidized": "Oxidized N",
}


def _norm_key(text: str) -> str:
    return "".join(str(text).strip().lower().split())


def extract_xps_ratios_from_summary(fitted_summary: list[dict] | None) -> dict[str, float]:
    """
    fitted_summary의 Contribution (%) 값을 0.0~1.0 비율로 변환해
    모델 입력 피처 딕셔너리로 반환합니다.
    """
    extracted: dict[str, float] = {}
    if not fitted_summary:
        return extracted

    for row in fitted_summary:
        peak_name = str(row.get("Peak Name", "")).strip()
        if not peak_name:
            continue
        raw_pct = row.get("Contribution (%)", 0.0)
        try:
            pct_val = float(raw_pct)
        except (TypeError, ValueError):
            continue

        ratio = max(0.0, min(1.0, pct_val / 100.0))
        norm = _norm_key(peak_name)
        target_col = PEAK_ALIAS_MAP.get(norm)

        # 부분 문자열 fallback 매칭
        if target_col is None:
            if "c-c" in norm and "sp2" in norm:
                target_col = "C=C"
            elif "sp3" in norm or "c-n" in norm:
                target_col = "C-C/C-N"
            elif "o-c=o" in norm:
                target_col = "O-C=O"
            elif "c=o" in norm:
                target_col = "C=O"
            elif "pi" in norm:
                target_col = "pi-pi*"
            elif "pyridinic" in norm:
                target_col = "Pyridinic N"
            elif "nh2" in norm:
                target_col = "NH2"
            elif "pyrrolic" in norm:
                target_col = "Pyrrolic N"
            elif "graphitic" in norm:
                target_col = "graphitic N"
            elif "oxidized" in norm:
                target_col = "Oxidized N"

        if target_col:
            extracted[target_col] = round(extracted.get(target_col, 0.0) + ratio, 4)

    return extracted


def load_training_dataset(csv_source=None) -> tuple[pd.DataFrame, list[str]]:
    """
    학습용 CSV를 로드하고 숫자형 정제 및 결측 제거를 수행합니다.
    """
    if csv_source is not None:
        df = pd.read_csv(csv_source, header=0)
    elif DEFAULT_TRAINING_CSV.exists():
        df = pd.read_csv(DEFAULT_TRAINING_CSV, header=0)
    else:
        raise FileNotFoundError(
            f"Default training CSV not found at `{DEFAULT_TRAINING_CSV}`. "
            "Upload `carbon_data_Temp.csv` in the sidebar."
        )

    df = df.dropna(how="all").reset_index(drop=True)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    needed_cols = FEATURE_COLS + ["Conductivity", "Carbonization Temp"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training dataset: {missing}")

    clean_df = df[needed_cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if len(clean_df) < 2:
        raise ValueError("Not enough valid rows after dropping missing values (need at least 2).")

    return clean_df, FEATURE_COLS


def fit_conductivity_and_temp_models(
    train_df: pd.DataFrame,
    n_estimators: int = 300,
    random_state: int = 42,
) -> dict:
    """
    1) 전기전도도 예측 모델(입력: 16개 피처 + Carbonization Temp, 타깃: Conductivity)
    2) 탄화온도 역산 모델(입력: 16개 피처 + Conductivity, 타깃: Carbonization Temp)
    을 학습합니다.
    """
    X_16 = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    temp_col = train_df["Carbonization Temp"].to_numpy(dtype=np.float32).reshape(-1, 1)
    cond_col = train_df["Conductivity"].to_numpy(dtype=np.float32)

    X_cond_model = np.hstack((X_16, temp_col))
    rf_cond = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf_cond.fit(X_cond_model, cond_col)
    cond_pred_train = rf_cond.predict(X_cond_model)
    r2_cond = r2_score(cond_col, cond_pred_train) if len(cond_col) > 1 else np.nan

    X_temp_model = np.hstack((X_16, cond_col.reshape(-1, 1)))
    temp_target = train_df["Carbonization Temp"].to_numpy(dtype=np.float32)
    rf_temp = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf_temp.fit(X_temp_model, temp_target)
    temp_pred_train = rf_temp.predict(X_temp_model)
    r2_temp = r2_score(temp_target, temp_pred_train) if len(temp_target) > 1 else np.nan

    cond_features = FEATURE_COLS + ["Carbonization Temp"]
    imp_df = (
        pd.DataFrame({
            "Feature": cond_features,
            "Importance (Conductivity Model)": rf_cond.feature_importances_,
        })
        .sort_values(by="Importance (Conductivity Model)", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "rf_cond": rf_cond,
        "rf_temp": rf_temp,
        "r2_cond": r2_cond,
        "r2_temp": r2_temp,
        "importance_df": imp_df,
        "n_samples": len(train_df),
    }


def predict_sigma_temp_curve(
    rf_cond: RandomForestRegressor,
    base_16_features: np.ndarray,
    t_min: int = 800,
    t_max: int = 1600,
    t_step: int = 25,
    sigma_req: float = 38.0,
) -> dict:
    """
    주어진 16개 XPS/구조 피처를 고정하고 탄화온도 T를 스윕하며
    예상 전기전도도(σ) 곡선과 목표 σ_req 달성 최소 탄화온도(T_min)를 계산합니다.
    """
    t_values = np.arange(t_min, t_max + 1, t_step, dtype=np.float32)
    x_batch = np.hstack((
        np.repeat(base_16_features.reshape(1, -1), len(t_values), axis=0),
        t_values.reshape(-1, 1),
    ))
    sigma_preds = rf_cond.predict(x_batch)

    matched = t_values[sigma_preds >= sigma_req]
    t_min_required = float(matched[0]) if len(matched) > 0 else None

    return {
        "T_values": t_values,
        "sigma_preds": sigma_preds,
        "sigma_req": float(sigma_req),
        "T_min_required": t_min_required,
    }


def predict_required_temp_from_target_sigma(
    rf_temp: RandomForestRegressor,
    base_16_features: np.ndarray,
    target_sigma: float,
) -> float:
    """
    목표 전기전도도(target_sigma)를 입력했을 때 탄화온도 회귀 모델이 예측하는 온도(°C)를 반환합니다.
    """
    x_query = np.hstack((
        base_16_features.reshape(1, -1),
        np.array([[float(target_sigma)]], dtype=np.float32),
    ))
    return float(rf_temp.predict(x_query)[0])