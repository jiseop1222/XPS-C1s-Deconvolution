from datetime import datetime
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3
import streamlit as st

DB_PATH = Path(__file__).resolve().parent / "data" / "xps_history.db"
DEFAULT_HOST_EMAIL = "host@xps-studio.local"


def get_host_email() -> str:
    try:
        if "HOST_EMAIL" in st.secrets and st.secrets["HOST_EMAIL"]:
            return str(st.secrets["HOST_EMAIL"]).strip().lower()
    except Exception:
        pass
    return DEFAULT_HOST_EMAIL


def _hash_password(password: str, salt: str | None = None) -> str:
    if salt is None:
        salt = hashlib.sha256(datetime.now().isoformat().encode("utf-8")).hexdigest()[:16]
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
    return f"{salt}${digest}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected_digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    actual_digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
    return hmac.compare_digest(actual_digest, expected_digest)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    host_email = get_host_email()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_email TEXT NOT NULL,
                record_name TEXT NOT NULL,
                element TEXT NOT NULL,
                uploaded_filename TEXT NOT NULL,
                r2_score REAL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # 기본 호스트 계정 보장
        cur = conn.execute("SELECT id, role FROM users WHERE email = ?", (host_email,))
        row = cur.fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, 'host', ?)",
                (host_email, _hash_password("host1234!"), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
        elif row[1] != "host":
            conn.execute("UPDATE users SET role = 'host' WHERE email = ?", (host_email,))
        conn.commit()


def register_user(email: str, password: str) -> tuple[bool, str]:
    init_db()
    clean_email = email.strip().lower()
    if not clean_email or "@" not in clean_email:
        return False, "올바른 이메일 주소를 입력해 주세요."
    if len(password) < 6:
        return False, "비밀번호는 6자 이상으로 입력해 주세요."

    role = "host" if clean_email == get_host_email() else "user"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (clean_email, _hash_password(password), role, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
        return True, "회원가입이 완료되었습니다. 로그인해 주세요."
    except sqlite3.IntegrityError:
        return False, "이미 등록된 이메일입니다."


def authenticate_user(email: str, password: str) -> tuple[bool, dict | None, str]:
    init_db()
    clean_email = email.strip().lower()
    host_email = get_host_email()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT email, password_hash, role FROM users WHERE email = ?", (clean_email,))
        row = cur.fetchone()
        if row is None or not _verify_password(password, row[1]):
            return False, None, "이메일 또는 비밀번호가 올바르지 않습니다."
        role = "host" if row[0] == host_email else row[2]
        if role != row[2]:
            conn.execute("UPDATE users SET role = ? WHERE email = ?", (role, row[0]))
            conn.commit()
        return True, {"email": row[0], "role": role}, "로그인되었습니다."


def require_login() -> dict:
    init_db()
    user = st.session_state.get("auth_user")
    if not user:
        st.warning("로그인이 필요한 페이지입니다. 메인(`app.py`)에서 로그인해 주세요.")
        st.stop()
    return user


def clean_filename_token(text: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(text).strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.strip("_")


def build_record_name(uploaded_name: str, bg_mode: str, lorentzian_ratio: float) -> str:
    stem = clean_filename_token(
        uploaded_name.rsplit(".", 1)[0]
        if uploaded_name and "." in uploaded_name
        else (uploaded_name or "spectrum")
    )
    bg_clean = clean_filename_token(bg_mode)
    lor_clean = clean_filename_token(f"{lorentzian_ratio:g}").replace(".", "_")
    return f"{stem}_{bg_clean}_{lor_clean}"


def save_history_record(
    *,
    owner_email: str,
    element: str,
    uploaded_filename: str,
    bg_mode: str,
    lorentzian_ratio: float,
    payload: dict,
) -> str:
    init_db()
    record_name = build_record_name(uploaded_filename, bg_mode, lorentzian_ratio)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    enriched_payload = {
        **payload,
        "owner_email": owner_email,
        "record_name": record_name,
        "saved_at": now_str,
    }
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO history_records
                (owner_email, record_name, element, uploaded_filename, r2_score, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_email,
                record_name,
                element,
                uploaded_filename,
                payload.get("r2_score"),
                json.dumps(enriched_payload, ensure_ascii=False),
                now_str,
            ),
        )
        conn.commit()
    return record_name


def fetch_history_records(current_user: dict) -> list[dict]:
    init_db()
    is_host = current_user.get("role") == "host"
    with sqlite3.connect(DB_PATH) as conn:
        if is_host:
            cur = conn.execute(
                "SELECT id, owner_email, record_name, element, uploaded_filename, r2_score, payload_json, created_at FROM history_records ORDER BY id DESC"
            )
        else:
            cur = conn.execute(
                "SELECT id, owner_email, record_name, element, uploaded_filename, r2_score, payload_json, created_at FROM history_records WHERE owner_email = ? ORDER BY id DESC",
                (current_user["email"],),
            )
        rows = cur.fetchall()

    results = []
    for row in rows:
        try:
            data = json.loads(row[6])
        except Exception:
            data = {}
        data["_db_id"] = row[0]
        data["owner_email"] = data.get("owner_email") or row[1]
        data["record_name"] = data.get("record_name") or row[2]
        data["element"] = data.get("element") or row[3]
        data["uploaded_filename"] = data.get("uploaded_filename") or row[4]
        data["r2_score"] = data.get("r2_score") if data.get("r2_score") is not None else row[5]
        data["saved_at"] = data.get("saved_at") or row[7]
        results.append(data)
    return results


def delete_history_record(record_id: int, current_user: dict) -> None:
    init_db()
    is_host = current_user.get("role") == "host"
    with sqlite3.connect(DB_PATH) as conn:
        if is_host:
            conn.execute("DELETE FROM history_records WHERE id = ?", (record_id,))
        else:
            conn.execute("DELETE FROM history_records WHERE id = ? AND owner_email = ?", (record_id, current_user["email"]))
        conn.commit()