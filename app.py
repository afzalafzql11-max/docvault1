import os
import re
import sqlite3
import hashlib
import secrets
from datetime import datetime, date, timedelta
from pathlib import Path

import streamlit as st
from PIL import Image
import pandas as pd

try:
    from google import genai
except Exception:
    genai = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "documents.db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="DocuVault AI",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Database ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        original_name TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        doc_name TEXT,
        doc_type TEXT,
        issue_date TEXT,
        expiry_date TEXT,
        translated_text TEXT,
        extracted_text TEXT,
        uploaded_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- Security helpers ----------
def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000
    ).hex()
    return digest, salt

def verify_password(password, stored_hash, salt):
    digest, _ = hash_password(password, salt)
    return secrets.compare_digest(digest, stored_hash)

def valid_email(email):
    return re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or "") is not None

# ---------- Gemini ----------
def gemini_available():
    return bool(os.getenv("GEMINI_API_KEY")) and genai is not None

def get_gemini_client():
    if not gemini_available():
        return None
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_with_gemini(uploaded_file):
    """
    Uses the current Google GenAI SDK and Gemini Files API.
    This is important for PDFs: Gemini receives the actual PDF as a document,
    rather than a generic bytes dictionary.
    """
    client = get_gemini_client()
    if client is None:
        return None, "Gemini API is not configured."

    suffix = Path(uploaded_file.name).suffix.lower()
    mime = uploaded_file.type or (
        "application/pdf" if suffix == ".pdf" else "image/jpeg"
    )

    # Gemini's Files API accepts a file path/file-like object. We use a
    # temporary file so Streamlit's UploadedFile is handled consistently.
    import tempfile
    temp_path = None

    prompt = """
You are a document verification and information extraction assistant.

Analyze the uploaded document carefully. It may be a PDF, scanned document,
photograph, certificate, ID, licence, passport, insurance document, or other
official-style document.

Return ONLY valid JSON with exactly these keys:
{
  "document_name": "",
  "document_type": "",
  "person_name": "",
  "issue_date": "",
  "expiry_date": "",
  "translated_text": "",
  "confidence_note": ""
}

Rules:
1. Read the document visually and from its text.
2. If the document contains a language other than English, translate the
   important content into clear English.
3. document_name: identify the document title/name.
4. document_type: short category such as Passport, Driving Licence,
   Certificate, Insurance, Identity Document, etc.
5. person_name: extract the holder/owner name if clearly visible.
6. issue_date and expiry_date must be YYYY-MM-DD if clearly present.
7. If there is no expiry date, return an empty string.
8. Never invent a value. If uncertain, use an empty string.
9. translated_text should contain the important readable document content
   translated into English.
10. confidence_note should briefly mention if any field was unclear.
"""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        # Upload the actual document to Gemini's Files API.
        uploaded = client.files.upload(
            file=temp_path,
            config={"mime_type": mime}
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, uploaded],
        )

        text = (response.text or "").strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

        result = json.loads(text)

        required = [
            "document_name", "document_type", "person_name",
            "issue_date", "expiry_date", "translated_text",
            "confidence_note"
        ]
        for key in required:
            result.setdefault(key, "")

        return result, None

    except Exception as exc:
        return None, f"Gemini document processing failed: {exc}"
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

def local_ocr(uploaded_file):
    try:
        import pytesseract
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception:
        return ""

def parse_local_text(text):
    lower = text.lower()
    doc_type = "Unknown Document"
    choices = [
        ("Passport", ["passport"]),
        ("Driving Licence", ["driving licence", "driving license"]),
        ("Aadhaar Card", ["aadhaar", "uidai"]),
        ("PAN Card", ["income tax", "permanent account number", "pan"]),
        ("Voter ID", ["election commission", "voter"]),
        ("Birth Certificate", ["birth certificate"]),
        ("Degree Certificate", ["degree certificate", "university"]),
        ("Insurance", ["insurance", "policy"]),
    ]
    for name, keywords in choices:
        if any(k in lower for k in keywords):
            doc_type = name
            break

    dates = re.findall(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", text)
    expiry = ""
    for marker in ["expiry", "expires", "valid till", "valid until", "valid upto"]:
        idx = lower.find(marker)
        if idx >= 0:
            after = text[idx:idx+100]
            m = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", after)
            if m:
                expiry = m.group(1).replace("/", "-")
                break

    return {
        "document_name": doc_type,
        "document_type": doc_type,
        "person_name": "",
        "issue_date": "",
        "expiry_date": expiry,
        "translated_text": text[:5000],
        "confidence_note": "Local OCR fallback; please verify extracted fields."
    }

# ---------- Authentication ----------
def signup(full_name, email, password):
    if not full_name.strip() or not valid_email(email):
        return False, "Enter a valid name and email."
    if len(password) < 6:
        return False, "Password must contain at least 6 characters."

    conn = db()
    try:
        pwd_hash, salt = hash_password(password)
        conn.execute(
            "INSERT INTO users(full_name,email,password_hash,salt,created_at) VALUES(?,?,?,?,?)",
            (full_name.strip(), email.strip().lower(), pwd_hash, salt, datetime.utcnow().isoformat())
        )
        conn.commit()
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        conn.close()

def login(email, password):
    conn = db()
    row = conn.execute(
        "SELECT * FROM users WHERE email=?", (email.strip().lower(),)
    ).fetchone()
    conn.close()
    if row and verify_password(password, row["password_hash"], row["salt"]):
        return dict(row)
    return None

# ---------- Documents ----------
def save_document(user_id, original_name, file_bytes, metadata, translated, extracted):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name)
    unique = f"{user_id}_{secrets.token_hex(8)}_{safe}"
    path = UPLOAD_DIR / unique
    path.write_bytes(file_bytes)

    conn = db()
    conn.execute(
        """INSERT INTO documents
        (user_id, original_name, stored_path, doc_name, doc_type, issue_date,
         expiry_date, translated_text, extracted_text, uploaded_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            user_id, original_name, str(path),
            metadata.get("document_name", ""),
            metadata.get("document_type", ""),
            metadata.get("issue_date", ""),
            metadata.get("expiry_date", ""),
            translated,
            extracted,
            datetime.utcnow().isoformat(),
        )
    )
    conn.commit()
    conn.close()

def get_documents(user_id):
    conn = db()
    rows = conn.execute(
        "SELECT * FROM documents WHERE user_id=? ORDER BY uploaded_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_document(doc_id, user_id):
    conn = db()
    row = conn.execute(
        "SELECT stored_path FROM documents WHERE id=? AND user_id=?",
        (doc_id, user_id)
    ).fetchone()
    if row:
        try:
            Path(row["stored_path"]).unlink(missing_ok=True)
        except Exception:
            pass
        conn.execute("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))
        conn.commit()
    conn.close()

def days_to_expiry(expiry):
    if not expiry:
        return None
    try:
        return (date.fromisoformat(expiry) - date.today()).days
    except Exception:
        return None

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.main-title {font-size: 2.3rem; font-weight: 800; margin-bottom: .2rem;}
.subtitle {color: #64748b; margin-bottom: 1.4rem;}
.doc-card {
    padding: 1rem; border: 1px solid #e5e7eb; border-radius: 14px;
    background: #ffffff; margin-bottom: .8rem;
}
.warning {padding: .8rem 1rem; border-radius: 10px; background: #fff7ed; border:1px solid #fed7aa;}
.danger {padding: .8rem 1rem; border-radius: 10px; background: #fef2f2; border:1px solid #fecaca;}
.success {padding: .8rem 1rem; border-radius: 10px; background: #f0fdf4; border:1px solid #bbf7d0;}
</style>
""", unsafe_allow_html=True)

# ---------- Login screen ----------
if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    st.markdown('<div class="main-title">📁 DocuVault AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A DigiLocker-style document organizer with AI extraction, '
        'English translation, validity tracking and reminders.</div>',
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            user = login(email, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid email or password.")

    with tab2:
        with st.form("signup_form"):
            full_name = st.text_input("Full name")
            email2 = st.text_input("Email address")
            password2 = st.text_input("Create password", type="password")
            password3 = st.text_input("Confirm password", type="password")
            submitted2 = st.form_submit_button("Create Account", use_container_width=True)
        if submitted2:
            if password2 != password3:
                st.error("Passwords do not match.")
            else:
                ok, msg = signup(full_name, email2, password2)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.info(
        "AI features are optional. The app still runs if GEMINI_API_KEY is not configured. "
        "For public deployment, use an external persistent database/storage; Render Free "
        "web-service local files are ephemeral."
    )
    st.stop()

# ---------- Main app ----------
user = st.session_state.user
st.sidebar.title("📁 DocuVault AI")
st.sidebar.write(f"Signed in as **{user['full_name']}**")
page = st.sidebar.radio("Navigate", ["Dashboard", "Upload Document", "My Documents", "Calendar", "Settings"])
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

docs = get_documents(user["id"])

if page == "Dashboard":
    st.markdown('<div class="main-title">Welcome back 👋</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your documents, validity and reminders in one place.</div>', unsafe_allow_html=True)

    total = len(docs)
    expiring = 0
    expired = 0
    valid = 0
    for d in docs:
        days = days_to_expiry(d["expiry_date"])
        if days is None:
            continue
        if days < 0:
            expired += 1
        elif days <= 30:
            expiring += 1
        else:
            valid += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", total)
    c2.metric("Valid", valid)
    c3.metric("Ending ≤ 30 days", expiring)
    c4.metric("Expired", expired)

    for d in docs:
        days = days_to_expiry(d["expiry_date"])
        if days is not None and days < 0:
            st.markdown(
                f'<div class="danger">⚠️ <b>{d["doc_name"] or d["original_name"]}</b> expired on '
                f'{d["expiry_date"]}.</div>', unsafe_allow_html=True)
        elif days is not None and days <= 30:
            st.markdown(
                f'<div class="warning">🔔 <b>{d["doc_name"] or d["original_name"]}</b> '
                f'expires in {days} day(s) on {d["expiry_date"]}.</div>', unsafe_allow_html=True)

    st.subheader("Recent documents")
    if docs:
        table = pd.DataFrame([{
            "Document": d["doc_name"] or d["original_name"],
            "Type": d["doc_type"] or "—",
            "Holder": d.get("person_name") or "—",
            "Expiry": d["expiry_date"] or "No expiry found",
            "Uploaded": d["uploaded_at"][:10],
        } for d in docs[:10]])
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No documents yet. Open 'Upload Document' to add one.")

elif page == "Upload Document":
    st.markdown('<div class="main-title">Upload & Scan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload a PDF or image. Gemini can extract fields and translate '
        'important text into English.</div>', unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose a document",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        help="For best AI extraction, use a clear scan/photo."
    )

    if uploaded:
        st.write(f"**File:** {uploaded.name} • {uploaded.size/1024:.1f} KB")
        if uploaded.type.startswith("image/"):
            st.image(uploaded, caption="Document preview", width=420)

        if st.button("🔎 Scan Document", type="primary", use_container_width=True):
            with st.spinner("Scanning document and extracting information..."):
                metadata, err = extract_with_gemini(uploaded)

                if metadata is None:
                    if uploaded.type.startswith("image/"):
                        ocr_text = local_ocr(uploaded)
                        metadata = parse_local_text(ocr_text)
                        err = err + " Using local OCR fallback."
                    else:
                        metadata = {
                            "document_name": "Unclassified Document",
                            "document_type": "Unknown",
                            "person_name": "",
                            "issue_date": "",
                            "expiry_date": "",
                            "translated_text": "",
                            "confidence_note": err,
                        }

                st.session_state.scan_result = metadata
                st.session_state.scan_error = err

    if "scan_result" in st.session_state:
        result = st.session_state.scan_result
        if st.session_state.get("scan_error"):
            st.warning(st.session_state.scan_error)

        st.subheader("Review extracted information")
        col1, col2 = st.columns(2)
        with col1:
            doc_name = st.text_input("Document name", result.get("document_name", ""))
            doc_type = st.text_input("Document type", result.get("document_type", ""))
            person_name = st.text_input("Holder name", result.get("person_name", ""))
        with col2:
            issue_date = st.text_input("Issue date (YYYY-MM-DD)", result.get("issue_date", ""))
            expiry_date = st.text_input("Expiry date (YYYY-MM-DD)", result.get("expiry_date", ""))

        translated = st.text_area(
            "English / translated text",
            result.get("translated_text", ""),
            height=180
        )

        st.caption(result.get("confidence_note", "Please verify important fields against the original document."))

        if uploaded and st.button("💾 Save to My Documents", type="primary", use_container_width=True):
            metadata = {
                "document_name": doc_name,
                "document_type": doc_type,
                "person_name": person_name,
                "issue_date": issue_date,
                "expiry_date": expiry_date,
            }
            save_document(
                user["id"], uploaded.name, uploaded.getvalue(),
                metadata, translated, translated
            )
            st.session_state.pop("scan_result", None)
            st.session_state.pop("scan_error", None)
            st.success("Document saved successfully.")
            st.rerun()

elif page == "My Documents":
    st.markdown('<div class="main-title">My Documents</div>', unsafe_allow_html=True)
    if not docs:
        st.info("No documents stored yet.")
    else:
        for d in docs:
            with st.expander(f"📄 {d['doc_name'] or d['original_name']}"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Type:** {d['doc_type'] or '—'}")
                c2.write(f"**Name:** {d.get('person_name') or '—'}")
                c3.write(f"**Expiry:** {d['expiry_date'] or 'No expiry'}")

                days = days_to_expiry(d["expiry_date"])
                if days is not None:
                    if days < 0:
                        st.error(f"Expired {abs(days)} day(s) ago.")
                    elif days <= 30:
                        st.warning(f"Expires in {days} day(s).")
                    else:
                        st.success(f"Valid for {days} more day(s).")

                if d["translated_text"]:
                    st.text_area("Stored English text", d["translated_text"], height=150, key=f"txt_{d['id']}")

                if d["stored_path"] and Path(d["stored_path"]).exists():
                    try:
                        with open(d["stored_path"], "rb") as f:
                            st.download_button(
                                "⬇️ Download original document",
                                f.read(),
                                file_name=d["original_name"],
                                key=f"dl_{d['id']}"
                            )
                    except Exception:
                        pass

                if st.button("Delete document", key=f"del_{d['id']}"):
                    delete_document(d["id"], user["id"])
                    st.rerun()

elif page == "Calendar":
    st.markdown('<div class="main-title">📅 Validity Calendar</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Expiry dates are shown relative to today.</div>', unsafe_allow_html=True)

    valid_dates = []
    for d in docs:
        if d["expiry_date"]:
            try:
                valid_dates.append((date.fromisoformat(d["expiry_date"]), d))
            except Exception:
                pass

    if not valid_dates:
        st.info("No valid expiry dates are stored yet.")
    else:
        cal = pd.DataFrame([
            {
                "Date": dt,
                "Document": d["doc_name"] or d["original_name"],
                "Type": d["doc_type"] or "—",
                "Days Remaining": days_to_expiry(d["expiry_date"]),
            }
            for dt, d in sorted(valid_dates)
        ])
        st.dataframe(cal, use_container_width=True, hide_index=True)

        st.subheader("Upcoming expirations")
        upcoming = cal[(cal["Days Remaining"] >= 0) & (cal["Days Remaining"] <= 365)]
        if upcoming.empty:
            st.write("No expirations in the next year.")
        else:
            st.dataframe(upcoming, use_container_width=True, hide_index=True)

elif page == "Settings":
    st.markdown('<div class="main-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.write(f"**Name:** {user['full_name']}")
    st.write(f"**Email:** {user['email']}")
    st.write("**AI status:** " + ("Connected" if gemini_available() else "Not configured"))
    st.info(
        "Reminder logic is calculated from the current date every time the dashboard loads. "
        "For true email/SMS push notifications, add a separate notification service."
    )
    st.warning(
        "IMPORTANT FOR RENDER FREE: local files and SQLite are ephemeral. "
        "For a real multi-user document vault, connect an external persistent database "
        "and object storage such as Supabase before treating this as production storage."
    )
