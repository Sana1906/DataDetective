"""Layered privacy detection and protection for DataDetective.

Detection combines three independent signals per column:
  1. Column name   (whole-word match, camelCase aware)
  2. Column values (regex + checksum validators: Verhoeff for Aadhaar, Luhn for cards)
  3. Statistics    (uniqueness = identifier-like, name-shaped text, PII inside free text)
Protection offers mask / hash (salted SHA-256) / generalize / drop, and a
k-anonymity before/after check shows how much re-identification risk was removed.
"""
import hashlib
import re
import secrets
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

RISK_LABEL = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low risk", "None": "🟢 Normal"}
RISK_RANK = {"High": 3, "Medium": 2, "Low": 1}
CONF_RANK = {"High": 3, "Medium": 2, "Low": 1}
ACTIONS = ["keep", "mask", "hash", "generalize", "drop"]

# ------------------------------------------------------------------ validators
_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
      [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
      [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
      [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def _verhoeff(num: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def _luhn(num: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def _digits(v: str) -> str:
    return re.sub(r"[\s-]", "", v)


EMAIL = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
PHONE = re.compile(r"^(?:\+?91)?[6-9]\d{9}$")
PAN = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
IFSC = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
IPV4 = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")
NAME_LIKE = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}$")
EMBED_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
EMBED_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")

# (category, risk, test, confidence)
VALUE_CHECKS = [
    ("Email address", "High", lambda v: bool(EMAIL.match(v)), "High"),
    ("Phone number", "High", lambda v: bool(PHONE.match(_digits(v))), "Medium"),
    ("Aadhaar number", "High",
     lambda v: bool(re.fullmatch(r"[2-9]\d{11}", _digits(v))) and _verhoeff(_digits(v)), "High"),
    ("PAN number", "High", lambda v: bool(PAN.match(v.upper())), "High"),
    ("Card number", "High",
     lambda v: bool(re.fullmatch(r"\d{13,19}", _digits(v))) and _luhn(_digits(v)), "High"),
    ("IFSC code", "Medium", lambda v: bool(IFSC.match(v.upper())), "Medium"),
    ("IP address", "Medium", lambda v: bool(IPV4.match(v)), "Medium"),
]

# ------------------------------------------------------------------ name rules
NAME_RULES = {
    "email": ("Email address", "High"), "mail": ("Email address", "High"),
    "phone": ("Phone number", "High"), "mobile": ("Phone number", "High"),
    "tel": ("Phone number", "High"), "contact": ("Phone number", "Medium"),
    "aadhaar": ("Aadhaar number", "High"), "aadhar": ("Aadhaar number", "High"),
    "uid": ("Aadhaar number", "Medium"), "pan": ("PAN number", "High"),
    "passport": ("Passport number", "High"), "ssn": ("National ID", "High"),
    "card": ("Card number", "High"), "credit": ("Card number", "High"), "cvv": ("Card number", "High"),
    "account": ("Bank account", "High"), "iban": ("Bank account", "High"), "ifsc": ("Bank account", "Medium"),
    "password": ("Credential", "High"), "pwd": ("Credential", "High"),
    "token": ("Credential", "High"), "secret": ("Credential", "High"),
    "address": ("Address", "High"), "street": ("Address", "High"),
    "pincode": ("Location", "Medium"), "zip": ("Location", "Medium"),
    "postal": ("Location", "Medium"), "city": ("Location", "Low"),
    "latitude": ("Geolocation", "Medium"), "longitude": ("Geolocation", "Medium"),
    "lat": ("Geolocation", "Medium"), "lon": ("Geolocation", "Medium"), "gps": ("Geolocation", "Medium"),
    "name": ("Person name", "Medium"), "firstname": ("Person name", "Medium"),
    "lastname": ("Person name", "Medium"), "fullname": ("Person name", "Medium"),
    "surname": ("Person name", "Medium"),
    "dob": ("Date of birth", "High"), "birth": ("Date of birth", "High"), "birthday": ("Date of birth", "High"),
    "salary": ("Financial", "Medium"), "income": ("Financial", "Medium"),
    "wage": ("Financial", "Medium"), "balance": ("Financial", "Medium"),
    "religion": ("Sensitive attribute", "High"), "caste": ("Sensitive attribute", "High"),
    "race": ("Sensitive attribute", "High"), "ethnicity": ("Sensitive attribute", "High"),
    "disease": ("Sensitive attribute", "High"), "diagnosis": ("Sensitive attribute", "High"),
    "disability": ("Sensitive attribute", "High"), "political": ("Sensitive attribute", "High"),
    "age": ("Age", "Low"), "gender": ("Gender", "Low"),
    "ip": ("IP address", "Medium"), "ipaddress": ("IP address", "Medium"),
    "id": ("Identifier", "Medium"), "userid": ("Identifier", "Medium"),
    "customerid": ("Identifier", "Medium"),
}

DEFAULT_ACTION = {
    "Email address": "mask", "Phone number": "mask", "IP address": "mask", "IFSC code": "mask",
    "Aadhaar number": "hash", "PAN number": "hash", "Passport number": "hash",
    "National ID": "hash", "Card number": "hash", "Bank account": "hash",
    "Person name": "hash", "Identifier": "hash",
    "Credential": "drop", "Address": "drop", "Sensitive attribute": "drop", "PII in free text": "drop",
    "Date of birth": "generalize", "Location": "generalize", "Geolocation": "generalize",
    "Financial": "generalize", "Age": "keep", "Gender": "keep",
}
QUASI_CATEGORIES = {"Date of birth", "Age", "Location", "Gender", "Geolocation"}


def _tokens(col) -> set:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(col))
    toks = {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}
    toks.add(re.sub(r"[^a-z0-9]", "", str(col).lower()))
    return toks


# ------------------------------------------------------------------ detection
def _scan_column(s: pd.Series) -> list:
    finds = []  # (category, risk, confidence, evidence, source)
    for t in _tokens(s.name):
        if t in NAME_RULES:
            cat, risk = NAME_RULES[t]
            finds.append((cat, risk, "Medium", f"Column name contains '{t}'", "name"))

    vals = s.dropna().astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    vals = vals[vals != ""]
    n = len(vals)
    if n < 5:
        return finds
    sample = vals.sample(min(n, 300), random_state=0)

    for cat, risk, test, conf in VALUE_CHECKS:
        try:
            share = float(sample.map(test).mean())
        except Exception:
            continue
        if share >= 0.6:
            extra = " (checksum valid)" if cat in ("Aadhaar number", "Card number") else ""
            finds.append((cat, risk, conf, f"{share:.0%} of values look like {cat.lower()}{extra}", "value"))

    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    if is_text:
        if sample.str.len().mean() > 20:
            em = float(sample.map(lambda v: bool(EMBED_EMAIL.search(v))).mean())
            ph = float(sample.map(lambda v: bool(EMBED_PHONE.search(v))).mean())
            if max(em, ph) >= 0.05:
                finds.append(("PII in free text", "High", "Medium",
                              f"Emails or phone numbers found inside text ({max(em, ph):.0%} of rows)", "value"))
        elif sample.map(lambda v: bool(NAME_LIKE.match(v))).mean() >= 0.7 and vals.nunique() / n > 0.5:
            finds.append(("Person name", "Medium", "Medium", "Values look like personal names", "value"))

    uniq = vals.nunique() / n
    if uniq >= 0.95 and n >= 20 and not pd.api.types.is_float_dtype(s):
        finds.append(("Identifier", "Medium", "Medium",
                      f"Unique in {uniq:.0%} of rows (can single a person out)", "stat"))
    return finds


def scan_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        finds = _scan_column(df[col])
        if not finds:
            rows.append({"Column": col, "Category": "-", "Risk": RISK_LABEL["None"], "Confidence": "-",
                         "Evidence": "No sensitive pattern found", "Recommended action": "keep"})
            continue
        best = max(finds, key=lambda f: (RISK_RANK[f[1]], CONF_RANK[f[2]]))
        cat, risk, conf = best[0], best[1], best[2]
        sources = {f[4] for f in finds if f[0] == cat}
        if len(sources) > 1:  # name AND values agree
            conf = "High"
        evidence = "; ".join(dict.fromkeys(f[3] for f in finds))[:220]
        rows.append({"Column": col, "Category": cat, "Risk": RISK_LABEL[risk], "Confidence": conf,
                     "Evidence": evidence, "Recommended action": DEFAULT_ACTION.get(cat, "keep")})
    return pd.DataFrame(rows)


def scan_compat(df: pd.DataFrame) -> pd.DataFrame:
    """Old-style (Column, Detection) table used by the report builder."""
    s = scan_dataset(df)
    s["Detection"] = [r if c == "-" else f"{r} - {c}" for r, c in zip(s["Risk"], s["Category"])]
    return s[["Column", "Detection", "Evidence"]]


def exposure_score(scan: pd.DataFrame) -> int:
    high = scan["Risk"].str.contains("High").sum()
    med = scan["Risk"].str.contains("Medium").sum()
    low = scan["Risk"].str.contains("Low risk").sum()
    return int(min(100, 20 * high + 8 * med + low))


# ------------------------------------------------------------------ protection
def _mask(v):
    if pd.isna(v):
        return v
    s = str(v)
    if "@" in s:
        user, dom = s.split("@", 1)
        return (user[:1] + "***@" + dom)
    return "*" * max(len(s) - 2, 0) + s[-2:]


def _generalize(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        clean = s.dropna()
        if len(clean) and clean.min() >= 0 and clean.max() <= 120 and (clean % 1 == 0).all():
            return s.map(lambda v: v if pd.isna(v) else f"{int(v // 10 * 10)}-{int(v // 10 * 10) + 9}")
        if s.nunique() > 6:
            return pd.qcut(s, q=5, duplicates="drop").astype(str).where(s.notna())
        return s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().mean() > 0.8:
        return parsed.dt.year.astype("Int64").astype(str).where(parsed.notna())
    return s.map(lambda v: v if pd.isna(v) else str(v)[:3] + "***")


def protect(df: pd.DataFrame, actions: dict, salt: str) -> pd.DataFrame:
    out = df.copy()
    for col, act in actions.items():
        if col not in out.columns or act == "keep":
            continue
        if act == "drop":
            out = out.drop(columns=col)
        elif act == "hash":
            out[col] = out[col].map(
                lambda v: v if pd.isna(v) else hashlib.sha256((salt + str(v)).encode()).hexdigest()[:12])
        elif act == "mask":
            out[col] = out[col].map(_mask)
        elif act == "generalize":
            out[col] = _generalize(out[col])
    return out


def k_anonymity(df: pd.DataFrame, cols: list):
    cols = [c for c in cols if c in df.columns]
    if not cols or df.empty:
        return None
    g = df.groupby(cols, dropna=False).size()
    return {"k": int(g.min()),
            "unique_pct": round(100 * g[g == 1].sum() / len(df), 1),
            "under5_pct": round(100 * g[g < 5].sum() / len(df), 1)}


# ------------------------------------------------------------------ Streamlit UI
def render_privacy(df: pd.DataFrame):
    import streamlit as st

    st.markdown('<div class="section-title">Privacy and Data Protection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Detection uses column names, value patterns with checksums '
                '(Aadhaar, card) and statistics. Then you choose how to protect each flagged column.</div>',
                unsafe_allow_html=True)

    scan = scan_dataset(df)
    flagged = scan[~scan["Risk"].str.contains("Normal")]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Columns scanned", len(scan))
    c2.metric("High risk", int(scan["Risk"].str.contains("High").sum()))
    c3.metric("Medium risk", int(scan["Risk"].str.contains("Medium").sum()))
    score = exposure_score(scan)
    c4.metric("Privacy exposure", f"{score}/100",
              help="0 means nothing sensitive was found. Higher means more exposure.")

    st.markdown("#### Findings")
    st.dataframe(scan[["Column", "Category", "Risk", "Confidence", "Evidence"]],
                 use_container_width=True, hide_index=True)

    if flagged.empty:
        st.success("No sensitive columns were detected.")
        return

    st.markdown("#### Protection plan")
    st.caption("Pick an action for each column. Defaults follow the data-minimization principle.")
    plan = flagged[["Column", "Category", "Recommended action"]].rename(columns={"Recommended action": "Action"})
    edited = st.data_editor(
        plan, hide_index=True, use_container_width=True, disabled=["Column", "Category"],
        column_config={"Action": st.column_config.SelectboxColumn("Action", options=ACTIONS, required=True)},
        key=f"privacy_plan_{abs(hash(tuple(df.columns)))}",
    )
    salt = st.session_state.setdefault("privacy_salt", secrets.token_hex(8))
    protected = protect(df, dict(zip(edited["Column"], edited["Action"])), salt)

    st.markdown("#### Re-identification check (k-anonymity)")
    default_q = [c for c, cat in zip(scan["Column"], scan["Category"]) if cat in QUASI_CATEGORIES]
    qcols = st.multiselect("Quasi-identifiers (columns that could identify someone when combined)",
                           list(df.columns), default=default_q)
    before, after = k_anonymity(df, qcols), k_anonymity(protected, qcols)
    if before and after:
        m1, m2, m3 = st.columns(3)
        m1.metric("Smallest group (k)", after["k"], after["k"] - before["k"])
        m2.metric("Rows unique on these columns", f"{after['unique_pct']}%",
                  f"{after['unique_pct'] - before['unique_pct']:+.1f} pts", delta_color="inverse")
        m3.metric("Rows in groups smaller than 5", f"{after['under5_pct']}%",
                  f"{after['under5_pct'] - before['under5_pct']:+.1f} pts", delta_color="inverse")
        st.caption("Deltas compare the protected data with the original. k = 1 means someone can be singled out.")
    else:
        st.info("Select at least one quasi-identifier that is still in the protected data.")

    st.markdown("#### Protected dataset preview")
    st.dataframe(protected.head(15), use_container_width=True)
    audit = edited.assign(Risk=flagged["Risk"].values,
                          Time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), Rows=len(df))
    d1, d2 = st.columns(2)
    d1.download_button("Download protected dataset (.csv)", protected.to_csv(index=False),
                       "protected_dataset.csv", "text/csv", use_container_width=True)
    d2.download_button("Download privacy audit log (.csv)", audit.to_csv(index=False),
                       "privacy_audit_log.csv", "text/csv", use_container_width=True)

    with st.expander("How this maps to data-protection principles"):
        st.markdown(
            "- **Data minimization** (DPDP Act 2023, GDPR Art. 5): drop columns the analysis does not need.\n"
            "- **Pseudonymisation** (GDPR Art. 4(5)): salted SHA-256 hashing replaces identifiers.\n"
            "- **Special categories** (GDPR Art. 9): religion, caste, health and similar columns are flagged High.\n"
            "- **Audit trail**: the log records which action was chosen for each column.\n\n"
            "Hashing is pseudonymisation, not anonymisation. This tool supports a human review "
            "and does not certify legal compliance.")
    st.caption("Limits: pattern detection can miss names inside free text, and rules cover Indian ID formats "
               "and common global ones. Always review the findings before sharing data.")
