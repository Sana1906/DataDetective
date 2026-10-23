"""
DataDetective — An Autonomous Data Investigation & ML Platform
================================================================
A Streamlit web app that lets a user upload a real dataset (CSV / Excel)
and automatically:
  1. Understands the dataset (types, shape, target candidates)
  2. Scores its Data Health (missing values, duplicates, outliers, imbalance)
  3. Runs a Privacy Scan (flags likely sensitive columns)
  4. Performs context-aware, automated EDA
  5. Runs "Data Detective Mode" — hidden pattern & anomaly discovery
  6. Trains and compares multiple ML models (classification or regression,
     auto-detected from the chosen target column)
  7. Explains *why* the best model was chosen and flags possible data leakage
  8. Generates a downloadable investigation report

Run with:
    streamlit run app.py
"""

import io
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re as _re

# Streamlit treats HTML indented by 4+ spaces (or containing blank lines) as a
# code block. Normalise every HTML snippet so it renders as real HTML.
_orig_markdown = st.markdown


def _markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = _re.sub(r"(?m)^[ \t]+", "", body)
        body = _re.sub(r"\n\s*\n", "\n", body)
    return _orig_markdown(body, *args, **kwargs)


st.markdown = _markdown
import extensions as ext
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Optional: gradient boosting if available, else silently skip
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ----------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DataDetective",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f7fb;
        color: #172033;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1f3a;
        border-right: 1px solid #173455;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    section[data-testid="stSidebar"] * {
        color: #eef5ff !important;
    }

    .brand {
        padding: 6px 8px 20px 8px;
    }

    .brand-name {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    .brand-name span {
        color: #4d9cff;
    }

    .brand-tagline {
        color: #91a7c2 !important;
        font-size: 11px;
        margin-top: 4px;
        line-height: 1.4;
    }

    .sidebar-label {
        color: #7890ad !important;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 18px 8px 8px;
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 22px;
        box-shadow: 0 5px 18px rgba(22, 43, 72, 0.04);
    }

    .header-title {
        font-size: 25px;
        font-weight: 800;
        color: #10233f;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #718096;
        font-size: 12px;
        margin-top: 4px;
    }

    .secure-pill {
        background: #edf7f2;
        color: #16845a;
        border: 1px solid #cdebdc;
        border-radius: 20px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 700;
    }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 14px;
        padding: 18px;
        min-height: 118px;
        box-shadow: 0 5px 18px rgba(22, 43, 72, 0.035);
    }

    .kpi-label {
        color: #74839a;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .4px;
    }

    .kpi-value {
        color: #10233f;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-note {
        color: #16845a;
        font-size: 11px;
        margin-top: 5px;
        font-weight: 600;
    }

    /* Section headers */
    .section-title {
        color: #10233f;
        font-size: 19px;
        font-weight: 750;
        margin: 22px 0 10px;
    }

    .section-caption {
        color: #7a8799;
        font-size: 12px;
        margin-bottom: 14px;
    }

    /* Welcome / empty state */
    .welcome {
        background: linear-gradient(135deg, #0b1f3a 0%, #12355e 100%);
        border-radius: 18px;
        padding: 30px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px rgba(11, 31, 58, .16);
    }

    .welcome h1 {
        margin: 0 0 8px 0;
        font-size: 30px;
        letter-spacing: -0.7px;
    }

    .welcome p {
        color: #c5d5e8;
        font-size: 13px;
        max-width: 720px;
        line-height: 1.6;
    }

    .workflow {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 22px;
    }

    .workflow-item {
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 11px;
        padding: 14px;
    }

    .workflow-number {
        color: #66a9ff;
        font-weight: 800;
        font-size: 11px;
    }

    .workflow-name {
        font-size: 13px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Discovery */
    .discovery-box {
        border: 1px solid #dbe6f4;
        border-left: 4px solid #347fdb;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(22,43,72,.035);
    }

    .anomaly-box {
        border: 1px solid #f0d8c9;
        border-left: 4px solid #df784f;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: #fffaf7;
        border-radius: 10px;
    }

    /* Health score */
    .health-panel {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 5px 18px rgba(22,43,72,.035);
    }

    .big-score {
        font-size: 48px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #102a4b;
        border-radius: 12px;
        padding: 4px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #102a4b !important;
        border: 1px dashed #4a6f98 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #687890;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #246fc1;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 650;
        border: 1px solid #d7e0eb;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #e3e9f1;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #91a0b2;
        font-size: 10px;
        padding: 25px 0 8px;
    }

    @media (max-width: 900px) {
        .workflow {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    
    /* Reset workspace button color */
section[data-testid="stSidebar"] button {
    background-color: #7C3AED !important;
    color: #FFFFFF !important;
    border: 1px solid #A78BFA !important;
    border-radius: 12px !important;
}

section[data-testid="stSidebar"] button:hover {
    background-color: #6D28D9 !important;
    color: #FFFFFF !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "report_log" not in st.session_state:
    st.session_state.report_log = []


def log(section: str, text: str):
    """Append a line to the report log so the final report can be built."""
    st.session_state.report_log.append((section, text))


# ----------------------------------------------------------------------
# ---- STAGE 1: Dataset Understanding ----
# ----------------------------------------------------------------------
def understand_dataset(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    # try to auto-detect date-like object columns
    for c in categorical_cols[:]:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(20)
            if len(sample) and sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.7:
                datetime_cols.append(c)
                categorical_cols.remove(c)

    # candidate targets: low-cardinality categorical, or any numeric column
    target_candidates = [
        c for c in categorical_cols if 1 < df[c].nunique() <= 20
    ] + numeric_cols

    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "target_candidates": target_candidates,
    }


# ----------------------------------------------------------------------
# ---- STAGE 2: Data Quality / Health Score ----
# ----------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> dict:
    n = len(df)
    missing = df.isna().sum()
    missing_pct = (missing.sum() / (df.shape[0] * df.shape[1]) * 100) if n else 0
    duplicates = df.duplicated().sum()
    duplicate_pct = (duplicates / n * 100) if n else 0

    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_counts = {}
    total_outliers = 0
    for c in numeric_cols:
        col = df[c].dropna()
        if len(col) < 5:
            continue
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cnt = ((col < lower) | (col > upper)).sum()
        if cnt > 0:
            outlier_counts[c] = int(cnt)
            total_outliers += cnt

    # highly correlated pairs (potential redundancy)
    high_corr_pairs = []
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.notna(val) and val > 0.9:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], round(val, 3)))

    # class imbalance check on low-cardinality categorical columns
    imbalanced_cols = []
    for c in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[c].value_counts(normalize=True)
        if 1 < len(vc) <= 15 and vc.iloc[0] > 0.85:
            imbalanced_cols.append((c, round(vc.iloc[0] * 100, 1)))

    # ---- Health score (100 = perfect) ----
    score = 100
    score -= min(missing_pct * 1.5, 30)
    score -= min(duplicate_pct * 1.5, 15)
    outlier_pct = (total_outliers / (n * max(len(numeric_cols), 1)) * 100) if n else 0
    score -= min(outlier_pct * 2, 20)
    score -= min(len(high_corr_pairs) * 3, 15)
    score -= min(len(imbalanced_cols) * 5, 20)
    score = max(0, round(score))

    if score >= 85:
        verdict = "Excellent — ready for modeling"
        color = "#00b894"
    elif score >= 65:
        verdict = "Good — minor cleanup recommended"
        color = "#fdcb6e"
    elif score >= 45:
        verdict = "Fair — cleaning strongly recommended before modeling"
        color = "#e17055"
    else:
        verdict = "Poor — this dataset needs significant cleaning"
        color = "#d63031"

    return {
        "score": score,
        "verdict": verdict,
        "color": color,
        "missing": missing[missing > 0].sort_values(ascending=False),
        "missing_pct": round(missing_pct, 2),
        "duplicates": int(duplicates),
        "duplicate_pct": round(duplicate_pct, 2),
        "outlier_counts": outlier_counts,
        "total_outliers": int(total_outliers),
        "high_corr_pairs": high_corr_pairs,
        "imbalanced_cols": imbalanced_cols,
    }


# ----------------------------------------------------------------------
# ---- Privacy / Security Scan ----
# ----------------------------------------------------------------------
SENSITIVE_KEYWORDS = {
    "email": "🔴 Sensitive (PII)",
    "phone": "🔴 Sensitive (PII)",
    "mobile": "🔴 Sensitive (PII)",
    "ssn": "🔴 Sensitive (ID)",
    "aadhar": "🔴 Sensitive (ID)",
    "passport": "🔴 Sensitive (ID)",
    "credit": "🔴 Sensitive (Financial)",
    "card": "🔴 Sensitive (Financial)",
    "salary": "🟡 Potentially sensitive (Financial)",
    "income": "🟡 Potentially sensitive (Financial)",
    "address": "🔴 Sensitive (PII)",
    "name": "🟡 Potentially sensitive (PII)",
    "dob": "🟡 Potentially sensitive (PII)",
    "birth": "🟡 Potentially sensitive (PII)",
    "age": "🟢 Low risk",
    "gender": "🟢 Low risk",
    "password": "🔴 Sensitive (Credential)",
    "ip_address": "🔴 Sensitive (Technical)",
}


def privacy_scan(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        lc = col.lower()
        detection = "🟢 Normal"
        for kw, label in SENSITIVE_KEYWORDS.items():
            if kw in lc:
                detection = label
                break
        rows.append({"Column": col, "Detection": detection})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# ---- STAGE 3: Automated, context-aware EDA ----
# ----------------------------------------------------------------------
def render_eda(df: pd.DataFrame, info: dict):
    numeric_cols = info["numeric_cols"]
    categorical_cols = info["categorical_cols"]

    st.subheader("📊 Distributions")
    if numeric_cols:
        col_choice = st.multiselect(
            "Numeric columns to visualize",
            numeric_cols,
            default=numeric_cols[: min(4, len(numeric_cols))],
        )
        cols = st.columns(2)
        for i, c in enumerate(col_choice):
            fig = px.histogram(df, x=c, nbins=30, title=f"Distribution of {c}", marginal="box")
            cols[i % 2].plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns detected for distribution plots.")

    if categorical_cols:
        st.subheader("🧩 Category Breakdown")
        cat_choice = st.selectbox("Categorical column", categorical_cols)
        vc = df[cat_choice].value_counts().head(15).reset_index()
        vc.columns = [cat_choice, "count"]
        fig = px.bar(vc, x=cat_choice, y="count", title=f"Top categories in {cat_choice}")
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Analysis")
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation heatmap",
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.subheader("🔍 Feature Relationship Explorer")
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", numeric_cols, index=0, key="rel_x")
        y = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="rel_y")
        color_by = None
        if categorical_cols:
            color_by = st.selectbox("Color by (optional)", ["None"] + categorical_cols, key="rel_color")
            color_by = None if color_by == "None" else color_by
        fig = px.scatter(df, x=x, y=y, color=color_by, title=f"{x} vs {y}")
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# ---- STAGE 4: Data Detective Mode — hidden pattern & anomaly discovery ----
# ----------------------------------------------------------------------
def discover_patterns(df: pd.DataFrame, info: dict) -> list:
    discoveries = []
    numeric_cols = info["numeric_cols"]
    categorical_cols = info["categorical_cols"]

    # 1. Strong correlations the user probably didn't ask about
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.notna(val) and 0.7 <= val < 1.0:
                    pairs.append((corr.columns[i], corr.columns[j], val))
        pairs.sort(key=lambda x: -x[2])
        for a, b, v in pairs[:3]:
            discoveries.append(
                f"An unusually strong relationship was found between **{a}** and **{b}** "
                f"(correlation = {v:.2f})."
            )

    # 2. Subgroup behaves differently (categorical vs numeric mean shift)
    if categorical_cols and numeric_cols:
        for cat in categorical_cols[:3]:
            if df[cat].nunique() < 2 or df[cat].nunique() > 12:
                continue
            for num in numeric_cols[:3]:
                overall_mean = df[num].mean()
                group_means = df.groupby(cat)[num].mean()
                if overall_mean == 0 or pd.isna(overall_mean):
                    continue
                dev = ((group_means - overall_mean).abs() / (abs(overall_mean) + 1e-9))
                worst = dev.idxmax()
                if dev.max() > 0.5:
                    discoveries.append(
                        f"Group **{worst}** in **{cat}** shows a notably different average "
                        f"**{num}** ({group_means[worst]:.2f}) compared to the overall average "
                        f"({overall_mean:.2f})."
                    )
                    break

    # 3. Sudden trend break over a datetime column
    for dcol in info["datetime_cols"]:
        try:
            ts = pd.to_datetime(df[dcol], errors="coerce")
            if ts.notna().sum() < 10 or not numeric_cols:
                continue
            tmp = df.copy()
            tmp["_date"] = ts
            tmp = tmp.dropna(subset=["_date"]).sort_values("_date")
            num = numeric_cols[0]
            monthly = tmp.set_index("_date")[num].resample("M").mean().dropna()
            if len(monthly) > 3:
                pct_change = monthly.pct_change().abs()
                spike = pct_change.idxmax()
                if pct_change.max() > 0.4:
                    discoveries.append(
                        f"A sudden change in **{num}** was detected around "
                        f"**{spike.strftime('%B %Y')}** (change of {pct_change.max()*100:.0f}%)."
                    )
        except Exception:
            continue

    if not discoveries:
        discoveries.append(
            "No strongly unusual relationships were detected — the dataset looks fairly consistent."
        )
    return discoveries


def detect_anomalies(df: pd.DataFrame, info: dict):
    numeric_cols = info["numeric_cols"]
    if len(numeric_cols) < 2 or len(df) < 20:
        return None, "Not enough numeric data to run anomaly detection."

    X = df[numeric_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(X)
    scores = model.decision_function(X)

    result = df.copy()
    result["_anomaly"] = preds == -1
    result["_anomaly_score"] = scores
    n_anomalies = int(result["_anomaly"].sum())
    pct = n_anomalies / len(df) * 100
    return result, f"{n_anomalies} records ({pct:.1f}%) flagged as unusual out of {len(df)}."


# ----------------------------------------------------------------------
# ---- STAGE 5: ML Engine (auto classification / regression) ----
# ----------------------------------------------------------------------
def prep_features(df: pd.DataFrame, target: str, feature_cols: list):
    X = df[feature_cols].copy()
    y = df[target].copy()

    # drop rows where target is missing
    mask = y.notna()
    X, y = X[mask], y[mask]

    # encode categoricals
    encoders = {}
    for c in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[c] = X[c].astype(str).fillna("missing")
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le

    # fill numeric NaNs with median
    for c in X.select_dtypes(include=np.number).columns:
        X[c] = X[c].fillna(X[c].median())

    is_classification = (y.dtype == object) or (y.nunique() <= 15 and y.dtype != float)
    if is_classification:
        y_le = LabelEncoder()
        y_enc = y_le.fit_transform(y.astype(str))
        return X, y_enc, True, y_le
    return X, y.values, False, None


def check_leakage(X: pd.DataFrame, y, is_classification: bool) -> list:
    """Flag features suspiciously perfectly correlated with the target."""
    warnings = []
    try:
        if is_classification:
            for c in X.columns:
                # a feature with as many unique values as rows, or that perfectly
                # separates the target, is suspicious
                if X[c].nunique() == len(X):
                    warnings.append(
                        f"**{c}** has a unique value for every row — it may be an identifier "
                        f"leaking information rather than a real predictor."
                    )
        else:
            corr = X.assign(_target=y).corr(numeric_only=True)["_target"].drop("_target", errors="ignore")
            for c, v in corr.items():
                if pd.notna(v) and abs(v) > 0.98:
                    warnings.append(
                        f"**{c}** is correlated with the target at {v:.3f} — almost too perfectly. "
                        f"Check whether it is derived from the target itself."
                    )
    except Exception:
        pass
    return warnings


def run_ml_lab(df: pd.DataFrame, info: dict):
    st.subheader("🎯 Choose a target column")
    target = st.selectbox("What should the model predict?", info["target_candidates"])
    feature_cols = [c for c in df.columns if c != target]
    feature_cols = st.multiselect(
        "Feature columns to use", feature_cols, default=feature_cols[: min(10, len(feature_cols))]
    )

    if not feature_cols:
        st.warning("Select at least one feature column.")
        return

    if st.button("🚀 Run automated model comparison", type="primary"):
        with st.spinner("Training and comparing models..."):
            X, y, is_classification, y_encoder = prep_features(df, target, feature_cols)

            if len(X) < 20:
                st.error("Not enough rows with a valid target value to train a model (need at least 20).")
                return

            leakage_warnings = check_leakage(X, y, is_classification)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42,
                stratify=y if is_classification and len(np.unique(y)) > 1 else None,
            )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            results = []
            models = {}

            if is_classification:
                candidates = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
                    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                }
                if HAS_XGB and len(np.unique(y)) > 1:
                    candidates["XGBoost"] = XGBClassifier(
                        eval_metric="logloss", random_state=42, verbosity=0
                    )
                for name, model in candidates.items():
                    try:
                        use_scaled = name == "Logistic Regression"
                        model.fit(X_train_s if use_scaled else X_train, y_train)
                        preds = model.predict(X_test_s if use_scaled else X_test)
                        avg = "binary" if len(np.unique(y)) == 2 else "weighted"
                        results.append({
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, preds),
                            "Precision": precision_score(y_test, preds, average=avg, zero_division=0),
                            "Recall": recall_score(y_test, preds, average=avg, zero_division=0),
                            "F1": f1_score(y_test, preds, average=avg, zero_division=0),
                        })
                        models[name] = model
                    except Exception as e:
                        st.warning(f"{name} failed: {e}")
                metric_for_best = "F1"
            else:
                candidates = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
                    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
                }
                if HAS_XGB:
                    candidates["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
                for name, model in candidates.items():
                    try:
                        use_scaled = name == "Linear Regression"
                        model.fit(X_train_s if use_scaled else X_train, y_train)
                        preds = model.predict(X_test_s if use_scaled else X_test)
                        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                        results.append({
                            "Model": name,
                            "R2": r2_score(y_test, preds),
                            "MAE": mean_absolute_error(y_test, preds),
                            "RMSE": rmse,
                        })
                        models[name] = model
                    except Exception as e:
                        st.warning(f"{name} failed: {e}")
                metric_for_best = "R2"

            if not results:
                st.error("All models failed to train — check your feature/target selection.")
                return

            results_df = pd.DataFrame(results).sort_values(metric_for_best, ascending=False)
            best_row = results_df.iloc[0]
            best_name = best_row["Model"]
            best_model = models[best_name]

            st.session_state["ml_results"] = {
                "target": target,
                "is_classification": is_classification,
                "results_df": results_df,
                "best_name": best_name,
                "leakage_warnings": leakage_warnings,
                "feature_cols": feature_cols,
            }

        st.success(f"Best model: **{best_name}**")
        st.dataframe(results_df.style.format(precision=3), use_container_width=True)

        fig = px.bar(
            results_df, x="Model", y=metric_for_best,
            title=f"Model comparison ({metric_for_best})", color="Model",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Why this model? ---
        st.subheader("🧠 Why this model?")
        reasons = []
        n_rows = len(X)
        n_features = len(feature_cols)
        if best_name == "Random Forest":
            reasons += [
                "Handles nonlinear relationships and mixed feature types well.",
                "Robust to outliers compared to linear models.",
            ]
        elif best_name == "XGBoost":
            reasons += [
                "Gradient boosting often captures complex feature interactions.",
                "Regularization helps control overfitting on this dataset size.",
            ]
        elif "Linear" in best_name or "Logistic" in best_name:
            reasons += [
                "The relationship between features and target appears largely linear.",
                "Simpler model reduces overfitting risk on this dataset.",
            ]
        else:
            reasons.append("Performed best among the tested candidates on the held-out test set.")
        reasons.append(f"Dataset size: {n_rows} rows, {n_features} features used for training.")
        for r in reasons:
            st.markdown(f"- {r}")

        # --- Leakage warnings ---
        if leakage_warnings:
            st.subheader("🚨 Possible data leakage detected")
            for w in leakage_warnings:
                st.markdown(f"<div class='anomaly-box'>⚠️ {w}</div>", unsafe_allow_html=True)
        else:
            st.info("No obvious data-leakage red flags detected in the selected features.")

        # --- Feature importance ---
        if hasattr(best_model, "feature_importances_"):
            st.subheader("📌 Feature Importance")
            fi = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": best_model.feature_importances_,
            }).sort_values("Importance", ascending=False)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="What drives the prediction?")
            st.plotly_chart(fig, use_container_width=True)

        log("ML Lab", f"Target: {target}. Best model: {best_name}.")
        log("ML Lab", results_df.to_string(index=False))


# ----------------------------------------------------------------------
# ---- STAGE 6: Report ----
# ----------------------------------------------------------------------
def build_report(df: pd.DataFrame, info: dict, quality: dict, discoveries: list) -> str:
    lines = []
    lines.append("DATADETECTIVE — AUTOMATED DATA INVESTIGATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    lines.append("\n1. DATASET OVERVIEW")
    lines.append(f"Rows: {info['rows']} | Columns: {info['cols']}")
    lines.append(f"Numeric columns: {', '.join(info['numeric_cols']) or 'None'}")
    lines.append(f"Categorical columns: {', '.join(info['categorical_cols']) or 'None'}")

    lines.append("\n2. DATA QUALITY")
    lines.append(f"Data Health Score: {quality['score']}/100 — {quality['verdict']}")
    lines.append(f"Missing values: {quality['missing_pct']}% of all cells")
    lines.append(f"Duplicate rows: {quality['duplicates']} ({quality['duplicate_pct']}%)")
    lines.append(f"Total outlier values (IQR method): {quality['total_outliers']}")
    if quality["high_corr_pairs"]:
        lines.append("Highly correlated feature pairs (>0.9):")
        for a, b, v in quality["high_corr_pairs"]:
            lines.append(f"  - {a} <-> {b}: {v}")
    if quality["imbalanced_cols"]:
        lines.append("Imbalanced categorical columns:")
        for c, pct in quality["imbalanced_cols"]:
            lines.append(f"  - {c}: dominant class = {pct}%")

    lines.append("\n3. UNEXPECTED DISCOVERIES")
    for d in discoveries:
        clean = d.replace("**", "")
        lines.append(f"  - {clean}")

    if "ml_results" in st.session_state:
        ml = st.session_state["ml_results"]
        lines.append("\n4. MACHINE LEARNING RESULTS")
        lines.append(f"Target column: {ml['target']}")
        lines.append(f"Problem type: {'Classification' if ml['is_classification'] else 'Regression'}")
        lines.append(f"Best model: {ml['best_name']}")
        lines.append("\nModel comparison:")
        lines.append(ml["results_df"].to_string(index=False))
        best_rows = ml["results_df"][
            ml["results_df"]["Model"] == ml["best_name"]
        ]

        if not best_rows.empty:
            lines.append("\nBest model performance metrics:")
            for metric, value in best_rows.iloc[0].items():
                if metric != "Model":
                    lines.append(f"- {metric}: {value}")
        if ml["leakage_warnings"]:
            lines.append("\nData leakage warnings:")
            for w in ml["leakage_warnings"]:
                lines.append(f"  - {w.replace('**', '')}")

  # Privacy scan summary
    lines.append("\n5. PRIVACY & SECURITY")
    privacy_results = privacy_scan(df)
    sensitive = privacy_results[
    ~privacy_results["Detection"].str.contains("Normal", case=False, na=False)
]
    lines.append(f"Columns scanned: {len(privacy_results)}")
    lines.append(f"Columns flagged as potentially sensitive: {len(sensitive)}")
    if not sensitive.empty:
        lines.append("Review these columns before sharing the dataset:")
        for column in sensitive["Column"].tolist():
            lines.append(f"  - {column}")
    else:
        lines.append("No columns were flagged by the name-based privacy scan.")

    # Anomaly detection summary
    lines.append("\n6. ANOMALY DETECTION")
    anomaly_result, anomaly_message = detect_anomalies(df, info)
    lines.append(str(anomaly_message))

if anomaly_result is not None and "_anomaly" in anomaly_result.columns:
    anomaly_count = int(anomaly_result["_anomaly"].sum())
    lines.append(f"Records flagged as unusual: {anomaly_count}")
else:
    lines.append("Anomaly count unavailable for this dataset.")

    lines.append("\n7. RECOMMENDATIONS")
    if quality["score"] < 65:
        lines.append("  - Clean the dataset (handle missing values, duplicates, outliers) before modeling.")
    if quality["imbalanced_cols"]:
        lines.append("  - Consider resampling or class-weighting for imbalanced target classes.")
    if quality["high_corr_pairs"]:
        lines.append("  - Review highly correlated features for redundancy.")
    lines.append("  - Validate any 'unexpected discoveries' against domain knowledge before acting on them.")

    lines.append("\n" + "=" * 60)
    lines.append("End of automated report. This report is model-generated and should")
    lines.append("be reviewed by a human analyst before use in decision-making.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# ---- Professional application shell ----
# ----------------------------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div class="brand">
        <div class="brand-name">🔎 Data<span>Detective</span></div>
        <div class="brand-tagline">Autonomous Data Investigation & ML Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">DATA WORKSPACE</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a real CSV or Excel dataset for investigation."
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.session_state.df = df
            st.session_state.dataset_name = uploaded.name
            st.session_state.report_log = []
            st.session_state.pop("ml_results", None)

        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown('<div class="sidebar-label">CURRENT DATASET</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background:#102a4b;border:1px solid #21466f;
                        border-radius:10px;padding:12px;">
                <div style="font-size:12px;font-weight:700;">
                    {st.session_state.get("dataset_name", "Dataset")}
                </div>
                <div style="font-size:10px;color:#9db2cb;margin-top:5px;">
                    {st.session_state.df.shape[0]:,} records ·
                    {st.session_state.df.shape[1]} attributes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("↻  Reset workspace", use_container_width=True):
            st.session_state.df = None
            st.session_state.dataset_name = None
            st.session_state.report_log = []
            st.session_state.pop("ml_results", None)
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-label">PLATFORM</div>', unsafe_allow_html=True)
    st.caption("Data Visualization")
    st.caption("Machine Learning")
    st.caption("Privacy & Security")
    st.caption("Automated Investigation")

    st.markdown(
        """
        <div style="margin-top:28px;color:#7890ad !important;
                    font-size:10px;line-height:1.6;">
            Runs locally in your Streamlit session.<br>
            Review generated findings before making decisions.
        </div>
        """,
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------
# ---- Main ----
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="header-title">Data Investigation Workspace</div>
            <div class="header-subtitle">
                Turn raw datasets into trusted, explainable insights.
            </div>
        </div>
        <div class="secure-pill">● LOCAL & SECURE</div>
    </div>
    """,
    unsafe_allow_html=True
)

if st.session_state.df is None:

    st.markdown(
        """
        <div class="welcome">
            <h1>Find what your data is hiding.</h1>
            <p>
                DataDetective automatically profiles your dataset, checks data
                quality and privacy risks, discovers unexpected patterns,
                detects anomalies, compares machine-learning models and builds
                an investigation report.
            </p>

            <div class="workflow">
                <div class="workflow-item">
                    <div class="workflow-number">01</div>
                    <div class="workflow-name">Understand</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">02</div>
                    <div class="workflow-name">Investigate</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">03</div>
                    <div class="workflow-name">Model</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">04</div>
                    <div class="workflow-name">Report</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Start a new investigation</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-caption">Upload a CSV or Excel file using the workspace panel.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Dataset profiling</div>
            <div class="kpi-value">01</div>
            <div class="kpi-note">Structure & target discovery</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Detective mode</div>
            <div class="kpi-value">02</div>
            <div class="kpi-note">Patterns & anomalies</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">ML intelligence</div>
            <div class="kpi-value">03</div>
            <div class="kpi-note">Models & explanations</div>
        </div>
        """, unsafe_allow_html=True)

else:

    df = st.session_state.df
    info = understand_dataset(df)

    # Dataset context bar
    st.markdown(
        f"""
        <div style="background:#fff;border:1px solid #e5eaf1;border-radius:11px;
                    padding:11px 15px;margin-bottom:16px;">
            <span style="font-size:11px;color:#7b8798;">ACTIVE DATASET</span>
            <span style="font-size:13px;font-weight:700;color:#10233f;margin-left:10px;">
                {st.session_state.get("dataset_name", "Uploaded dataset")}
            </span>
            <span style="font-size:11px;color:#7b8798;margin-left:12px;">
                {info['rows']:,} rows · {info['cols']} columns
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "Overview",
        "Data Health",
        "Privacy",
        "EDA",
        "Detective Mode",
        "Model Lab",
        "Spark",
        "Cloud",
        "Deep Learning",
        "Reports"
    ])

    # ---- Overview ----
    with tabs[0]:

        quality_preview = data_quality_report(df)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Records</div>
                    <div class="kpi-value">{info['rows']:,}</div>
                    <div class="kpi-note">Rows detected</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Attributes</div>
                    <div class="kpi-value">{info['cols']}</div>
                    <div class="kpi-note">Columns detected</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Data health</div>
                    <div class="kpi-value">{quality_preview['score']}/100</div>
                    <div class="kpi-note">Automated quality score</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Target candidates</div>
                    <div class="kpi-value">{len(info['target_candidates'])}</div>
                    <div class="kpi-note">Potential prediction targets</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-title">Dataset preview</div>',
            unsafe_allow_html=True
        )
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown(
            '<div class="section-title">Column intelligence</div>',
            unsafe_allow_html=True
        )

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing": df.isna().sum().values,
            "Unique Values": [df[c].nunique() for c in df.columns]
        })

        st.dataframe(dtype_df, use_container_width=True)

        st.markdown(
            '<div class="section-title">Detected structure</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3 = st.columns(3)

        with s1:
            st.info(f"**Numeric**\n\n{len(info['numeric_cols'])} columns")

        with s2:
            st.info(f"**Categorical**\n\n{len(info['categorical_cols'])} columns")

        with s3:
            st.info(f"**Date / Time**\n\n{len(info['datetime_cols'])} columns")

    # ---- Data Health ----
    with tabs[1]:

        quality = data_quality_report(df)
        st.session_state["quality"] = quality

        st.markdown(
            '<div class="section-title">Data Health Assessment</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">A single score summarizing common data-quality risks.</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns([1, 2])

        with left:
            st.markdown(
                f"""
                <div class="health-panel">
                    <div style="font-size:11px;color:#7a8799;font-weight:700;">
                        DATA HEALTH SCORE
                    </div>
                    <div class="big-score" style="color:{quality['color']};">
                        {quality['score']}/100
                    </div>
                    <div style="font-size:12px;color:#526174;">
                        {quality['verdict']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=quality["score"],
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": quality["color"]},
                        "steps": [
                            {"range": [0, 45], "color": "#f4dddd"},
                            {"range": [45, 65], "color": "#f6ead8"},
                            {"range": [65, 85], "color": "#f3efd9"},
                            {"range": [85, 100], "color": "#dff1e7"},
                        ],
                    },
                    number={"font": {"size": 46}},
                    domain={"x": [0.1, 0.9], "y": [0, 1]},
                    title={"text": "Overall Data Health"}
                )
            )
            fig.update_layout(
                height=290,
                margin=dict(l=40, r=40, t=60, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="section-title">Quality indicators</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Missing cells", f"{quality['missing_pct']}%")
        m2.metric("Duplicate rows", f"{quality['duplicates']}")
        m3.metric("Outlier values", quality["total_outliers"])
        m4.metric("Imbalanced columns", len(quality["imbalanced_cols"]))

        if len(quality["missing"]) > 0:
            st.markdown(
                '<div class="section-title">Missing values</div>',
                unsafe_allow_html=True
            )
            st.bar_chart(quality["missing"])

        if quality["outlier_counts"]:
            st.markdown(
                '<div class="section-title">Outlier detection</div>',
                unsafe_allow_html=True
            )
            st.bar_chart(pd.Series(quality["outlier_counts"]))

        if quality["high_corr_pairs"]:
            st.markdown(
                '<div class="section-title">Potential feature redundancy</div>',
                unsafe_allow_html=True
            )
            st.table(
                pd.DataFrame(
                    quality["high_corr_pairs"],
                    columns=["Feature A", "Feature B", "Correlation"]
                )
            )

        if quality["imbalanced_cols"]:
            st.markdown(
                '<div class="section-title">Class imbalance signals</div>',
                unsafe_allow_html=True
            )
            st.table(
                pd.DataFrame(
                    quality["imbalanced_cols"],
                    columns=["Column", "Dominant class %"]
                )
            )

    # ---- Privacy ----
    with tabs[2]:

        st.markdown(
            '<div class="section-title">Privacy Scan</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">Flags columns whose names resemble common sensitive-data fields.</div>',
            unsafe_allow_html=True
        )

        scan = privacy_scan(df)
        st.dataframe(scan, use_container_width=True)

        sensitive_cols = scan[
            scan["Detection"].str.contains("🔴|🟡")
        ]["Column"].tolist()

        if sensitive_cols:
            st.warning(
                f"{len(sensitive_cols)} column(s) may contain sensitive information."
            )

            mask_cols = st.multiselect(
                "Mask selected columns in the preview",
                sensitive_cols
            )

            if mask_cols:
                masked = df.copy()

                for c in mask_cols:
                    masked[c] = "•••••"

                st.dataframe(
                    masked.head(20),
                    use_container_width=True
                )
        else:
            st.success("No obviously sensitive columns detected by the current name-based scan.")
        ext.render_security(df)

    # ---- EDA ----
    with tabs[3]:

        st.markdown(
            '<div class="section-title">Exploratory Data Analysis</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">Explore distributions, categories, relationships and correlations.</div>',
            unsafe_allow_html=True
        )

        render_eda(df, info)

    # ---- Detective Mode ----
    with tabs[4]:

        st.markdown(
            '<div class="section-title">Data Detective Mode</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">Automated investigation for patterns that may not be obvious from basic EDA.</div>',
            unsafe_allow_html=True
        )

        discoveries = discover_patterns(df, info)
        st.session_state["discoveries"] = discoveries

        st.markdown("#### Unexpected discoveries")

        for d in discoveries:
            st.markdown(
                f"<div class='discovery-box'>🔎 {d}</div>",
                unsafe_allow_html=True
            )

        st.markdown("#### Anomaly detection")

        st.caption(
            "Isolation Forest is applied to numeric columns to identify unusual records."
        )

        result, msg = detect_anomalies(df, info)

        st.info(msg)

        if result is not None:

            anomalies = (
                result[result["_anomaly"]]
                .drop(columns=["_anomaly"])
                .sort_values("_anomaly_score")
                .head(20)
            )

            st.markdown(
                "<div class='anomaly-box'><b>Records requiring investigation</b><br>"
                "These are the most unusual observations detected by the model.</div>",
                unsafe_allow_html=True
            )

            st.dataframe(
                anomalies,
                use_container_width=True
            )

    # ---- Model Lab ----
    with tabs[5]:

        st.markdown(
            '<div class="section-title">Model Lab</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">Compare candidate models and inspect why a model performed well.</div>',
            unsafe_allow_html=True
        )

        run_ml_lab(df, info)

    # ---- Reports ----
    with tabs[6]:
        ext.render_spark(df)

    with tabs[7]:
        ext.render_cloud(df)

    with tabs[8]:
        ext.render_dl(df, info, prep_features)

    with tabs[9]:

        st.markdown(
            '<div class="section-title">Investigation Report</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">A consolidated summary of dataset quality, discoveries and machine-learning results.</div>',
            unsafe_allow_html=True
        )

        quality = st.session_state.get("quality") or data_quality_report(df)
        discoveries = (
            st.session_state.get("discoveries")
            or discover_patterns(df, info)
        )

        report_text = build_report(
            df,
            info,
            quality,
            discoveries
        )

        st.markdown("### 📄 Investigation Report Preview")

with st.container(border=True):
    st.markdown("#### 🔎 Automated Data Investigation Report")
    st.caption(
        f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}"
    )

    report_sections = report_text.split("\n")
    for line in report_sections:
        cleaned_line = line.strip()

        if not cleaned_line:
            continue
        elif cleaned_line.startswith("===="):
            st.divider()
        elif cleaned_line.split(".", 1)[0].isdigit() and "." in cleaned_line:
    st.markdown(f"### {cleaned_line}")
        elif cleaned_line.startswith("-"):
            st.markdown(cleaned_line)
        else:
            st.write(cleaned_line)
        )

        # ---- Download Investigation Report ----
st.markdown("### 📥 Download Investigation Report")
st.caption("Choose your preferred format.")

report_format = st.selectbox(
    "Report format",
    ["PDF (.pdf)", "Word (.docx)", "Text (.txt)"],
    key="investigation_report_format"
)

if report_format == "Text (.txt)":
    st.download_button(
        label="⬇️ Download TXT Report",
        data=report_text,
        file_name="datadetective_investigation_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

elif report_format == "Word (.docx)":
    from docx import Document
    from io import BytesIO

    doc = Document()
    doc.add_heading("Data Detective", 0)
    doc.add_heading("Automated Data Investigation Report", level=1)
    doc.add_paragraph(
        f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}"
    )

    for line in report_text.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        elif cleaned_line.startswith("==="):
            doc.add_paragraph(cleaned_line.replace("=", "").strip())
        elif cleaned_line[:1].isdigit() and ". " in cleaned_line:
            doc.add_heading(cleaned_line, level=2)
        elif cleaned_line.startswith("- "):
            doc.add_paragraph(cleaned_line[2:], style="List Bullet")
        else:
            doc.add_paragraph(cleaned_line)

    word_buffer = BytesIO()
    doc.save(word_buffer)
    word_buffer.seek(0)

    st.download_button(
        label="⬇️ Download Word Report",
        data=word_buffer.getvalue(),
        file_name="datadetective_investigation_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )

elif report_format == "PDF (.pdf)":
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    )
    from xml.sax.saxutils import escape
    from io import BytesIO

    pdf_buffer = BytesIO()
    pdf_doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Data Detective Investigation Report",
        author="Data Detective",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4338CA"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#4338CA"),
        spaceBefore=12,
        spaceAfter=6,
    ))

    story = [
        Paragraph("DATA DETECTIVE", styles["ReportTitle"]),
        Paragraph("Automated Data Investigation Report", styles["Heading2"]),
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Normal"],
        ),
        Spacer(1, 12),
        HRFlowable(width="100%", color=colors.HexColor("#C7D2FE")),
        Spacer(1, 12),
    ]

    for line in report_text.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            story.append(Spacer(1, 5))
        elif cleaned_line.startswith("==="):
            story.append(HRFlowable(
                width="100%", color=colors.HexColor("#C7D2FE")
            ))
            story.append(Spacer(1, 6))
        elif cleaned_line[:1].isdigit() and ". " in cleaned_line:
            story.append(Paragraph(escape(cleaned_line), styles["SectionHeading"]))
        elif cleaned_line.startswith("- "):
            story.append(Paragraph(
                "&#8226; " + escape(cleaned_line[2:]),
                styles["BodyText"],
            ))
        else:
            story.append(Paragraph(escape(cleaned_line), styles["BodyText"]))

    pdf_doc.build(story)
    pdf_buffer.seek(0)

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buffer.getvalue(),
        file_name="datadetective_investigation_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
st.markdown(
    """
    <div class="app-footer">
        DataDetective · Autonomous Data Investigation Platform ·
        Generated insights should be reviewed by a human analyst.
    </div>
    """,
    unsafe_allow_html=True
)
