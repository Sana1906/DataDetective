"""Sem V subject modules for DataDetective: Spark, Cloud, Deep Learning, Security."""
import hashlib
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

BUCKET_DIR = "cloud_bucket"


# ---------------------------------------------------------------- Spark
def _prepare_spark_env():
    """Avoid clashes with other Spark/Java installs on the machine (Windows)."""
    import glob
    # A separate Spark install (e.g. C:\\spark\\spark-3.5.9) conflicts with pip's PySpark.
    os.environ.pop("SPARK_HOME", None)
    for pat in (r"C:\Program Files\Eclipse Adoptium\jdk-17*",
                r"C:\Program Files\Java\jdk-17*",
                r"C:\Program Files\Microsoft\jdk-17*"):
        hits = sorted(glob.glob(pat))
        if hits:
            os.environ["JAVA_HOME"] = hits[-1]
            os.environ["PATH"] = os.path.join(hits[-1], "bin") + os.pathsep + os.environ.get("PATH", "")
            return


def spark_benchmark(df: pd.DataFrame, mult: int = 1):
    """Time the same aggregation in pandas and PySpark. Returns (rows, t_pandas, t_spark, error)."""
    import shutil
    _prepare_spark_env()
    num = df.select_dtypes(include=np.number).columns.tolist()
    big = pd.concat([df[num]] * mult, ignore_index=True) if mult > 1 else df[num]
    t = time.perf_counter()
    big.mean(), big.std()
    t_pd = time.perf_counter() - t
    if shutil.which("java") is None:
        return len(big), t_pd, None, "Java was not found on this computer."
    try:
        from pyspark.sql import SparkSession, functions as F

        spark = (SparkSession.builder.master("local[*]").appName("DataDetective")
                 .config("spark.ui.enabled", "false")
                 .config("spark.sql.shuffle.partitions", "4").getOrCreate())
        sdf = spark.createDataFrame(big.fillna(0).astype(float).values.tolist(), schema=num).cache()
        sdf.count()  # load data into Spark first, so only the aggregation is timed
        t = time.perf_counter()
        sdf.select([F.mean(c).alias(c) for c in num] +
                   [F.stddev(c).alias(c + "_sd") for c in num]).collect()
        return len(big), t_pd, time.perf_counter() - t, None
    except Exception as ex:
        return len(big), t_pd, None, "Spark could not start: " + str(ex).splitlines()[0][:110]


def render_spark(df):
    st.subheader("Distributed Processing with Apache Spark")
    st.write("Small datasets run on pandas. Large datasets are routed to Spark, "
             "which splits the work across CPU cores or a cluster.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Engine selected", "Spark" if len(df) > 1_000_000 else "pandas")
    c3.metric("Spark threshold", "1M rows")
    mult = st.select_slider("Test size (copies of your dataset)", [1, 20, 100], value=20,
                            help="Repeats your rows to simulate a bigger dataset.")
    if st.button("Run pandas vs Spark benchmark"):
        with st.spinner("Running (the first Spark run takes about 20 seconds to start)..."):
            rows, t_pd, t_sp, err = spark_benchmark(df, mult)
        st.write(f"Benchmark ran on **{rows:,} rows**.")
        if t_sp is None:
            st.warning(f"{err} Spark needs Java 17: install it from adoptium.net, restart "
                       "VS Code, and check that `java -version` works in the terminal.")
            st.metric("pandas time", f"{t_pd:.4f} s")
        else:
            fig = go.Figure(go.Bar(x=["pandas", "Spark"], y=[t_pd, t_sp],
                                   marker_color=["#2f6fed", "#f59e0b"]))
            fig.update_layout(title="Aggregation time (seconds, lower is faster)", height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.info("On small data pandas is usually faster because Spark has startup overhead. "
                    "Spark wins on very large data or on a cluster. That is why this app "
                    "switches engines by dataset size.")
    st.markdown("**PySpark code (Python)**")
    st.code('''from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.master("local[*]").getOrCreate()
sdf = spark.read.csv("data.csv", header=True, inferSchema=True)
sdf.groupBy("plan").agg(F.avg("monthly_charge")).show()''', language="python")
    st.markdown("**Same job in Scala** (see `scala/DataQuality.scala`)")
    st.code('''val df = spark.read.option("header","true").option("inferSchema","true").csv("data.csv")
df.groupBy("plan").agg(avg("monthly_charge")).show()''', language="scala")


# ---------------------------------------------------------------- Cloud
def cloud_upload(df: pd.DataFrame, name: str) -> dict:
    os.makedirs(BUCKET_DIR, exist_ok=True)
    version = len([f for f in os.listdir(BUCKET_DIR) if f.startswith(name)]) + 1
    key = f"{name}_v{version}.csv"
    path = os.path.join(BUCKET_DIR, key)
    df.to_csv(path, index=False)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    meta = {"object": key, "rows": len(df), "sha256": digest,
            "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    return meta


def render_cloud(df):
    st.subheader("Cloud Data Management")
    st.write("Datasets are stored as versioned objects with checksums, the same layout "
             "used by Amazon S3, Google Cloud Storage and Azure Blob.")
    name = st.text_input("Dataset name", "customer_data")
    if st.button("Upload to bucket"):
        m = cloud_upload(df, name)
        st.success(f"Stored {m['object']} (checksum {m['sha256']})")
    if os.path.isdir(BUCKET_DIR):
        rows = []
        for f in sorted(os.listdir(BUCKET_DIR)):
            if f.endswith(".meta.json"):
                with open(os.path.join(BUCKET_DIR, f)) as fh:
                    rows.append(json.load(fh))
        if rows:
            st.markdown("**Objects in bucket**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("Currently a local bucket folder. To use a real cloud, replace "
               "`cloud_upload` with `boto3.client('s3').upload_file(...)`.")


# ---------------------------------------------------------------- Deep Learning
def render_dl(df, info, prep_features):
    st.subheader("Deep Learning (Neural Network)")
    st.write("A multi-layer neural network is trained on the selected target. "
             "The loss curve shows how well it is learning.")
    target = st.selectbox("Target column", info["target_candidates"], key="dl_target")
    feats = [c for c in df.columns if c != target][:12]
    c1, c2 = st.columns(2)
    layers = c1.selectbox("Hidden layers", ["(64,)", "(64, 32)", "(128, 64, 32)"], index=1)
    epochs = c2.slider("Training epochs", 50, 500, 200, step=50)
    if st.button("Train neural network"):
        X, y, is_clf, _ = prep_features(df, target, feats)
        if len(X) < 50:
            st.error("Need at least 50 valid rows.")
            return
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        sc = StandardScaler().fit(Xtr)
        cls = MLPClassifier if is_clf else MLPRegressor
        sizes = {"(64,)": (64,), "(64, 32)": (64, 32), "(128, 64, 32)": (128, 64, 32)}[layers]
        net = cls(hidden_layer_sizes=sizes, max_iter=epochs, random_state=42)
        with st.spinner("Training..."):
            net.fit(sc.transform(Xtr), ytr)
        pred = net.predict(sc.transform(Xte))
        score = accuracy_score(yte, pred) if is_clf else r2_score(yte, pred)
        rf = (RandomForestClassifier if is_clf else RandomForestRegressor)(
            n_estimators=150, random_state=42).fit(Xtr, ytr)
        rf_pred = rf.predict(Xte)
        rf_score = accuracy_score(yte, rf_pred) if is_clf else r2_score(yte, rf_pred)
        label = "Accuracy" if is_clf else "R2 score"
        m1, m2 = st.columns(2)
        m1.metric(f"Neural network ({label})", f"{score:.3f}")
        m2.metric(f"Random Forest ({label})", f"{rf_score:.3f}", f"{rf_score - score:+.3f} vs NN")
        if score > rf_score + 0.01:
            st.success("The neural network clearly wins here, so its extra complexity is justified.")
        elif score >= rf_score - 0.01:
            st.info("Both perform about the same. Prefer the simpler model (Random Forest).")
        else:
            st.warning("Classic ML wins on this dataset, so deep learning is not worth it here.")
        fig = go.Figure(go.Scatter(y=net.loss_curve_, mode="lines", line_color="#2f6fed"))
        fig.update_layout(title="Training loss per epoch", xaxis_title="Epoch",
                          yaxis_title="Loss", height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("If a classic model such as Random Forest scores the same or better, "
                   "prefer it: deep learning is only worth its cost when it wins clearly.")


# ---------------------------------------------------------------- Security
def render_security(df):
    st.markdown("---")
    st.subheader("Data Protection and Compliance")
    cols = df.columns.tolist()
    hash_cols = st.multiselect("Pseudonymize columns (SHA-256 hash)", cols, key="sec_hash")
    quasi = st.multiselect("Quasi-identifiers for k-anonymity check", cols, key="sec_q")
    out = df.copy()
    for c in hash_cols:
        out[c] = out[c].astype(str).map(lambda v: hashlib.sha256(v.encode()).hexdigest()[:12])
    if quasi:
        k = int(out.groupby(quasi, dropna=False).size().min())
        st.metric("k-anonymity", k)
        st.caption("k = size of the smallest group sharing the same quasi-identifier values. "
                   "Higher is safer; k = 1 means someone can be singled out.")
    if hash_cols:
        st.dataframe(out.head(10), use_container_width=True)
        st.download_button("Download protected dataset", out.to_csv(index=False),
                           "protected_dataset.csv", "text/csv")
    st.markdown("**Compliance checklist**")
    for item, ok in [("Sensitive columns identified", True),
                     ("Pseudonymization available", True),
                     ("No data sent to external servers", True),
                     ("Access logging", False)]:
        st.markdown(f"- {'Done' if ok else 'Planned'}: {item}")
