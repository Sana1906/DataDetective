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
    st.markdown("## 🧠 Deep Learning Lab")
    st.caption(
        "Train a neural network, compare it with Random Forest, "
        "and explore how the model learns from your dataset."
    )

    target_candidates = info.get("target_candidates", [])
    if not target_candidates:
        st.warning("No suitable target columns were detected in this dataset.")
        return

    # Dataset overview
    numeric_count = len(df.select_dtypes(include="number").columns)
    overview = st.columns(4)

    overview[0].metric("Dataset rows", f"{len(df):,}")
    overview[1].metric("Total columns", f"{len(df.columns):,}")
    overview[2].metric("Numeric columns", f"{numeric_count:,}")
    overview[3].metric("Available targets", f"{len(target_candidates):,}")

    st.divider()

    # Model configuration
    st.markdown("### ⚙️ Model configuration")
    left, right = st.columns([1, 1])

    with left:
        target = st.selectbox(
            "🎯 Target column",
            target_candidates,
            key="dl_target",
        )

        layers = st.selectbox(
            "🧩 Neural network architecture",
            ["(64,)", "(64, 32)", "(128, 64, 32)"],
            index=1,
            key="dl_layers",
        )

    with right:
        epochs = st.slider(
            "🔁 Training epochs",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            key="dl_epochs",
        )

        feats = [c for c in df.columns if c != target][:12]
        st.info(
            f"**Target:** `{target}`\n\n"
            f"**Input features:** {len(feats)}\n\n"
            f"**Hidden layers:** {layers}\n\n"
            f"**Maximum iterations:** {epochs}"
        )

    st.caption(
        "The target is what the model predicts. Input features are the "
        "columns used to make that prediction."
    )

    train_clicked = st.button(
        "🚀 Train and compare models",
        type="primary",
        use_container_width=True,
        key="dl_train_button",
    )

    if train_clicked:
        if not feats:
            st.error("The dataset needs at least one input feature.")
            return

        try:
            X, y, is_clf, _ = prep_features(df, target, feats)

            if len(X) < 50:
                st.error("At least 50 valid rows are required for training.")
                return

            if len(set(y)) < 2 and is_clf:
                st.error("The classification target needs at least two classes.")
                return

            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            sc = StandardScaler().fit(Xtr)
            Xtr_scaled = sc.transform(Xtr)
            Xte_scaled = sc.transform(Xte)

            cls = MLPClassifier if is_clf else MLPRegressor
            sizes = {
                "(64,)": (64,),
                "(64, 32)": (64, 32),
                "(128, 64, 32)": (128, 64, 32),
            }[layers]

            net = cls(
                hidden_layer_sizes=sizes,
                max_iter=epochs,
                random_state=42,
            )

            rf_cls = RandomForestClassifier if is_clf else RandomForestRegressor
            rf = rf_cls(n_estimators=150, random_state=42)

            with st.spinner("Training neural network and Random Forest..."):
                net.fit(Xtr_scaled, ytr)
                pred = net.predict(Xte_scaled)
                rf.fit(Xtr, ytr)
                rf_pred = rf.predict(Xte)

            score = (
                accuracy_score(yte, pred)
                if is_clf else r2_score(yte, pred)
            )
            rf_score = (
                accuracy_score(yte, rf_pred)
                if is_clf else r2_score(yte, rf_pred)
            )

            st.session_state["dl_results"] = {
                "target": target,
                "is_clf": is_clf,
                "score": score,
                "rf_score": rf_score,
                "loss_curve": list(getattr(net, "loss_curve_", [])),
                "y_test": list(yte),
                "pred": list(pred),
                "rf_pred": list(rf_pred),
                "features": feats,
                "layers": layers,
                "epochs": epochs,
                "rows": len(X),
            }

        except Exception as e:
            st.error(f"Training failed: {e}")
            return

    results = st.session_state.get("dl_results")

    if not results:
        st.markdown("### 📊 Your results will appear here")
        st.info(
            "Configure your model above and select **Train and compare models** "
            "to generate performance metrics and learning visualizations."
        )
        return

    if results["target"] != target:
        st.info("Your target selection changed. Train again to update the results.")
        return

    is_clf = results["is_clf"]
    score = results["score"]
    rf_score = results["rf_score"]
    label = "Accuracy" if is_clf else "R² score"

    st.divider()
    st.markdown("### 📈 Model performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("Neural Network", f"{score:.3f}")
    m2.metric("Random Forest", f"{rf_score:.3f}")
    m3.metric(
        "Difference",
        f"{score - rf_score:+.3f}",
        help="Positive means the neural network scored higher.",
    )

    if score > rf_score + 0.01:
        st.success("The neural network scored higher on this test split.")
    elif rf_score > score + 0.01:
        st.info("Random Forest scored higher on this test split.")
    else:
        st.info("The two models performed similarly on this test split.")

    st.caption(
        f"Metric: {label}. Results use a held-out test split and may vary "
        "with different datasets or random splits."
    )

    chart_col, loss_col = st.columns(2)

    with chart_col:
        comparison = pd.DataFrame({
            "Model": ["Neural Network", "Random Forest"],
            label: [score, rf_score],
        })

        fig_compare = px.bar(
            comparison,
            x="Model",
            y=label,
            color="Model",
            title="Model score comparison",
            text_auto=".3f",
        )
        fig_compare.update_layout(
            height=360,
            showlegend=False,
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with loss_col:
        losses = results["loss_curve"]

        if losses:
            loss_df = pd.DataFrame({
                "Epoch": range(1, len(losses) + 1),
                "Training loss": losses,
            })
            fig_loss = px.line(
                loss_df,
                x="Epoch",
                y="Training loss",
                title="Neural network learning curve",
            )
            fig_loss.update_layout(
                height=360,
                margin=dict(t=60, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("A training loss curve is not available for this model.")

    # Detailed results
    with st.expander("🔍 Explore detailed results"):
        if is_clf:
            st.markdown("#### Classification report")
            st.code(
                classification_report(
                    results["y_test"],
                    results["pred"],
                    zero_division=0,
                )
            )

            st.markdown("#### Neural network confusion matrix")
            cm = confusion_matrix(results["y_test"], results["pred"])
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Actual vs predicted classes",
                labels={
                    "x": "Predicted class",
                    "y": "Actual class",
                    "color": "Count",
                },
            )
            st.plotly_chart(cm_fig, use_container_width=True)
        else:
            actual = np.asarray(results["y_test"], dtype=float)
            predicted = np.asarray(results["pred"], dtype=float)

            residual_df = pd.DataFrame({
                "Actual": actual,
                "Predicted": predicted,
                "Residual": actual - predicted,
            })

            st.markdown("#### Actual vs predicted")
            actual_fig = px.scatter(
                residual_df,
                x="Actual",
                y="Predicted",
                title="Actual vs predicted values",
            )
            st.plotly_chart(actual_fig, use_container_width=True)

            st.markdown("#### Prediction errors")
            residual_fig = px.scatter(
                residual_df,
                x="Predicted",
                y="Residual",
                title="Residual plot",
            )
            residual_fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(residual_fig, use_container_width=True)

    st.caption(
        "These scores are comparisons on the same held-out test split, "
        "not proof that one model will perform better on future data."
    )


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
