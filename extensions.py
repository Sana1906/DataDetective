"""Sem V subject modules for DataDetective: Spark, Cloud, Deep Learning, Security."""
import hashlib
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
from io import StringIO
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
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
        "Train a neural network, evaluate its performance, compare it "
        "with Random Forest, and investigate prediction quality."
    )

    target_candidates = info.get("target_candidates", [])
    if not target_candidates:
        st.warning("No suitable target columns were detected.")
        return

    numeric_count = len(df.select_dtypes(include="number").columns)

    st.markdown("### 📊 Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", f"{len(df):,}")
    c2.metric("Total columns", f"{len(df.columns):,}")
    c3.metric("Numeric columns", f"{numeric_count:,}")
    c4.metric("Available targets", len(target_candidates))

    st.divider()
    st.markdown("### ⚙️ Model configuration")

    left, right = st.columns(2)

    with left:
        target = st.selectbox(
            "🎯 Target column",
            target_candidates,
            key="dl_target",
        )
        layers = st.selectbox(
            "🧩 Hidden-layer architecture",
            ["(64,)", "(64, 32)", "(128, 64, 32)"],
            index=1,
            key="dl_layers",
        )

    with right:
        epochs = st.slider(
            "🔁 Maximum training iterations",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            key="dl_epochs",
        )
        test_fraction = st.select_slider(
            "Test-set proportion",
            options=[0.15, 0.20, 0.25, 0.30],
            value=0.25,
            format_func=lambda x: f"{int(x * 100)}%",
            key="dl_test_fraction",
        )

    feats = [c for c in df.columns if c != target][:12]

    st.info(
        f"**Target:** `{target}`  ·  "
        f"**Input features:** {len(feats)}  ·  "
        f"**Hidden layers:** {layers}  ·  "
        f"**Test set:** {test_fraction:.0%}"
    )

    st.caption(
        "The target is the value being predicted. The first 12 available "
        "non-target columns are used as input features."
    )

    if st.button(
        "🚀 Train and evaluate models",
        type="primary",
        use_container_width=True,
        key="dl_train_button",
    ):
        st.session_state.pop("dl_results", None)

        if not feats:
            st.error("The dataset needs at least one input feature.")
            return

        try:
            X, y, is_clf, _ = prep_features(df, target, feats)

            if len(X) < 50:
                st.error("At least 50 valid rows are required for training.")
                return

            y_array = np.asarray(y).ravel()

            if pd.isna(y_array).any():
                st.error("The target still contains missing values after preprocessing.")
                return

            if is_clf and len(np.unique(y_array)) < 2:
                st.error("Classification requires at least two target classes.")
                return

            if not is_clf and not np.issubdtype(y_array.dtype, np.number):
                st.error(
                    "The regression target must be numeric. "
                    "Choose a suitable target or review preprocessing."
                )
                return

            # Preserve class proportions when possible.
            stratify_y = None
            if is_clf:
                _, counts = np.unique(y_array, return_counts=True)
                if len(counts) > 1 and counts.min() >= 2:
                    stratify_y = y_array

            Xtr, Xte, ytr, yte = train_test_split(
                X,
                y_array,
                test_size=test_fraction,
                random_state=42,
                stratify=stratify_y,
            )

            sc = StandardScaler()
            Xtr_scaled = sc.fit_transform(Xtr)
            Xte_scaled = sc.transform(Xte)

            sizes = {
                "(64,)": (64,),
                "(64, 32)": (64, 32),
                "(128, 64, 32)": (128, 64, 32),
            }[layers]

            net_cls = MLPClassifier if is_clf else MLPRegressor
            rf_cls = RandomForestClassifier if is_clf else RandomForestRegressor

            net = net_cls(
                hidden_layer_sizes=sizes,
                max_iter=epochs,
                random_state=42,
                early_stopping=False,
            )
            rf = rf_cls(n_estimators=150, random_state=42)

            convergence_messages = []
            start_time = time.perf_counter()

            with st.spinner("Training both models..."):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)

                    net.fit(Xtr_scaled, ytr)
                    net_pred = net.predict(Xte_scaled)

                    rf.fit(Xtr, ytr)
                    rf_pred = rf.predict(Xte)

                convergence_messages = [
                    str(w.message)
                    for w in caught
                    if issubclass(w.category, ConvergenceWarning)
                ]

            elapsed = time.perf_counter() - start_time

            # Calculate task-appropriate metrics.
            if is_clf:
                metrics = {
                    "Accuracy": accuracy_score(yte, net_pred),
                    "Precision (weighted)": precision_score(
                        yte, net_pred, average="weighted", zero_division=0
                    ),
                    "Recall (weighted)": recall_score(
                        yte, net_pred, average="weighted", zero_division=0
                    ),
                    "F1-score (weighted)": f1_score(
                        yte, net_pred, average="weighted", zero_division=0
                    ),
                }
                rf_score = accuracy_score(yte, rf_pred)
                primary_score = metrics["Accuracy"]
                primary_label = "Accuracy"
            else:
                actual = np.asarray(yte, dtype=float)
                predicted = np.asarray(net_pred, dtype=float)

                metrics = {
                    "MAE": mean_absolute_error(actual, predicted),
                    "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                    "R²": r2_score(actual, predicted),
                }
                rf_score = r2_score(yte, rf_pred)
                primary_score = metrics["R²"]
                primary_label = "R² score"

            # Permutation importance: measures the effect of shuffling
            # each input feature on held-out model performance.
            importance = None
            importance_error = None

            try:
                importance_result = permutation_importance(
                    net,
                    Xte_scaled,
                    yte,
                    n_repeats=3,
                    random_state=42,
                    scoring="accuracy" if is_clf else "r2",
                    n_jobs=1,
                )

                feature_names = (
                    list(X.columns)
                    if hasattr(X, "columns")
                    else feats[:X.shape[1]]
                )

                if len(feature_names) == len(importance_result.importances_mean):
                    importance = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importance_result.importances_mean,
                    }).sort_values("Importance", ascending=False)

            except Exception as exc:
                importance_error = str(exc)

            st.session_state["dl_results"] = {
                "target": target,
                "is_clf": is_clf,
                "metrics": metrics,
                "score": primary_score,
                "rf_score": rf_score,
                "primary_label": primary_label,
                "loss_curve": list(getattr(net, "loss_curve_", [])),
                "y_test": np.asarray(yte).tolist(),
                "pred": np.asarray(net_pred).tolist(),
                "rf_pred": np.asarray(rf_pred).tolist(),
                "features": feats,
                "layers": layers,
                "epochs": epochs,
                "rows": len(X),
                "train_rows": len(Xtr),
                "test_rows": len(Xte),
                "training_seconds": elapsed,
                "iterations": getattr(net, "n_iter_", None),
                "converged": len(convergence_messages) == 0,
                "convergence_messages": convergence_messages,
                "importance": (
                    importance.to_dict("records")
                    if importance is not None else None
                ),
                "importance_error": importance_error,
                "test_fraction": test_fraction,
            }

        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

    results = st.session_state.get("dl_results")

    if not results:
        st.markdown("### 📈 Your model results")
        st.info(
            "Choose your target and settings above, then train the models "
            "to see evaluation metrics, learning curves, and model insights."
        )
        return

    if results["target"] != target:
        st.info("The selected target differs from the last run. Train again to refresh results.")
        return

    is_clf = results["is_clf"]
    score = results["score"]
    rf_score = results["rf_score"]

    st.divider()
    st.markdown("### 📈 Evaluation summary")

    a, b, c, d = st.columns(4)
    a.metric(results["primary_label"], f"{score:.3f}")
    b.metric("Random Forest score", f"{rf_score:.3f}")
    c.metric("Training time", f"{results['training_seconds']:.2f} s")
    d.metric("Training iterations", results["iterations"] or "N/A")

    if results["converged"]:
        st.success(
            "No neural-network convergence warning was recorded. "
            "This does not guarantee that the model generalizes well."
        )
    else:
        st.warning(
            "The neural network reached its iteration limit or reported "
            "a convergence warning. Consider adjusting settings and retraining."
        )
        for message in results["convergence_messages"]:
            st.caption(message)

    if score > rf_score + 0.01:
        st.info("The neural network scored higher on the selected comparison metric.")
    elif rf_score > score + 0.01:
        st.info("Random Forest scored higher on the selected comparison metric.")
    else:
        st.info("The models scored similarly on the selected comparison metric.")

    st.caption(
        "For classification, the comparison uses accuracy. For regression, "
        "it uses R². Higher is better for these two comparison metrics."
    )

    st.markdown("### 🧾 Neural network metrics")
    metric_cols = st.columns(min(4, len(results["metrics"])))
    for i, (name, value) in enumerate(results["metrics"].items()):
        metric_cols[i % len(metric_cols)].metric(name, f"{value:.4f}")

    chart_col, loss_col = st.columns(2)

    with chart_col:
        compare_df = pd.DataFrame({
            "Model": ["Neural Network", "Random Forest"],
            "Score": [score, rf_score],
        })
        compare_fig = px.bar(
            compare_df,
            x="Model",
            y="Score",
            color="Model",
            text_auto=".3f",
            title=f"Model comparison — {results['primary_label']}",
        )
        compare_fig.update_layout(
            height=360,
            showlegend=False,
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(compare_fig, use_container_width=True)

    with loss_col:
        losses = results["loss_curve"]
        if losses:
            loss_df = pd.DataFrame({
                "Iteration": range(1, len(losses) + 1),
                "Training loss": losses,
            })
            loss_fig = px.line(
                loss_df,
                x="Iteration",
                y="Training loss",
                title="Neural network training loss",
            )
            loss_fig.update_layout(
                height=360,
                margin=dict(t=60, b=20, l=20, r=20),
            )
            st.plotly_chart(loss_fig, use_container_width=True)
        else:
            st.info("No training loss curve is available.")

    with st.expander("🔎 Feature importance"):
        if results["importance"]:
            importance_df = pd.DataFrame(results["importance"])
            importance_fig = px.bar(
                importance_df.head(12).sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Permutation importance on held-out data",
            )
            importance_fig.update_layout(height=400)
            st.plotly_chart(importance_fig, use_container_width=True)
            st.caption(
                "Permutation importance measures how the chosen score changes "
                "when a feature is shuffled. Negative values can occur; this "
                "is not a causal explanation."
            )
        else:
            st.info(
                "Feature importance could not be calculated for this run. "
                "Check that preprocessing preserves the input feature mapping."
            )
            if results.get("importance_error"):
                st.caption(results["importance_error"])

    with st.expander("🧪 Detailed evaluation"):
        if is_clf:
            st.markdown("#### Classification report")
            st.code(
                classification_report(
                    results["y_test"],
                    results["pred"],
                    zero_division=0,
                )
            )

            st.markdown("#### Confusion matrix")
            cm = confusion_matrix(
                results["y_test"],
                results["pred"],
            )
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

            actual_fig = px.scatter(
                residual_df,
                x="Actual",
                y="Predicted",
                title="Actual vs predicted values",
            )
            st.plotly_chart(actual_fig, use_container_width=True)

            residual_fig = px.scatter(
                residual_df,
                x="Predicted",
                y="Residual",
                title="Prediction residuals",
            )
            residual_fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(residual_fig, use_container_width=True)

    # Downloadable plain-text evaluation report.
    report_lines = [
        "DATA DETECTIVE — DEEP LEARNING EVALUATION REPORT",
        "=" * 48,
        f"Target: {results['target']}",
        f"Task: {'Classification' if is_clf else 'Regression'}",
        f"Input features: {', '.join(results['features'])}",
        f"Hidden layers: {results['layers']}",
        f"Maximum iterations configured: {results['epochs']}",
        f"Rows used: {results['rows']}",
        f"Training rows: {results['train_rows']}",
        f"Test rows: {results['test_rows']}",
        f"Training time: {results['training_seconds']:.2f} seconds",
        f"Iterations completed: {results['iterations']}",
        f"Convergence warning recorded: {not results['converged']}",
        "",
        "NEURAL NETWORK METRICS",
    ]

    report_lines.extend(
        f"{name}: {value:.5f}"
        for name, value in results["metrics"].items()
    )
    report_lines.extend([
        "",
        f"Random Forest {results['primary_label']}: {rf_score:.5f}",
        "",
        "Limitations: These results use one held-out test split. "
        "They do not guarantee future performance or prove causality.",
    ])

    st.download_button(
        "📥 Download evaluation report (.txt)",
        data="\n".join(report_lines),
        file_name="datadetective_deep_learning_report.txt",
        mime="text/plain",
        use_container_width=True,
        key="dl_download_report",
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
