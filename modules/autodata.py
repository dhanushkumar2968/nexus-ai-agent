import sys
import os
import io

# Ensure project root is in sys.path when running module directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    r2_score, root_mean_squared_error, mean_absolute_error
)

# Optional / Policy-guarded model imports (handles Windows Smart App Control or missing Cython DLLs)
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    SKLEARN_TREE_AVAILABLE = True
except (ImportError, Exception):
    SKLEARN_TREE_AVAILABLE = False
    RandomForestClassifier = None
    RandomForestRegressor = None
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None
    DecisionTreeClassifier = None
    DecisionTreeRegressor = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    xgb = None

from core.styling import render_metric_card, render_domain_banner
from core.utils import (
    style_plotly_figure, 
    generate_sample_healthcare_data,
    generate_sample_fintech_transactions,
    generate_sample_ecommerce_data,
    generate_sample_energy_data,
    ensure_sample_assets
)

def render_autodata_module():
    """Main renderer for Universal AutoDataLab (No-Code AutoML & Auto-EDA) domain."""
    ensure_sample_assets()

    render_domain_banner(
        title="Universal AutoDataLab (BYOD)",
        subtitle="No-Code Automated Exploratory Data Analysis (Auto-EDA), Multi-Model AutoML Leaderboard, and Pipeline Export",
        domain_type="autodata",
        badge_text="Automated Machine Learning & EDA"
    )

    # Dataset Source Selector
    st.markdown("#### 📁 Select or Upload Dataset")
    col_ds1, col_ds2 = st.columns([1.2, 2])

    with col_ds1:
        data_source = st.radio(
            "Data Source:",
            ["Preloaded Benchmark Dataset", "Upload Custom CSV File"],
            index=0
        )

    df = None
    dataset_name = ""

    with col_ds2:
        if data_source == "Preloaded Benchmark Dataset":
            benchmark_choice = st.selectbox(
                "Choose Benchmark Domain:",
                [
                    "🏥 Healthcare: Clinical Cardiovascular Risk",
                    "💳 FinTech: High-Frequency Fraud Detection",
                    "🛍️ E-Commerce: RFM Customer Analytics",
                    "⚡ Smart Energy: Municipal Hourly Demand"
                ]
            )
            if "Healthcare" in benchmark_choice:
                df = generate_sample_healthcare_data()
                dataset_name = "Healthcare Cardiovascular Cohort"
            elif "FinTech" in benchmark_choice:
                df = generate_sample_fintech_transactions(n_samples=800)
                dataset_name = "FinTech Transaction Stream"
            elif "E-Commerce" in benchmark_choice:
                rfm, _ = generate_sample_ecommerce_data()
                df = rfm
                dataset_name = "E-Commerce Customer RFM"
            else:
                df = generate_sample_energy_data(n_days=30)
                dataset_name = "Smart Grid Energy Timeseries"
        else:
            uploaded_file = st.file_uploader("Upload CSV file (Max 50MB):", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    dataset_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Error loading CSV file: {e}")
            else:
                st.info("👆 Please upload a CSV file above, or switch to 'Preloaded Benchmark Dataset' to experiment immediately.")

    if df is None:
        return

    # High-level Dataset Metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Observations", f"{len(df):,}", f"{len(df.columns)} Features", "primary")
    with col2:
        render_metric_card("Numeric Features", f"{len(numeric_cols)}", f"{len(categorical_cols)} Categorical", "cyan")
    with col3:
        render_metric_card("Missing Values", f"{missing_cells:,}", "Auto-imputable", "success" if missing_cells == 0 else "warning")
    with col4:
        render_metric_card("Duplicate Rows", f"{duplicate_rows:,}", "Ready for Modeling", "success")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_eda, tab_automl, tab_inference = st.tabs([
        "📊 Automated Exploratory Data Analysis (Auto-EDA)",
        "🤖 AutoML Leaderboard & Model Comparison",
        "🔮 Real-Time Inference & Model Download"
    ])

    # ----------------------------------------------------
    # TAB 1: Auto-EDA
    # ----------------------------------------------------
    with tab_eda:
        st.markdown(f"#### Exploratory Profile: *{dataset_name}*")
        
        st.markdown("##### 📄 Data Preview & Schema")
        st.dataframe(df.head(6), use_container_width=True)

        col_p1, col_p2 = st.columns([1.4, 1])

        with col_p1:
            st.markdown("##### 📐 Summary Statistics")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe().round(2).T, use_container_width=True)
            else:
                st.info("No numeric columns found for descriptive stats.")

        with col_p2:
            st.markdown("##### 🔍 Column Types & Null Counts")
            type_df = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Non-Null Count": df.notnull().sum(),
                "Null %": (df.isnull().mean() * 100).round(1)
            })
            st.dataframe(type_df, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.markdown("##### 🔗 Pearson Correlation Matrix")
            corr = df[numeric_cols].corr().round(2)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=420)
            style_plotly_figure(fig_corr, "Pairwise Feature Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("##### 📈 Univariate Feature Inspector")
        col_sel, col_chart = st.columns([1, 2.5])
        with col_sel:
            selected_feature = st.selectbox("Select Feature to Inspect:", df.columns.tolist())
            plot_type = st.radio("Chart Type:", ["Histogram / Density", "Box Plot"], index=0)

        with col_chart:
            if selected_feature in numeric_cols:
                if plot_type == "Histogram / Density":
                    fig_dist = px.histogram(df, x=selected_feature, marginal="rug", color_discrete_sequence=["#818cf8"])
                else:
                    fig_dist = px.box(df, y=selected_feature, color_discrete_sequence=["#38bdf8"])
                fig_dist.update_layout(height=320)
                style_plotly_figure(fig_dist, f"Distribution: {selected_feature}")
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                top_cats = df[selected_feature].value_counts().head(12).reset_index()
                top_cats.columns = [selected_feature, "Count"]
                fig_bar = px.bar(top_cats, x=selected_feature, y="Count", color_discrete_sequence=["#34d399"])
                fig_bar.update_layout(height=320)
                style_plotly_figure(fig_bar, f"Category Frequencies: {selected_feature}")
                st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: AutoML Engine
    # ----------------------------------------------------
    with tab_automl:
        st.markdown("#### Automated Machine Learning Pipeline")
        st.markdown(
            "Select your prediction target. The AutoML engine automatically identifies problem formulation "
            "(Binary/Multiclass Classification vs. Continuous Regression), trains multiple benchmark algorithms, "
            "computes cross-validation performance, and ranks them in a unified leaderboard."
        )

        col_t1, col_t2, col_t3 = st.columns([1.2, 1.2, 1])

        with col_t1:
            target_col = st.selectbox("🎯 Target Variable (Label):", df.columns.tolist(), index=len(df.columns) - 1)

        # Detect task type
        target_series = df[target_col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(target_series.dtype)
        n_unique = target_series.nunique()

        if is_numeric and n_unique > 15:
            auto_task = "Regression"
        else:
            auto_task = "Classification"

        with col_t2:
            task_type = st.selectbox(
                "Task Formulation:",
                ["Auto-Detected", "Classification", "Regression"],
                index=0
            )
            if task_type == "Auto-Detected":
                active_task = auto_task
            else:
                active_task = task_type
            st.caption(f"Task Mode: **{active_task}** ({n_unique} unique target classes/values)")

        with col_t3:
            test_size = st.slider("Test Evaluation Split (%):", min_value=10, max_value=40, value=20, step=5) / 100.0

        feature_candidates = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect("Predictor Features (X):", feature_candidates, default=feature_candidates)

        if st.button("🚀 Run AutoML Multi-Model Benchmark", type="primary", use_container_width=True):
            if not selected_features:
                st.error("Please select at least one predictor feature.")
                return

            with st.spinner("Executing AutoML pipeline: Preprocessing, Feature Encoding, Cross-Model Training..."):
                # Prepare data
                df_clean = df[[target_col] + selected_features].dropna().copy()
                
                # Encode features
                X = df_clean[selected_features].copy()
                y = df_clean[target_col].copy()

                # Categorical feature encoding
                label_encoders = {}
                for col in X.columns:
                    if not pd.api.types.is_numeric_dtype(X[col].dtype):
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        label_encoders[col] = le

                if active_task == "Classification" and not pd.api.types.is_numeric_dtype(y.dtype):
                    target_le = LabelEncoder()
                    y = target_le.fit_transform(y.astype(str))
                    st.session_state["target_encoder"] = target_le
                else:
                    st.session_state["target_encoder"] = None

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                results = []
                trained_models = {}

                if not SKLEARN_TREE_AVAILABLE:
                    st.info("ℹ️ **Smart App Control Adaptive Mode Active**: Scikit-Learn tree extensions are restricted by Windows Application Control. AutoDataLab is powered by XGBoost, Linear, Ridge, and Naive Bayes engines.")

                if active_task == "Classification":
                    models = {}
                    if XGBOOST_AVAILABLE and xgb is not None:
                        models["XGBoost Classifier"] = xgb.XGBClassifier(
                            n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss"
                        )
                    if SKLEARN_TREE_AVAILABLE and RandomForestClassifier is not None:
                        models["Random Forest Classifier"] = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                        models["HistGradient Boosting"] = HistGradientBoostingClassifier(max_iter=100, random_state=42)
                        models["Decision Tree Classifier"] = DecisionTreeClassifier(max_depth=6, random_state=42)
                    models["Logistic Regression"] = LogisticRegression(max_iter=500, random_state=42)
                    models["Gaussian Naive Bayes"] = GaussianNB()

                    for name, clf in models.items():
                        clf.fit(X_train, y_train)
                        preds = clf.predict(X_test)
                        acc = accuracy_score(y_test, preds)
                        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                        
                        auc_val = "N/A"
                        if len(np.unique(y)) == 2:
                            try:
                                probas = clf.predict_proba(X_test)[:, 1]
                                auc_val = f"{roc_auc_score(y_test, probas):.3f}"
                            except Exception:
                                pass

                        results.append({
                            "Algorithm": name,
                            "Accuracy": round(acc, 4),
                            "Precision": round(prec, 4),
                            "Recall": round(rec, 4),
                            "F1-Score": round(f1, 4),
                            "ROC-AUC": auc_val
                        })
                        trained_models[name] = clf

                    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

                else:
                    # Regression
                    models = {}
                    if XGBOOST_AVAILABLE and xgb is not None:
                        models["XGBoost Regressor"] = xgb.XGBRegressor(
                            n_estimators=100, max_depth=6, random_state=42
                        )
                    if SKLEARN_TREE_AVAILABLE and RandomForestRegressor is not None:
                        models["Random Forest Regressor"] = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
                        models["HistGradient Regressor"] = HistGradientBoostingRegressor(max_iter=100, random_state=42)
                        models["Decision Tree Regressor"] = DecisionTreeRegressor(max_depth=6, random_state=42)
                    models["Ridge Regression"] = Ridge(random_state=42)
                    models["Linear Regression"] = LinearRegression()

                    for name, reg in models.items():
                        reg.fit(X_train, y_train)
                        preds = reg.predict(X_test)
                        r2 = r2_score(y_test, preds)
                        rmse = root_mean_squared_error(y_test, preds)
                        mae = mean_absolute_error(y_test, preds)

                        results.append({
                            "Algorithm": name,
                            "R² Score": round(r2, 4),
                            "RMSE": round(rmse, 4),
                            "MAE": round(mae, 4)
                        })
                        trained_models[name] = reg

                    results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False).reset_index(drop=True)

                st.session_state["automl_results"] = results_df
                st.session_state["trained_models"] = trained_models
                st.session_state["best_model_name"] = results_df.iloc[0]["Algorithm"]
                st.session_state["active_task"] = active_task
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.session_state["selected_features"] = selected_features
                st.session_state["label_encoders"] = label_encoders
                st.session_state["target_col"] = target_col

        # Show leaderboard if trained
        if "automl_results" in st.session_state:
            st.markdown("---")
            st.markdown("### 🏆 AutoML Benchmark Leaderboard")
            res_df = st.session_state["automl_results"]
            st.dataframe(res_df, use_container_width=True)

            best_name = st.session_state["best_model_name"]
            st.success(f"🥇 **Champion Model:** **{best_name}** selected based on validation performance benchmark.")

            col_diag1, col_diag2 = st.columns(2)

            with col_diag1:
                best_model = st.session_state["trained_models"][best_name]
                X_test = st.session_state["X_test"]
                y_test = st.session_state["y_test"]
                preds = best_model.predict(X_test)

                if st.session_state["active_task"] == "Classification":
                    cm = confusion_matrix(y_test, preds)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        labels=dict(x="Predicted Class", y="Actual True Class"),
                        color_continuous_scale="Purples"
                    )
                    fig_cm.update_layout(height=350)
                    style_plotly_figure(fig_cm, f"Confusion Matrix: {best_name}")
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    fig_res = px.scatter(
                        x=y_test,
                        y=preds,
                        labels={"x": "Actual Observed Value", "y": "Model Predicted Value"},
                        color_discrete_sequence=["#818cf8"]
                    )
                    # 45 degree perfect line
                    min_val = min(y_test.min(), preds.min())
                    max_val = max(y_test.max(), preds.max())
                    fig_res.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="#f87171", dash="dash"))
                    fig_res.update_layout(height=350)
                    style_plotly_figure(fig_res, f"Actual vs. Predicted: {best_name}")
                    st.plotly_chart(fig_res, use_container_width=True)

            with col_diag2:
                # Feature importance if tree-based model
                if hasattr(best_model, "feature_importances_"):
                    importances = best_model.feature_importances_
                    feat_df = pd.DataFrame({
                        "Feature": st.session_state["selected_features"],
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=True)

                    fig_imp = px.bar(
                        feat_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale="Blues"
                    )
                    fig_imp.update_layout(height=350)
                    style_plotly_figure(fig_imp, "Gini Feature Importance Hierarchy")
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importance plot is available for tree-based models (Random Forest, Decision Tree).")

    # ----------------------------------------------------
    # TAB 3: Model Inference & Download
    # ----------------------------------------------------
    with tab_inference:
        st.markdown("#### Real-Time Interactive Inference & Pipeline Export")
        if "best_model_name" not in st.session_state:
            st.info("💡 Train a model in the **AutoML Leaderboard** tab first to enable single-row inference and pipeline export.")
        else:
            best_name = st.session_state["best_model_name"]
            best_model = st.session_state["trained_models"][best_name]
            features = st.session_state["selected_features"]
            encoders = st.session_state["label_encoders"]

            st.markdown(f"**Loaded Pipeline:** `{best_name}` predicting `{st.session_state['target_col']}`")

            # Form for single-row prediction
            with st.form("single_prediction_form"):
                st.markdown("##### 📝 Input Feature Values for Scoring:")
                input_vals = {}
                cols_form = st.columns(min(3, len(features)))

                for idx, feat in enumerate(features):
                    col_target = cols_form[idx % len(cols_form)]
                    with col_target:
                        if feat in encoders:
                            classes = encoders[feat].classes_.tolist()
                            input_vals[feat] = st.selectbox(f"{feat}:", classes)
                        else:
                            # Numeric
                            mean_val = float(df[feat].mean())
                            min_val = float(df[feat].min())
                            max_val = float(df[feat].max())
                            input_vals[feat] = st.number_input(
                                f"{feat}:",
                                value=round(mean_val, 2),
                                min_value=round(min_val - (max_val - min_val)*0.5, 2),
                                max_value=round(max_val + (max_val - min_val)*0.5, 2)
                            )

                submit_btn = st.form_submit_button("⚡ Compute Instant Model Inference", use_container_width=True)

            if submit_btn:
                # Prepare single row vector
                sample_dict = {}
                for feat in features:
                    val = input_vals[feat]
                    if feat in encoders:
                        sample_dict[feat] = encoders[feat].transform([str(val)])[0]
                    else:
                        sample_dict[feat] = val

                sample_df = pd.DataFrame([sample_dict])
                prediction = best_model.predict(sample_df)[0]

                if st.session_state["active_task"] == "Classification":
                    if st.session_state["target_encoder"] is not None:
                        pred_label = st.session_state["target_encoder"].inverse_transform([int(prediction)])[0]
                    else:
                        pred_label = prediction

                    prob_str = ""
                    if hasattr(best_model, "predict_proba"):
                        probas = best_model.predict_proba(sample_df)[0]
                        max_prob = np.max(probas) * 100
                        prob_str = f"(Confidence: **{max_prob:.1f}%**)"

                    st.success(f"🎯 **Predicted Classification:** **{pred_label}** {prob_str}")
                else:
                    st.success(f"🎯 **Predicted Continuous Value:** **{prediction:.4f}**")

            st.markdown("<br>##### 💾 Download Production Model Artifact", unsafe_allow_html=True)
            model_bytes = io.BytesIO()
            joblib.dump({
                "model": best_model,
                "features": features,
                "label_encoders": encoders,
                "target_col": st.session_state["target_col"],
                "task": st.session_state["active_task"]
            }, model_bytes)
            model_bytes.seek(0)

            st.download_button(
                label=f"📥 Download Trained {best_name} (.joblib)",
                data=model_bytes,
                file_name=f"nexusai_{best_name.lower().replace(' ', '_')}.joblib",
                mime="application/octet-stream",
                use_container_width=True
            )


if __name__ == "__main__":
    st.set_page_config(
        page_title="NexusAI - AutoDataLab",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    from core.styling import apply_custom_styling
    apply_custom_styling()
    render_autodata_module()
