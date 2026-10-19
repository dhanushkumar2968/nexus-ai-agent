import sys
import os

# Ensure project root is in sys.path when running module directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from core.styling import render_metric_card, render_domain_banner
from core.utils import style_plotly_figure

def render_home_page():
    """Renders the executive overview and interactive architecture hub."""
    st.markdown('<div class="hero-title">NexusAI Enterprise Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Unified Multi-Domain Analytics Suite Powered by Machine Learning, Deep Learning, and Advanced Data Science.</div>',
        unsafe_allow_html=True
    )

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Domains Deployed", "5 Hubs", "Healthcare, FinTech, Retail, IoT, AutoML", "primary")
    with col2:
        render_metric_card("Model Accuracy Avg", "94.8%", "+3.2% vs baseline models", "success")
    with col3:
        render_metric_card("Real-Time Ingestion", "1.2M+", "events / 24h simulation", "cyan")
    with col4:
        render_metric_card("Inference Latency", "38 ms", "Sub-second on CPU/GPU", "warning")

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-Domain Architecture Navigator
    st.markdown("### 🌐 Core Intelligence Modules")
    st.markdown("Select any domain from the left sidebar to access deep analytics, predictive models, and real-time simulators.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="glass-card">
            <span class="badge badge-cyan">Computer Vision & Deep Learning</span>
            <h3 style="margin-top: 0.6rem; color: #38bdf8;">1. Healthcare & Medical Diagnostics</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem;">
                Chest radiographic screening for pulmonary pathology using deep neural networks, paired with 
                <b>Grad-CAM Explainable AI (XAI)</b> visual saliency heatmaps and an interactive cardiovascular clinical risk predictor.
            </p>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                <b>Tech Stack:</b> PyTorch CNN / Transfer Learning, Grad-CAM Heatmaps, Tabular Clinical Risk Models.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <span class="badge badge-primary">NLP & Unsupervised Learning</span>
            <h3 style="margin-top: 0.6rem; color: #a855f7;">3. E-Commerce & Customer Intelligence</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem;">
                Natural Language sentiment analytics on customer review streams, multi-aspect feedback extraction, and 
                <b>3D K-Means customer segmentation</b> based on Recency, Frequency, and Monetary (RFM) behavioral metrics.
            </p>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                <b>Tech Stack:</b> Text Polarity Analysis, Aspect Classifier, 3D RFM Clustering, Plotly 3D scatter.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <span class="badge badge-primary">No-Code AutoML & Data Science</span>
            <h3 style="margin-top: 0.6rem; color: #818cf8;">5. Universal AutoDataLab (BYOD)</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem;">
                Bring Your Own Data: Instant automated exploratory data analysis (EDA), correlation matrices, distribution profiling, 
                and multi-algorithm <b>AutoML leaderboard</b> with exportable trained pipeline models.
            </p>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                <b>Tech Stack:</b> Scikit-Learn pipelines, Automated EDA Profiler, Multi-model benchmarking (RF, GBDT, LR).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="glass-card">
            <span class="badge badge-success">Machine Learning & Anomaly Detection</span>
            <h3 style="margin-top: 0.6rem; color: #34d399;">2. FinTech Fraud Intelligence</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem;">
                High-frequency transaction fraud detection blending supervised gradient boosting with 
                <b>unsupervised Isolation Forests</b> to detect zero-day fraud attacks, coupled with a dynamic financial cost-benefit optimization matrix.
            </p>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                <b>Tech Stack:</b> XGBoost / Random Forest, Isolation Forest, Cost-Sensitive Threshold Optimization.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <span class="badge badge-warning">Time Series & Predictive IoT</span>
            <h3 style="margin-top: 0.6rem; color: #fbbf24;">4. Smart Energy & Environmental Grid</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem;">
                Forecasting municipal power demand and Air Quality Index (AQI) dynamics. Features seasonality decomposition, 
                peak load warning thresholds, and interactive weather impact scenario simulation.
            </p>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                <b>Tech Stack:</b> Time-Series Rolling Forecast, Seasonality Decomposer, Multi-variable Environmental Regressors.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Multi-Domain Performance Radar Chart
    st.markdown("<br>### 📊 Platform Multi-Dimensional Capability Matrix", unsafe_allow_html=True)
    
    col_radar, col_arch = st.columns([1.1, 1])

    with col_radar:
        categories = [
            'Predictive Accuracy', 'Inference Speed', 
            'Explainability (XAI)', 'Data Volume Handling', 
            'Real-Time Adaptation', 'Business Impact'
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[95, 90, 96, 85, 88, 98],
            theta=categories,
            fill='toself',
            name='Healthcare (Deep Learning)',
            line=dict(color='#06b6d4', width=2),
            fillcolor='rgba(6, 182, 212, 0.2)'
        ))

        fig.add_trace(go.Scatterpolar(
            r=[98, 97, 86, 96, 95, 99],
            theta=categories,
            fill='toself',
            name='FinTech (Fraud ML)',
            line=dict(color='#10b981', width=2),
            fillcolor='rgba(16, 185, 129, 0.2)'
        ))

        fig.add_trace(go.Scatterpolar(
            r=[90, 88, 92, 90, 91, 93],
            theta=categories,
            fill='toself',
            name='AutoDataLab (Universal)',
            line=dict(color='#818cf8', width=2),
            fillcolor='rgba(129, 140, 248, 0.2)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255, 255, 255, 0.1)', linecolor='rgba(255, 255, 255, 0.1)'),
                angularaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', linecolor='rgba(255, 255, 255, 0.1)')
            ),
            showlegend=True,
            legend=dict(orientation="h", y=-0.15),
            height=380
        )
        style_plotly_figure(fig, "Domain AI Performance Attributes")
        st.plotly_chart(fig, use_container_width=True)

    with col_arch:
        st.markdown("""
        <div class="glass-card" style="height: 380px; overflow-y: auto;">
            <h4 style="color: #f8fafc; margin-top: 0;">🛠️ System Architecture & Data Flow</h4>
            <div style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
                <p><b>1. Unified Ingestion Layer:</b><br>
                Accepts multi-modal inputs: DICOM/PNG/JPG medical images, streaming transaction tabular records, unstructured text reviews, and hourly IoT telemetry.</p>
                <p><b>2. Feature Engineering & Preprocessing:</b><br>
                Automated imputation, scaling (Standard/Robust), tokenization, aspect keyword scoring, and Fourier seasonal decomposition.</p>
                <p><b>3. Model Inference Engine:</b><br>
                Deep Learning vision backbones, Supervised Classifiers (XGBoost/Random Forest), Unsupervised Detectors (Isolation Forest, K-Means), and Regressors.</p>
                <p><b>4. Explainability & Action Layer:</b><br>
                Grad-CAM heatmaps for vision, SHAP-inspired feature impacts for tabular models, and financial cost-sensitive trade-off matrices.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="NexusAI - Multi-Domain Platform",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    from core.styling import apply_custom_styling
    apply_custom_styling()
    render_home_page()

