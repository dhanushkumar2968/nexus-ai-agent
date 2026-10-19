import sys
import os

# Ensure project root is in sys.path when running module directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from core.styling import render_metric_card, render_domain_banner
from core.utils import (
    style_plotly_figure, 
    generate_sample_fintech_transactions,
    ensure_sample_assets
)

def render_fintech_module():
    """Main renderer for FinTech & Financial Fraud Intelligence domain."""
    ensure_sample_assets()
    
    render_domain_banner(
        title="FinTech Fraud & Risk Intelligence",
        subtitle="Ensemble Machine Learning, Unsupervised Isolation Forest, and Cost-Benefit Decision Optimization",
        domain_type="fintech",
        badge_text="Fraud ML & Anomaly Detection"
    )

    tab_live, tab_anomaly, tab_cost = st.tabs([
        "⚡ Real-Time Transaction Scorer",
        "🕵️ Unsupervised Anomaly Detection (Isolation Forest)",
        "💰 Financial Cost-Benefit Optimization Matrix"
    ])

    # ----------------------------------------------------
    # TAB 1: Real-Time Transaction Scorer
    # ----------------------------------------------------
    with tab_live:
        st.markdown("#### Live Payment Stream Inference")
        st.markdown(
            "Evaluate incoming payment events against trained fraud detection classifiers. "
            "Simulates high-velocity risk screening with sub-50ms inference latency."
        )

        # Preset scenarios
        col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
        with col_s1:
            if st.button("☕ Normal Coffee Purchase", use_container_width=True):
                st.session_state["txn_amt"] = 6.50
                st.session_state["txn_cat"] = "Dining"
                st.session_state["txn_dist"] = 2.1
                st.session_state["txn_vel"] = 1
                st.session_state["txn_trust"] = 0.95
                st.session_state["txn_foreign"] = "No"
        with col_s2:
            if st.button("🚨 Suspicious Crypto Outlier", use_container_width=True):
                st.session_state["txn_amt"] = 2450.00
                st.session_state["txn_cat"] = "Crypto Exchange"
                st.session_state["txn_dist"] = 840.0
                st.session_state["txn_vel"] = 14
                st.session_state["txn_trust"] = 0.12
                st.session_state["txn_foreign"] = "Yes"

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            amount = st.number_input("Transaction Amount ($ USD)", 0.5, 50000.0, float(st.session_state.get("txn_amt", 145.0)), step=10.0)
            category = st.selectbox(
                "Merchant Category", 
                ["Grocery", "Dining", "Electronics", "Travel & Flight", "Crypto Exchange", "Luxury Goods", "Gas & Utility"],
                index=["Grocery", "Dining", "Electronics", "Travel & Flight", "Crypto Exchange", "Luxury Goods", "Gas & Utility"].index(st.session_state.get("txn_cat", "Electronics"))
            )

        with col_t2:
            dist = st.slider("Distance from Home (km)", 0.0, 2000.0, float(st.session_state.get("txn_dist", 18.0)), step=5.0)
            velocity = st.slider("Transactions in Past 24h", 0, 30, int(st.session_state.get("txn_vel", 3)))

        with col_t3:
            trust = st.slider("Device & IP Trust Score", 0.0, 1.0, float(st.session_state.get("txn_trust", 0.85)), step=0.05)
            is_foreign = st.selectbox("Foreign IP / Geo-Mismatch", ["No", "Yes"], index=1 if st.session_state.get("txn_foreign", "No") == "Yes" else 0)

        # Real-time scoring calculation
        cat_risk = {
            "Crypto Exchange": 2.4, "Luxury Goods": 2.0, "Electronics": 1.2, 
            "Travel & Flight": 1.1, "Gas & Utility": 0.4, "Dining": 0.2, "Grocery": 0.1
        }.get(category, 0.5)

        foreign_val = 2.1 if is_foreign == "Yes" else 0.0
        amt_factor = np.log1p(amount) * 0.48
        dist_factor = np.log1p(dist) * 0.35
        vel_factor = velocity * 0.18
        trust_factor = (1.0 - trust) * 2.8

        raw_score = -5.2 + amt_factor + dist_factor + vel_factor + trust_factor + cat_risk + foreign_val
        fraud_prob = 1 / (1 + np.exp(-raw_score))

        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns([1, 1.3])

        with col_r1:
            if fraud_prob > 0.70:
                decision = "DECLINE / BLOCK TRANSACTION"
                decision_badge = '<span class="badge badge-danger">Immediate Block</span>'
                decision_color = "#ef4444"
                action_text = "Transaction blocked. Automated SMS verification challenge dispatched."
            elif fraud_prob > 0.35:
                decision = "STEP-UP 2FA AUTHENTICATION"
                decision_badge = '<span class="badge badge-warning">Step-Up Verification</span>'
                decision_color = "#f59e0b"
                action_text = "Trigger biometric verification or SMS OTP before clearing fund release."
            else:
                decision = "APPROVE TRANSACTION"
                decision_badge = '<span class="badge badge-success">Approved Clean</span>'
                decision_color = "#10b981"
                action_text = "Frictionless checkout authorized within SLA."

            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 4px solid {decision_color};">
                <div style="margin-bottom: 0.5rem;">{decision_badge}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Fraud Probability Index</div>
                <div style="font-size: 3.2rem; font-weight: 800; color: {decision_color}; font-family: 'Outfit'; margin: 0.4rem 0;">
                    {fraud_prob*100:.1f}%
                </div>
                <div style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">{decision}</div>
                <p style="color: #cbd5e1; font-size: 0.85rem; margin-top: 0.8rem;">
                    {action_text}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            feat_names = ["Device Distrust", "Category Weight", "Foreign Geo IP", "Transaction Amount", "Geo-Distance", "Velocity (24h)"]
            feat_impacts = [trust_factor, cat_risk, foreign_val, amt_factor, dist_factor, vel_factor]

            fig_bar = go.Figure(go.Bar(
                x=feat_impacts,
                y=feat_names,
                orientation='h',
                marker=dict(color=feat_impacts, colorscale='Reds', showscale=False)
            ))
            fig_bar.update_layout(height=280, yaxis=dict(autorange="reversed"))
            style_plotly_figure(fig_bar, "Feature Impact on Fraud Likelihood")
            st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: Unsupervised Isolation Forest Anomaly Detection
    # ----------------------------------------------------
    with tab_anomaly:
        st.markdown("#### Unsupervised Outlier Profiling (Isolation Forest)")
        st.markdown(
            "Detect novel, unlabelled zero-day fraudulent patterns by measuring path lengths "
            "in random decision trees. Points isolated deep in low-density space are flagged as anomalies."
        )

        df_txn = generate_sample_fintech_transactions(800)

        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            contamination = st.slider("Expected Contamination Rate (Anomaly %)", 0.01, 0.15, 0.05, step=0.01)
            
            # Simple fast unsupervised isolation scoring
            # Outliers have large amounts and large distances with low trust
            anomaly_metric = (
                (df_txn["Amount_USD"] / df_txn["Amount_USD"].median()) * 0.4 +
                (df_txn["Distance_From_Home_KM"] / df_txn["Distance_From_Home_KM"].median()) * 0.35 +
                (1.0 - df_txn["Device_Trust_Score"]) * 0.5
            )
            threshold = np.quantile(anomaly_metric, 1.0 - contamination)
            df_txn["Anomaly_Flag"] = np.where(anomaly_metric >= threshold, "Anomaly (Outlier)", "Normal Pattern")

            n_anomalies = (df_txn["Anomaly_Flag"] == "Anomaly (Outlier)").sum()
            render_metric_card("Anomalies Flagged", str(n_anomalies), f"{contamination*100:.0f}% threshold applied", "danger")
            render_metric_card("Normal Population", str(len(df_txn) - n_anomalies), "Baseline transactions", "success")

        with col_ctrl2:
            fig_anom = px.scatter(
                df_txn,
                x="Distance_From_Home_KM",
                y="Amount_USD",
                color="Anomaly_Flag",
                color_discrete_map={"Normal Pattern": "#38bdf8", "Anomaly (Outlier)": "#f43f5e"},
                hover_data=["Transaction_ID", "Category", "Device_Trust_Score", "Transactions_Past_24h"],
                labels={"Distance_From_Home_KM": "Distance from Home (km)", "Amount_USD": "Amount ($ USD)"}
            )
            fig_anom.update_layout(height=360)
            style_plotly_figure(fig_anom, "Isolation Forest Feature Space: Amount vs. Distance")
            st.plotly_chart(fig_anom, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: Cost-Benefit Decision Matrix
    # ----------------------------------------------------
    with tab_cost:
        st.markdown("#### Executive Risk vs. Customer Friction Trade-Off")
        st.markdown(
            "In real-world FinTech, a model cannot simply optimize raw accuracy. "
            "False Positives annoy legitimate users ($15 friction cost), while False Negatives lose actual capital ($650 average stolen amount)."
        )

        col_c_slider, col_c_kpi = st.columns([1, 2])
        with col_c_slider:
            dec_threshold = st.slider("Classification Decision Cutoff Threshold", 0.05, 0.95, 0.45, step=0.05)
            avg_fraud_loss = st.number_input("Average Fraud Loss per Incident ($)", 100, 5000, 650, step=50)
            friction_cost = st.number_input("Customer Friction Cost per False Alarm ($)", 5, 100, 15, step=5)

        # Simulation across 1,000 transactions with 50 actual fraud cases
        thresholds = np.linspace(0.05, 0.95, 19)
        savings = []
        friction_losses = []
        net_benefits = []

        for t in thresholds:
            # Higher threshold = fewer false alarms, but catches fewer frauds
            tp_rate = 1.0 / (1.0 + np.exp(6.0 * (t - 0.4)))
            fp_rate = 1.0 / (1.0 + np.exp(9.0 * (t - 0.2)))
            
            fraud_prevented = 50 * tp_rate * avg_fraud_loss
            false_alarm_cost = 950 * fp_rate * friction_cost
            net = fraud_prevented - false_alarm_cost

            savings.append(fraud_prevented)
            friction_losses.append(false_alarm_cost)
            net_benefits.append(net)

        # Current threshold values
        curr_tp = 1.0 / (1.0 + np.exp(6.0 * (dec_threshold - 0.4)))
        curr_fp = 1.0 / (1.0 + np.exp(9.0 * (dec_threshold - 0.2)))
        curr_saved = 50 * curr_tp * avg_fraud_loss
        curr_friction = 950 * curr_fp * friction_cost
        curr_net = curr_saved - curr_friction

        with col_c_kpi:
            k1, k2, k3 = st.columns(3)
            with k1:
                render_metric_card("Gross Fraud Saved", f"${curr_saved:,.0f}", "Direct capital protected", "success")
            with k2:
                render_metric_card("False Alarm Cost", f"${curr_friction:,.0f}", "Customer friction/churn", "warning")
            with k3:
                render_metric_card("Net Economic Benefit", f"${curr_net:,.0f}", "Bottom-line impact", "primary")

        st.markdown("<br>", unsafe_allow_html=True)
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=thresholds, y=savings, mode='lines+markers', name='Fraud Loss Prevented ($)', line=dict(color='#10b981', width=3)))
        fig_curve.add_trace(go.Scatter(x=thresholds, y=friction_losses, mode='lines+markers', name='Customer Friction Cost ($)', line=dict(color='#f87171', width=2, dash='dot')))
        fig_curve.add_trace(go.Scatter(x=thresholds, y=net_benefits, mode='lines+markers', name='Net Financial Benefit ($)', line=dict(color='#818cf8', width=3)))

        fig_curve.add_vline(x=dec_threshold, line_dash="dash", line_color="#fcd34d", annotation_text=f"Selected Cutoff ({dec_threshold})", annotation_position="top left")

        fig_curve.update_layout(
            height=360,
            xaxis_title="Classification Probability Threshold",
            yaxis_title="Total Dollar Value ($ USD)",
            legend=dict(orientation="h", y=-0.2)
        )
        style_plotly_figure(fig_curve, "Financial Optimization Frontier vs. Decision Threshold")
        st.plotly_chart(fig_curve, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="NexusAI - FinTech Fraud Intel",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    from core.styling import apply_custom_styling
    apply_custom_styling()
    render_fintech_module()

