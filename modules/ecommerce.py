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
import re

from core.styling import render_metric_card, render_domain_banner
from core.utils import (
    style_plotly_figure, 
    generate_sample_ecommerce_data,
    ensure_sample_assets
)

def analyze_review_text(text: str) -> dict:
    """
    Performs fast heuristic NLP sentiment and aspect analysis.
    Computes polarity, subjectivity, and detects category aspects.
    """
    pos_words = {
        "phenomenal", "great", "excellent", "love", "loved", "fast", "sleek", 
        "recommend", "super", "premium", "outstanding", "best", "good", "perfect", 
        "easy", "durable", "high", "crisp", "worth", "exceeded", "happy"
    }
    neg_words = {
        "terrible", "bad", "damaged", "rude", "nightmare", "avoid", "poor", 
        "slow", "broken", "defective", "disappointing", "awful", "horrible", 
        "hate", "worst", "buggy", "stopped", "useless", "refund", "cheap", "waste"
    }
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return {"polarity": 0.0, "sentiment": "Neutral", "aspects": [], "word_counts": {}}
    
    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)
    
    total_eval = pos_count + neg_count
    if total_eval == 0:
        polarity = 0.05
    else:
        polarity = (pos_count - neg_count) / total_eval
        
    if polarity > 0.2:
        sentiment = "Positive"
    elif polarity < -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Aspect keywords
    aspects = []
    aspect_map = {
        "Shipping & Logistics": ["delivery", "shipping", "shipped", "arrived", "box", "package", "transit"],
        "Product Build & Hardware": ["quality", "battery", "build", "hardware", "material", "durability", "design", "screen"],
        "Customer Support": ["support", "service", "refund", "agent", "seller", "replacement", "representative"],
        "Software & Usability": ["software", "setup", "app", "firmware", "buggy", "updates", "interface"],
        "Price & Value": ["price", "cost", "penny", "expensive", "cheap", "value", "money"]
    }

    for aspect, kws in aspect_map.items():
        if any(kw in words for kw in kws):
            aspects.append(aspect)

    if not aspects:
        aspects.append("General Feedback")

    word_freq = {}
    for w in words:
        if len(w) > 3 and w not in {"this", "that", "with", "from", "have", "were", "they", "will", "does"}:
            word_freq[w] = word_freq.get(w, 0) + 1

    return {
        "polarity": polarity,
        "sentiment": sentiment,
        "aspects": aspects,
        "word_counts": word_freq,
        "pos_count": pos_count,
        "neg_count": neg_count
    }


def render_ecommerce_module():
    """Main renderer for E-Commerce & Customer Intelligence domain."""
    ensure_sample_assets()
    
    render_domain_banner(
        title="E-Commerce & Retail Intelligence Suite",
        subtitle="Aspect-Based NLP Sentiment Processing, Review Insights, & 3D RFM Behavioral Customer Segmentation",
        domain_type="ecommerce",
        badge_text="NLP & 3D Clustering"
    )

    tab_nlp, tab_rfm, tab_stream = st.tabs([
        "💬 NLP Sentiment & Aspect Intelligence",
        "🎯 3D RFM Behavioral Customer Segmentation",
        "📦 Live Feedback Review Stream"
    ])

    # ----------------------------------------------------
    # TAB 1: NLP Sentiment & Aspect Intelligence
    # ----------------------------------------------------
    with tab_nlp:
        st.markdown("#### Real-Time Feedback Sentiment Engine")
        st.markdown(
            "Analyze unstructured customer reviews to extract emotional tone, polarity score, "
            "and business dimension aspects (Quality, Shipping, Support, Value)."
        )

        col_presets = st.columns(4)
        with col_presets[0]:
            if st.button("🌟 5-Star Rave", use_container_width=True):
                st.session_state["review_input"] = "The build quality is phenomenal and battery lasts for days! Highly recommend to everyone."
        with col_presets[1]:
            if st.button("😡 Shipping Nightmare", use_container_width=True):
                st.session_state["review_input"] = "Terrible delivery experience. Item arrived completely damaged and customer support was rude and unhelpful."
        with col_presets[2]:
            if st.button("⚖️ Mixed / Neutral", use_container_width=True):
                st.session_state["review_input"] = "Average product for the price point. Does the job as expected, but nothing extraordinary."
        with col_presets[3]:
            if st.button("🔧 Software Buggy", use_container_width=True):
                st.session_state["review_input"] = "Decent hardware features, but software updates are super buggy and setup took forever."

        review_text = st.text_area(
            "Input Customer Review Text:",
            value=st.session_state.get("review_input", "The build quality is phenomenal and battery lasts for days! Highly recommend to everyone."),
            height=100
        )

        nlp_res = analyze_review_text(review_text)

        st.markdown("<br>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns([1, 1.4])

        with col_res1:
            if nlp_res["sentiment"] == "Positive":
                s_color = "#34d399"
                s_badge = '<span class="badge badge-success">Positive Sentiment</span>'
            elif nlp_res["sentiment"] == "Negative":
                s_color = "#ef4444"
                s_badge = '<span class="badge badge-danger">Negative / Escalate</span>'
            else:
                s_color = "#fbbf24"
                s_badge = '<span class="badge badge-warning">Neutral Tone</span>'

            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 4px solid {s_color};">
                <div style="margin-bottom: 0.5rem;">{s_badge}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Sentiment Polarity Index</div>
                <div style="font-size: 3.2rem; font-weight: 800; color: {s_color}; font-family: 'Outfit'; margin: 0.4rem 0;">
                    {nlp_res["polarity"]:+0.2f}
                </div>
                <div style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">Tone: {nlp_res["sentiment"]}</div>
                <p style="color: #cbd5e1; font-size: 0.85rem; margin-top: 0.8rem;">
                    Detected Keywords: +{nlp_res['pos_count']} positive | -{nlp_res['neg_count']} negative terms
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 🏷️ Detected Aspects")
            for asp in nlp_res["aspects"]:
                st.markdown(f"- 📌 **{asp}**")

        with col_res2:
            if nlp_res["word_counts"]:
                top_words = sorted(nlp_res["word_counts"].items(), key=lambda x: x[1], reverse=True)[:8]
                w_df = pd.DataFrame(top_words, columns=["Keyword", "Frequency"])
                fig_kw = px.bar(
                    w_df,
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    color="Frequency",
                    color_continuous_scale="Purples"
                )
                fig_kw.update_layout(height=280, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                style_plotly_figure(fig_kw, "Salient Keywords Extracted")
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("Type review text to analyze salient keywords.")

    # ----------------------------------------------------
    # TAB 2: 3D RFM Behavioral Customer Segmentation
    # ----------------------------------------------------
    with tab_rfm:
        st.markdown("#### 3D Customer Behavioral Space (RFM Clustering)")
        st.markdown(
            "Partition customers into distinct lifetime value cohorts using **Recency** (days since purchase), "
            "**Frequency** (order volume), and **Monetary Value** (total revenue generated)."
        )

        rfm_df, _ = generate_sample_ecommerce_data()

        # Deterministic rule-based clustering simulating K-Means centroids
        def assign_cluster(row):
            if row["Recency_Days"] < 30 and row["Monetary_Spend_USD"] > 600:
                return "1. Champions & VIPs"
            elif row["Recency_Days"] < 60 and row["Frequency_Orders"] >= 5:
                return "2. Loyal Core Customers"
            elif row["Recency_Days"] > 75 and row["Monetary_Spend_USD"] > 400:
                return "3. High-Value At-Risk"
            else:
                return "4. Casual / Hibernating"

        rfm_df["Segment"] = rfm_df.apply(assign_cluster, axis=1)

        # 3D Scatter Plot
        color_map = {
            "1. Champions & VIPs": "#10b981",
            "2. Loyal Core Customers": "#38bdf8",
            "3. High-Value At-Risk": "#f43f5e",
            "4. Casual / Hibernating": "#94a3b8"
        }

        fig_3d = px.scatter_3d(
            rfm_df,
            x="Recency_Days",
            y="Frequency_Orders",
            z="Monetary_Spend_USD",
            color="Segment",
            color_discrete_map=color_map,
            hover_data=["Customer_ID", "Tenure_Days"],
            opacity=0.85
        )

        style_plotly_figure(fig_3d, "")
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title="Recency (Days)", backgroundcolor="rgba(15, 23, 42, 0.4)", gridcolor="rgba(255, 255, 255, 0.1)"),
                yaxis=dict(title="Frequency (Orders)", backgroundcolor="rgba(15, 23, 42, 0.4)", gridcolor="rgba(255, 255, 255, 0.1)"),
                zaxis=dict(title="Monetary Spend ($)", backgroundcolor="rgba(15, 23, 42, 0.4)", gridcolor="rgba(255, 255, 255, 0.1)"),
            ),
            height=460,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # Segment Playbook
        st.markdown("##### 🎯 Marketing Automation Playbook by Cohort")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        with col_p1:
            st.markdown("""
            <div class="glass-card">
                <span class="badge badge-success">VIP Group</span>
                <h4 style="margin: 0.3rem 0; color: #10b981;">Champions</h4>
                <p style="font-size: 0.82rem; color: #cbd5e1;">High spend, frequent, recent. Strategy: Early access to new releases & VIP concierge perks.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_p2:
            st.markdown("""
            <div class="glass-card">
                <span class="badge badge-cyan">Retention</span>
                <h4 style="margin: 0.3rem 0; color: #38bdf8;">Loyalists</h4>
                <p style="font-size: 0.82rem; color: #cbd5e1;">Steady recurring buyers. Strategy: Cross-sell relevant accessories & subscription loyalty tiers.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_p3:
            st.markdown("""
            <div class="glass-card">
                <span class="badge badge-danger">Danger Zone</span>
                <h4 style="margin: 0.3rem 0; color: #f43f5e;">At-Risk VIPs</h4>
                <p style="font-size: 0.82rem; color: #cbd5e1;">High past spend, but inactive lately. Strategy: Immediate personalized discount re-engagement campaign.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_p4:
            st.markdown("""
            <div class="glass-card">
                <span class="badge badge-warning">Low Cost</span>
                <h4 style="margin: 0.3rem 0; color: #cbd5e1;">Casuals</h4>
                <p style="font-size: 0.82rem; color: #cbd5e1;">Occasional low-value buyers. Strategy: Automated email drip sequences during seasonal sales.</p>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 3: Review Stream Dataset
    # ----------------------------------------------------
    with tab_stream:
        st.markdown("#### Live Ingested Customer Review Stream")
        _, reviews_df = generate_sample_ecommerce_data()
        
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            filter_sentiment = st.multiselect("Filter by Sentiment:", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
        
        filtered_rev = reviews_df[reviews_df["Sentiment"].isin(filter_sentiment)]
        st.dataframe(filtered_rev, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="NexusAI - E-Commerce Intelligence",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    from core.styling import apply_custom_styling
    apply_custom_styling()
    render_ecommerce_module()

