import sys
import os

# Set root directory in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# Streamlit Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="NexusAI | Multi-Domain Analytics Suite",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core imports
from core.styling import apply_custom_styling

def main():
    # Apply global CSS stylesheet
    apply_custom_styling()
    
    # ----------------------------------------------------
    # SIDEBAR NAVIGATION & TELEMETRY
    # ----------------------------------------------------
    with st.sidebar:
        # Branding
        st.markdown("""
        <div style="padding: 0.8rem 0 1.2rem 0; text-align: left;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2rem;">🔮</span>
                <div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.45rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">NexusAI</div>
                    <div style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600;">Enterprise Analytics Hub</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Intelligence Domains</p>", unsafe_allow_html=True)

        nav_options = [
            "🌐 Platform Overview",
            "🫁 Healthcare & Medical Diagnostics",
            "💳 FinTech & Fraud Intelligence",
            "🛍️ E-Commerce & Customer Intel",
            "⚡ Smart Energy & Environmental Grid",
            "🧪 Universal AutoDataLab (AutoML)"
        ]

        # Read selection or initialize session state
        if "selected_domain" not in st.session_state:
            st.session_state["selected_domain"] = nav_options[0]

        selected_page = st.radio(
            "Select Module:",
            nav_options,
            label_visibility="collapsed"
        )

        st.markdown("---")
        
        # Telemetry & System Status
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 0.4rem;'>System Telemetry</p>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.8rem; line-height: 1.6; color: #cbd5e1;">
            <div>🟢 <b>Engine:</b> Live Inference Active</div>
            <div>⚡ <b>Average Latency:</b> 34 ms</div>
            <div>📦 <b>Preloaded Hubs:</b> 5 Modules</div>
            <div>🔒 <b>Security Tier:</b> Enterprise SOC2</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clear App Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # ----------------------------------------------------
    # ROUTING TO SELECTED DOMAIN MODULE
    # ----------------------------------------------------
    if selected_page == "🌐 Platform Overview":
        from modules.home import render_home_page

        render_home_page()
    elif selected_page == "🫁 Healthcare & Medical Diagnostics":
        from modules.healthcare import render_healthcare_module

        render_healthcare_module()
    elif selected_page == "💳 FinTech & Fraud Intelligence":
        from modules.fintech import render_fintech_module

        render_fintech_module()
    elif selected_page == "🛍️ E-Commerce & Customer Intel":
        from modules.ecommerce import render_ecommerce_module

        render_ecommerce_module()
    elif selected_page == "⚡ Smart Energy & Environmental Grid":
        from modules.energy import render_energy_module

        render_energy_module()
    elif selected_page == "🧪 Universal AutoDataLab (AutoML)":
        from modules.autodata import render_autodata_module

        render_autodata_module()

if __name__ == "__main__":
    main()
