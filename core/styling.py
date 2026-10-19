import streamlit as st

def apply_custom_styling():
    """Injects high-end glassmorphic dark styling, typography, and responsive component themes."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Gradient Header Text */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Glassmorphic Card Container */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(147, 197, 253, 0.25);
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.6);
    }

    /* Glassmorphic Metric Box */
    .metric-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: left;
        position: relative;
        overflow: hidden;
    }

    .metric-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .metric-delta {
        font-size: 0.8rem;
        color: #34d399;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .badge-primary { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); }
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-danger { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-cyan { background: rgba(6, 182, 212, 0.2); color: #67e8f9; border: 1px solid rgba(6, 182, 212, 0.4); }

    /* Custom Domain Header Banners */
    .domain-banner {
        padding: 1.4rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .banner-health { background: linear-gradient(135deg, rgba(14, 116, 144, 0.35) 0%, rgba(15, 23, 42, 0.6) 100%); border-left: 4px solid #06b6d4; }
    .banner-fintech { background: linear-gradient(135deg, rgba(5, 150, 105, 0.35) 0%, rgba(15, 23, 42, 0.6) 100%); border-left: 4px solid #10b981; }
    .banner-ecommerce { background: linear-gradient(135deg, rgba(147, 51, 234, 0.35) 0%, rgba(15, 23, 42, 0.6) 100%); border-left: 4px solid #a855f7; }
    .banner-energy { background: linear-gradient(135deg, rgba(217, 119, 6, 0.35) 0%, rgba(15, 23, 42, 0.6) 100%); border-left: 4px solid #f59e0b; }
    .banner-autodata { background: linear-gradient(135deg, rgba(79, 70, 229, 0.35) 0%, rgba(15, 23, 42, 0.6) 100%); border-left: 4px solid #6366f1; }

    /* Streamlit element overrides */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    .stButton>button:hover {
        border-color: #818cf8;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.3);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None, color_type: str = "primary"):
    """Renders a styled glassmorphic metric card."""
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_domain_banner(title: str, subtitle: str, domain_type: str, badge_text: str):
    """Renders a domain top banner with icon and badges."""
    css_class = f"banner-{domain_type}"
    badge_class = f"badge-{domain_type if domain_type in ['success', 'warning', 'danger'] else 'primary'}"
    st.markdown(f"""
    <div class="domain-banner {css_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <span class="badge {badge_class}">{badge_text}</span>
                <h2 style="margin: 0.4rem 0 0.2rem 0; color: #f8fafc; font-size: 1.8rem;">{title}</h2>
                <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">{subtitle}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
