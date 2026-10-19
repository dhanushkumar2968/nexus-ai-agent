import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import os

PLOTLY_THEME = {
    "paper_bgcolor": "rgba(15, 23, 42, 0.4)",
    "plot_bgcolor": "rgba(15, 23, 42, 0.4)",
    "font": {"family": "Inter, sans-serif", "color": "#cbd5e1"},
    "xaxis": {"gridcolor": "rgba(255, 255, 255, 0.08)", "zerolinecolor": "rgba(255, 255, 255, 0.15)"},
    "yaxis": {"gridcolor": "rgba(255, 255, 255, 0.08)", "zerolinecolor": "rgba(255, 255, 255, 0.15)"},
    "margin": dict(l=40, r=30, t=45, b=40)
}

COLOR_PALETTE = ["#38bdf8", "#818cf8", "#ec4899", "#34d399", "#fbbf24", "#f87171", "#a78bfa"]


def style_plotly_figure(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies unified dark glassmorphic styling to any Plotly chart."""
    fig.update_layout(
        **PLOTLY_THEME,
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"family": "Outfit, sans-serif", "size": 16, "color": "#f8fafc"},
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top"
        }
    )
    return fig


@st.cache_data(show_spinner=False)
def generate_sample_healthcare_data(n_samples: int = 400) -> pd.DataFrame:
    """Generates realistic clinical cardiovascular health records."""
    np.random.seed(42)
    age = np.random.randint(29, 78, size=n_samples)
    sex = np.random.choice(["Male", "Female"], size=n_samples, p=[0.55, 0.45])
    resting_bp = np.random.normal(130, 18, size=n_samples).astype(int)
    cholesterol = np.random.normal(245, 45, size=n_samples).astype(int)
    fasting_bs = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])
    max_hr = (220 - age) - np.random.randint(10, 45, size=n_samples)
    exercise_angina = np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])
    oldpeak = np.clip(np.random.exponential(1.1, size=n_samples), 0, 6.2).round(1)
    
    # Calculate physiological disease probability
    risk_score = (
        (age > 55) * 1.5 +
        (resting_bp > 140) * 1.8 +
        (cholesterol > 250) * 1.4 +
        (fasting_bs == 1) * 1.6 +
        (max_hr < 130) * 1.5 +
        (exercise_angina == "Yes") * 2.2 +
        oldpeak * 0.9
    )
    prob = 1 / (1 + np.exp(-(risk_score - 4.5)))
    target = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "Patient_ID": [f"PT-{1000+i}" for i in range(n_samples)],
        "Age": age,
        "Gender": sex,
        "Resting_BP_mmHg": resting_bp,
        "Cholesterol_mg_dL": cholesterol,
        "Fasting_Blood_Sugar": fasting_bs,
        "Max_Heart_Rate": max_hr,
        "Exercise_Angina": exercise_angina,
        "ST_Depression": oldpeak,
        "Heart_Disease_Risk": target
    })
    return df


@st.cache_data(show_spinner=False)
def generate_sample_fintech_transactions(n_samples: int = 1200) -> pd.DataFrame:
    """Generates synthetic high-throughput financial transactions with fraud patterns."""
    np.random.seed(101)
    time_index = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq="12min")
    
    categories = ["Grocery", "Electronics", "Travel & Flight", "Dining", "Crypto Exchange", "Luxury Goods", "Gas & Utility"]
    cat_weights = [0.35, 0.15, 0.08, 0.22, 0.04, 0.06, 0.10]
    category = np.random.choice(categories, size=n_samples, p=cat_weights)
    
    amount = []
    for cat in category:
        if cat in ["Electronics", "Luxury Goods", "Crypto Exchange"]:
            amount.append(np.round(np.random.exponential(650) + 120, 2))
        elif cat == "Travel & Flight":
            amount.append(np.round(np.random.normal(500, 150), 2))
        else:
            amount.append(np.round(np.random.exponential(45) + 10, 2))
    amount = np.array(amount)
    
    distance_from_home = np.random.exponential(15, size=n_samples).round(1)
    velocity_24h = np.random.poisson(3, size=n_samples)
    device_trust_score = np.random.uniform(0.1, 1.0, size=n_samples).round(2)
    foreign_transaction = np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08])
    
    # Fraud probability calculation
    fraud_indicator = (
        (amount > 750) * 2.5 +
        (distance_from_home > 80) * 3.0 +
        (velocity_24h > 7) * 2.2 +
        (device_trust_score < 0.3) * 3.1 +
        (foreign_transaction == 1) * 2.4
    )
    fraud_prob = 1 / (1 + np.exp(-(fraud_indicator - 5.5)))
    is_fraud = (np.random.rand(n_samples) < fraud_prob).astype(int)

    df = pd.DataFrame({
        "Transaction_ID": [f"TXN-{880000+i}" for i in range(n_samples)],
        "Timestamp": time_index,
        "Amount_USD": amount,
        "Category": category,
        "Distance_From_Home_KM": distance_from_home,
        "Transactions_Past_24h": velocity_24h,
        "Device_Trust_Score": device_trust_score,
        "Is_Foreign_IP": foreign_transaction,
        "Is_Fraudulent": is_fraud
    })
    return df


@st.cache_data(show_spinner=False)
def generate_sample_ecommerce_data(n_samples: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates sentiment review stream and RFM customer segmentation data."""
    np.random.seed(77)
    
    # Customer RFM data
    n_customers = 350
    recency = np.random.exponential(45, size=n_customers).astype(int) + 1
    frequency = np.random.poisson(6, size=n_customers) + 1
    monetary = (frequency * np.random.normal(85, 25, size=n_customers)).round(2)
    monetary = np.clip(monetary, 20, 3500)
    tenure_days = np.random.randint(30, 730, size=n_customers)

    rfm_df = pd.DataFrame({
        "Customer_ID": [f"CUST-{5000+i}" for i in range(n_customers)],
        "Recency_Days": recency,
        "Frequency_Orders": frequency,
        "Monetary_Spend_USD": monetary,
        "Tenure_Days": tenure_days
    })

    # Review text samples
    templates = [
        ("The build quality is phenomenal and battery lasts for days! Highly recommend.", "Positive", "Hardware", 5),
        ("Terrible delivery experience. Item arrived damaged and customer support was rude.", "Negative", "Shipping & Support", 1),
        ("Average product for the price point. Does the job, but nothing extraordinary.", "Neutral", "Value", 3),
        ("Exceeded all my expectations! Fast shipping, sleek design, very easy setup.", "Positive", "Experience", 5),
        ("Stopped working after 2 weeks. Refund process is a total nightmare. Avoid!", "Negative", "Reliability", 1),
        ("Decent features, but software updates are buggy. Hope they patch it soon.", "Neutral", "Software", 3),
        ("Super premium feel and crisp performance. Worth every single penny.", "Positive", "Performance", 5),
        ("Wrong color sent and box was open. Very disappointing service from seller.", "Negative", "Shipping", 2),
        ("Battery health degrades fast, but display is gorgeous. Mixed feelings.", "Neutral", "Hardware", 3),
        ("Outstanding customer support! Replaced my defective unit within 24 hours.", "Positive", "Customer Support", 5)
    ]
    
    reviews = []
    for i in range(n_samples):
        tmpl = templates[np.random.choice(len(templates))]
        reviews.append({
            "Review_ID": f"REV-{20000+i}",
            "Review_Text": tmpl[0],
            "Sentiment": tmpl[1],
            "Aspect_Category": tmpl[2],
            "Rating": tmpl[3],
            "Helpful_Votes": np.random.poisson(4)
        })
    reviews_df = pd.DataFrame(reviews)
    
    return rfm_df, reviews_df


@st.cache_data(show_spinner=False)
def generate_sample_energy_data(n_days: int = 90) -> pd.DataFrame:
    """Generates hourly smart grid load and Air Quality Index (AQI) time series."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days * 24, freq="h")
    hours = dates.hour
    dayofweek = dates.dayofweek
    
    # Daily diurnal curve (peak at 14:00 and 20:00)
    diurnal = 35 * np.sin((hours - 6) * np.pi / 12) + 20 * np.sin((hours - 17) * np.pi / 6)
    weekend_factor = np.where(dayofweek >= 5, -15, 10)
    
    # Temperature simulation (colder in winter, hotter in summer)
    base_temp = 22 + 8 * np.sin(np.linspace(0, 3 * np.pi, len(dates))) + np.random.normal(0, 3, len(dates))
    
    # Power demand MW
    base_load = 280
    cooling_heating_load = np.abs(base_temp - 20) * 4.5
    energy_demand_mw = base_load + diurnal + weekend_factor + cooling_heating_load + np.random.normal(0, 8, len(dates))
    energy_demand_mw = np.clip(energy_demand_mw, 180, 520).round(2)
    
    # AQI calculation (inversely correlated with wind, higher during weekday rush hours)
    rush_hour = np.where((hours >= 8) & (hours <= 10) | (hours >= 17) & (hours <= 20), 45, 15)
    aqi = 65 + rush_hour * 0.8 + (energy_demand_mw * 0.12) + np.random.normal(0, 12, len(dates))
    aqi = np.clip(aqi, 25, 310).round(1)

    df = pd.DataFrame({
        "Timestamp": dates,
        "Energy_Demand_MW": energy_demand_mw,
        "Temperature_C": np.round(base_temp, 1),
        "AQI": aqi,
        "Renewable_Solar_MW": np.clip(np.sin((hours - 6) * np.pi / 12) * 95 + np.random.normal(0, 5, len(dates)), 0, 100).round(2),
        "Grid_Frequency_Hz": np.round(50.0 + np.random.normal(0, 0.08, len(dates)), 3)
    })
    return df


@st.cache_data(show_spinner=False)
def create_synthetic_medical_xray(condition: str = "Normal") -> Image.Image:
    """Generates realistic synthetic chest radiographic scans for demonstration."""
    width, height = 320, 320
    img = Image.new("L", (width, height), color=15)
    draw = ImageDraw.Draw(img)
    
    # Draw ribcage outline & spine
    draw.ellipse([40, 20, 280, 310], outline=55, width=4)
    draw.line([160, 20, 160, 310], fill=90, width=12) # Spine column
    for y in range(60, 260, 28):
        draw.arc([55, y, 160, y + 40], start=180, end=360, fill=70, width=3)
        draw.arc([160, y, 265, y + 40], start=180, end=360, fill=70, width=3)

    # Left & Right lung fields (darker air radiolucency)
    draw.ellipse([65, 60, 145, 250], fill=30)
    draw.ellipse([175, 60, 255, 250], fill=30)
    
    # Cardiac silhouette (heart shadow in medial-lower left field)
    draw.ellipse([135, 140, 215, 245], fill=110)
    
    # Diaphragm domes
    draw.chord([50, 230, 155, 310], start=180, end=360, fill=90)
    draw.chord([165, 230, 270, 310], start=180, end=360, fill=85)
    
    # Add pathology (consolidation opacity / infiltrate) if condition is Pneumonia
    if condition == "Pneumonia":
        for _ in range(8):
            cx = np.random.randint(85, 130)
            cy = np.random.randint(110, 190)
            r = np.random.randint(18, 40)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=165)
            
    # Apply realistic medical blur & texture
    img = img.filter(ImageFilter.GaussianBlur(radius=3.5))
    
    # Add realistic radiographic grain
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 7.5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def ensure_sample_assets():
    """Generates and caches sample assets to disk so everything is instant."""
    os.makedirs("assets/sample_data", exist_ok=True)
    os.makedirs("assets/sample_xrays", exist_ok=True)

    # Sample CSVs
    if not os.path.exists("assets/sample_data/healthcare_data.csv"):
        generate_sample_healthcare_data().to_csv("assets/sample_data/healthcare_data.csv", index=False)
        
    if not os.path.exists("assets/sample_data/fintech_transactions.csv"):
        generate_sample_fintech_transactions().to_csv("assets/sample_data/fintech_transactions.csv", index=False)
        
    if not os.path.exists("assets/sample_data/energy_timeseries.csv"):
        generate_sample_energy_data().to_csv("assets/sample_data/energy_timeseries.csv", index=False)
        
    if not os.path.exists("assets/sample_data/ecommerce_rfm.csv"):
        rfm, reviews = generate_sample_ecommerce_data()
        rfm.to_csv("assets/sample_data/ecommerce_rfm.csv", index=False)
        reviews.to_csv("assets/sample_data/ecommerce_reviews.csv", index=False)

    # Sample X-Ray images
    normal_path = "assets/sample_xrays/chest_xray_normal.png"
    pneumonia_path = "assets/sample_xrays/chest_xray_pneumonia.png"
    if not os.path.exists(normal_path):
        create_synthetic_medical_xray("Normal").save(normal_path)
    if not os.path.exists(pneumonia_path):
        create_synthetic_medical_xray("Pneumonia").save(pneumonia_path)
