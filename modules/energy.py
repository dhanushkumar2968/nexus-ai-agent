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
from datetime import timedelta

from core.styling import render_metric_card, render_domain_banner
from core.utils import (
    style_plotly_figure, 
    generate_sample_energy_data,
    ensure_sample_assets
)

def render_energy_module():
    """Main renderer for Smart Energy & Environmental Grid Forecasting domain."""
    ensure_sample_assets()

    render_domain_banner(
        title="Smart Energy & Environmental Grid",
        subtitle="Time-Series Load Forecasting, Peak Demand Alerting, and Environmental AQI Multi-Variable Dynamics",
        domain_type="energy",
        badge_text="Time-Series ML & IoT Forecasting"
    )

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Peak Grid Demand", "482.4 MW", "Safety margin: +18%", "warning")
    with col2:
        render_metric_card("Current AQI", "74 (Moderate)", "PM2.5: 22.4 µg/m³", "primary")
    with col3:
        render_metric_card("Renewable Penetration", "31.8%", "Solar + Wind Contribution", "success")
    with col4:
        render_metric_card("Grid Frequency", "50.02 Hz", "Tolerance: ±0.20 Hz", "cyan")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_forecast, tab_aqi, tab_renewables = st.tabs([
        "📈 Smart Grid Load Forecasting",
        "🍃 Environmental AQI & Pollution Dynamics",
        "☀️ Renewable Mix & Frequency Stability"
    ])

    # Load baseline dataset
    df_energy = generate_sample_energy_data(n_days=60)

    # ----------------------------------------------------
    # TAB 1: Smart Grid Load Forecasting
    # ----------------------------------------------------
    with tab_forecast:
        st.markdown("#### Municipal Power Load Forecasting Horizon")
        st.markdown(
            "Predict electricity demand using historical consumption patterns, diurnal cycles, and temperature dependencies. "
            "Simulates dynamic load response to external meteorological stressors."
        )

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
        with col_ctrl1:
            horizon_days = st.slider("Forecast Horizon (Days):", min_value=3, max_value=30, value=7, step=1)
        with col_ctrl2:
            temp_delta = st.slider("Simulated Temp Delta (°C):", min_value=-10, max_value=12, value=0, step=1, 
                                   help="Adjust scenario temperature above/below seasonal baseline to simulate heatwave or cold snap.")
        with col_ctrl3:
            peak_threshold = st.number_input("Peak Load Alert Threshold (MW):", min_value=350, max_value=600, value=440, step=10)

        # Generate future forecast points
        last_timestamp = df_energy["Timestamp"].iloc[-1]
        future_hours = horizon_days * 24
        future_dates = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=future_hours, freq="h")
        
        f_hours = future_dates.hour
        f_dayofweek = future_dates.dayofweek
        
        # Diurnal pattern
        f_diurnal = 35 * np.sin((f_hours - 6) * np.pi / 12) + 20 * np.sin((f_hours - 17) * np.pi / 6)
        f_weekend = np.where(f_dayofweek >= 5, -15, 10)
        
        # Base temperature with user offset
        f_temp = 22 + temp_delta + 6 * np.sin(np.linspace(0, 2 * np.pi, future_hours)) + np.random.normal(0, 2, future_hours)
        f_cooling_heating = np.abs(f_temp - 20) * 4.5
        
        f_demand = 280 + f_diurnal + f_weekend + f_cooling_heating + np.random.normal(0, 5, future_hours)
        f_demand = np.clip(f_demand, 190, 550).round(2)
        
        # Confidence bands (95% CI)
        ci_spread = np.linspace(12, 28, future_hours)
        f_upper = f_demand + ci_spread
        f_lower = f_demand - ci_spread

        df_future = pd.DataFrame({
            "Timestamp": future_dates,
            "Forecast_Demand_MW": f_demand,
            "Upper_CI_95": f_upper,
            "Lower_CI_95": f_lower,
            "Simulated_Temp_C": f_temp.round(1)
        })

        # Chart: Past 7 days + Future Forecast
        recent_history = df_energy.iloc[-168:] # Last 7 days

        fig_load = go.Figure()

        # Historical line
        fig_load.add_trace(go.Scatter(
            x=recent_history["Timestamp"],
            y=recent_history["Energy_Demand_MW"],
            mode="lines",
            name="Observed History",
            line=dict(color="#38bdf8", width=2.5)
        ))

        # Upper CI
        fig_load.add_trace(go.Scatter(
            x=df_future["Timestamp"],
            y=df_future["Upper_CI_95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Lower CI with fill
        fig_load.add_trace(go.Scatter(
            x=df_future["Timestamp"],
            y=df_future["Lower_CI_95"],
            mode="lines",
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(129, 140, 248, 0.18)',
            name='95% Prediction Interval'
        ))

        # Forecast line
        fig_load.add_trace(go.Scatter(
            x=df_future["Timestamp"],
            y=df_future["Forecast_Demand_MW"],
            mode="lines",
            name="ML Model Forecast",
            line=dict(color="#a78bfa", width=2.5, dash="dash")
        ))

        # Threshold line
        fig_load.add_hline(
            y=peak_threshold,
            line_dash="dot",
            line_color="#f87171",
            annotation_text=f"Grid Strain Alert ({peak_threshold} MW)",
            annotation_position="bottom right"
        )

        fig_load.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Power Demand (Megawatts - MW)",
            legend=dict(orientation="h", y=-0.18)
        )
        style_plotly_figure(fig_load, f"Smart Grid Load Projection ({horizon_days}-Day Outlook, Temp Offset: {temp_delta:+}°C)")
        st.plotly_chart(fig_load, use_container_width=True)

        # Risk Alert if forecast exceeds threshold
        violations = df_future[df_future["Forecast_Demand_MW"] > peak_threshold]
        if not violations.empty:
            max_viol = violations["Forecast_Demand_MW"].max()
            first_viol = violations["Timestamp"].iloc[0].strftime('%Y-%m-%d %H:%M')
            st.error(f"⚠️ **Grid Load Warning:** Projected peak load of **{max_viol:.1f} MW** exceeds your safety limit of {peak_threshold} MW! First expected breach: **{first_viol}**. Consider activating reserve peaking turbines or initiating demand-response shedding.")
        else:
            st.success("✅ **Grid Stability Verified:** Projected peak load remains within safe operating bounds under this meteorological scenario.")

    # ----------------------------------------------------
    # TAB 2: Environmental AQI & Pollution Dynamics
    # ----------------------------------------------------
    with tab_aqi:
        st.markdown("#### Ambient Air Quality Index (AQI) & Multi-Variable Dynamics")
        st.markdown(
            "Track atmospheric particulate trends and inspect correlation between municipal energy generation, "
            "commute rush-hour intervals, and regional air pollution levels."
        )

        recent_aqi = df_energy.iloc[-336:] # 14 days

        col_a1, col_a2 = st.columns([1.8, 1.2])

        with col_a1:
            fig_aqi = go.Figure()
            fig_aqi.add_trace(go.Scatter(
                x=recent_aqi["Timestamp"],
                y=recent_aqi["AQI"],
                mode="lines",
                name="Monitored AQI",
                line=dict(color="#34d399", width=2)
            ))

            # AQI Threshold bands
            fig_aqi.add_hrect(y0=0, y1=50, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, annotation_text="Good (0-50)")
            fig_aqi.add_hrect(y0=51, y1=100, fillcolor="rgba(245, 158, 11, 0.08)", line_width=0, annotation_text="Moderate (51-100)")
            fig_aqi.add_hrect(y0=101, y1=150, fillcolor="rgba(249, 115, 22, 0.08)", line_width=0, annotation_text="Sensitive Groups (101-150)")
            fig_aqi.add_hrect(y0=151, y1=300, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, annotation_text="Unhealthy (>150)")

            fig_aqi.update_layout(
                height=370,
                xaxis_title="Timeline",
                yaxis_title="Air Quality Index (AQI)"
            )
            style_plotly_figure(fig_aqi, "Air Quality Index Historical Progression & EPA Tiers")
            st.plotly_chart(fig_aqi, use_container_width=True)

        with col_a2:
            # Scatter: Temperature vs AQI colored by Demand
            fig_corr = px.scatter(
                recent_aqi,
                x="Temperature_C",
                y="AQI",
                color="Energy_Demand_MW",
                color_continuous_scale="Viridis",
                labels={"Temperature_C": "Ambient Temp (°C)", "Energy_Demand_MW": "Grid MW"}
            )
            fig_corr.update_layout(height=370)
            style_plotly_figure(fig_corr, "Thermal Inversion & Load vs AQI")
            st.plotly_chart(fig_corr, use_container_width=True)

        # Diurnal Hour vs Average AQI
        df_energy["Hour"] = df_energy["Timestamp"].dt.hour
        hourly_stats = df_energy.groupby("Hour")[["AQI", "Energy_Demand_MW"]].mean().reset_index()

        fig_diurnal = go.Figure()
        fig_diurnal.add_trace(go.Bar(
            x=hourly_stats["Hour"],
            y=hourly_stats["AQI"],
            name="Average AQI",
            marker_color="#fbbf24",
            opacity=0.75
        ))
        fig_diurnal.add_trace(go.Scatter(
            x=hourly_stats["Hour"],
            y=hourly_stats["Energy_Demand_MW"] / 4,
            name="Power Demand (Normalized)",
            mode="lines+markers",
            line=dict(color="#ec4899", width=2.5)
        ))
        fig_diurnal.update_layout(
            height=320,
            xaxis=dict(tickmode="linear", tick0=0, dtick=2, title="Hour of Day (00:00 - 23:00)"),
            yaxis_title="Metric Level",
            legend=dict(orientation="h", y=-0.2)
        )
        style_plotly_figure(fig_diurnal, "Diurnal Traffic & Energy Surge Correlation with Air Pollution")
        st.plotly_chart(fig_diurnal, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: Renewable Mix & Frequency Stability
    # ----------------------------------------------------
    with tab_renewables:
        st.markdown("#### Clean Energy Integration & Synchronization Metrics")
        st.markdown(
            "Evaluate grid decarbonization metrics, solar generation dispatch curves, and frequency deviation limits (50.00 Hz nominal)."
        )

        week_sample = df_energy.iloc[-168:]

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            fig_solar = go.Figure()
            fig_solar.add_trace(go.Scatter(
                x=week_sample["Timestamp"],
                y=week_sample["Energy_Demand_MW"],
                mode="lines",
                name="Gross Demand",
                line=dict(color="#94a3b8", width=1.8)
            ))
            fig_solar.add_trace(go.Scatter(
                x=week_sample["Timestamp"],
                y=week_sample["Renewable_Solar_MW"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(245, 158, 11, 0.25)",
                name="Solar Dispatch (MW)",
                line=dict(color="#f59e0b", width=2)
            ))
            fig_solar.update_layout(height=360, yaxis_title="Megawatts (MW)", legend=dict(orientation="h", y=-0.2))
            style_plotly_figure(fig_solar, "Solar Peak Dispatch vs. Gross Electricity Demand")
            st.plotly_chart(fig_solar, use_container_width=True)

        with col_r2:
            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(
                x=week_sample["Timestamp"],
                y=week_sample["Grid_Frequency_Hz"],
                mode="lines",
                name="Grid Frequency",
                line=dict(color="#06b6d4", width=1.5)
            ))
            fig_freq.add_hline(y=50.2, line_dash="dash", line_color="#f87171", annotation_text="Upper Limit (+0.2 Hz)")
            fig_freq.add_hline(y=49.8, line_dash="dash", line_color="#f87171", annotation_text="Lower Limit (-0.2 Hz)")
            fig_freq.add_hline(y=50.0, line_dash="dot", line_color="#10b981", annotation_text="Target (50.0 Hz)")
            
            fig_freq.update_layout(height=360, yaxis_title="Frequency (Hertz - Hz)", legend=dict(orientation="h", y=-0.2))
            style_plotly_figure(fig_freq, "Synchronous Grid Frequency Stability Monitor")
            st.plotly_chart(fig_freq, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="NexusAI - Smart Energy & Environment",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    from core.styling import apply_custom_styling
    apply_custom_styling()
    render_energy_module()
