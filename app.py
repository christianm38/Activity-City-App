"""
Stuttgart ActiveCity — Streamlit Prototyp
Bewegung & nachhaltige Mobilität → StuttPunkte
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

from models.activity_classifier import ActivityClassifier
from models.points_predictor import PointsPredictor
from utils.data_generator import generate_week_data, generate_mobility_data
from utils.points_engine import PointsEngine

# ─── Seitenkonfiguration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stuttgart ActiveCity",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Stuttgart-Rot + cleanes Layout) ─────────────────────────────
st.markdown("""
<style>
:root { --stuttgart-red: #E8312A; }

.stApp { background: #f5f5f5; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #eee;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #E8312A;
    line-height: 1.1;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #888;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-pill {
    display: inline-block;
    background: #fff0f0;
    color: #E8312A;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
    border: 1px solid #fcc;
}
.green-pill {
    background: #eaf6ef;
    color: #2e7d32;
    border-color: #a5d6a7;
}
.ml-box {
    background: linear-gradient(135deg, #fff0f0 0%, #fff8f8 100%);
    border: 1.5px solid #fcc;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.header-banner {
    background: #E8312A;
    color: white;
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State initialisieren ────────────────────────────────────────────
if "points" not in st.session_state:
    st.session_state.points = 1240
if "week_data" not in st.session_state:
    st.session_state.week_data = generate_week_data()
if "mobility_log" not in st.session_state:
    st.session_state.mobility_log = generate_mobility_data()
if "ml_classifier" not in st.session_state:
    st.session_state.ml_classifier = ActivityClassifier()
if "points_predictor" not in st.session_state:
    st.session_state.points_predictor = PointsPredictor()
    st.session_state.points_predictor.fit(st.session_state.week_data)

engine = PointsEngine()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏃 Stuttgart ActiveCity")
    st.markdown("---")

    # Profil
    st.markdown("**Dein Profil**")
    name = st.text_input("Name", value="Christian M.", label_visibility="collapsed")
    level = engine.get_level(st.session_state.points)
    st.markdown(
        f"<span class='badge-pill'>Level {level['level']} — {level['title']}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{st.session_state.points:,} StuttPunkte**".replace(",", "."))

    progress = (st.session_state.points - level["min"]) / (level["next"] - level["min"])
    st.progress(min(progress, 1.0), text=f"→ Level {level['level']+1}: {level['next']:,} Pkte".replace(",","."))

    st.markdown("---")

    # Health-Sync (simuliert)
    st.markdown("**Health-Sync**")
    sync_option = st.selectbox(
        "Datenquelle", ["Apple Health", "Google Fit", "Garmin Connect", "Fitbit"]
    )
    st.success(f"✅ Verbunden — {sync_option}")
    st.caption("Letzte Sync: heute 06:32 Uhr")

    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio(
        "Seite",
        ["🏠 Dashboard", "🚶 Aktivität loggen", "🗺️ Mobilität & ÖPNV", "🤖 ML-Analyse", "🎁 Belohnungen"],
        label_visibility="collapsed",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown(
        f"""<div class='header-banner'>
        <h2 style='margin:0;color:white;'>Guten Morgen, {name.split()[0]} 👋</h2>
        <p style='margin:4px 0 0;opacity:0.85;font-size:0.9rem;'>
            Stuttgart ActiveCity — nachhaltige Mobilität wird belohnt
        </p></div>""",
        unsafe_allow_html=True,
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    today = st.session_state.week_data.iloc[-1]
    with c1:
        st.markdown(f"<div class='metric-card'><div class='value'>{st.session_state.points:,}</div><div class='label'>StuttPunkte gesamt</div></div>".replace(",","."), unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='value'>{int(today['steps']):,}</div><div class='label'>Schritte heute</div></div>".replace(",","."), unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='value'>{today['points_earned']:.0f}</div><div class='label'>Punkte heute</div></div>", unsafe_allow_html=True)
    with c4:
        co2 = st.session_state.mobility_log["co2_saved_kg"].sum()
        st.markdown(f"<div class='metric-card'><div class='value'>{co2:.1f} kg</div><div class='label'>CO₂ eingespart</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Punkte — letzte 7 Tage")
        fig = px.bar(
            st.session_state.week_data,
            x="day",
            y="points_earned",
            color="points_earned",
            color_continuous_scale=["#fcc", "#E8312A"],
            labels={"points_earned": "Punkte", "day": ""},
            text="points_earned",
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=20, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Mobilitätsmix")
        mob = st.session_state.mobility_log.groupby("mode")["km"].sum().reset_index()
        fig2 = px.pie(
            mob, values="km", names="mode",
            color_discrete_map={
                "Zu Fuß": "#E8312A", "Fahrrad": "#4a90d9",
                "ÖPNV": "#3a9e64", "E-Scooter": "#f5a623", "Auto": "#b0b0b0",
            },
            hole=0.55,
        )
        fig2.update_layout(
            margin=dict(t=20, b=10),
            height=280,
            legend=dict(orientation="v", x=1.0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ML-Vorhersage als Highlight
    st.markdown("---")
    st.subheader("🤖 ML-Prognose: morgen")
    pred = st.session_state.points_predictor.predict_tomorrow(st.session_state.week_data)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='ml-box'><b>Erwartete Punkte</b><br>"
            f"<span style='font-size:1.8rem;color:#E8312A;font-weight:700;'>{pred['points']:.0f}</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='ml-box'><b>Empfohlene Aktivität</b><br>"
            f"<span style='font-size:1.1rem;font-weight:600;'>{pred['recommended_activity']}</span></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='ml-box'><b>Optimale Abfahrtszeit</b><br>"
            f"<span style='font-size:1.1rem;font-weight:600;'>{pred['best_departure']}</span></div>",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AKTIVITÄT LOGGEN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚶 Aktivität loggen":
    st.header("Aktivität loggen")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Manuelle Eingabe")

        activity_type = st.selectbox(
            "Aktivitätstyp",
            ["Laufen", "Walken", "Fahrradfahren", "Schwimmen", "Fußball/Mannschaftssport",
             "Yoga/Fitness", "Klettern", "Inline-Skating"],
        )
        duration_min = st.slider("Dauer (Minuten)", 5, 180, 30, step=5)
        intensity = st.select_slider(
            "Intensität", options=["Locker", "Moderat", "Intensiv"], value="Moderat"
        )
        heart_rate = st.number_input("Ø Herzfrequenz (bpm)", min_value=60, max_value=200, value=130)

        # ML-Klassifikation der Aktivität
        classifier = st.session_state.ml_classifier
        features = {
            "duration_min": duration_min,
            "heart_rate": heart_rate,
            "intensity": intensity,
        }
        ml_result = classifier.classify(features)

        st.markdown(
            f"<div class='ml-box'>🤖 <b>ML-Erkennung:</b> "
            f"Erkannte Aktivität: <b>{ml_result['activity']}</b> "
            f"(Konfidenz: {ml_result['confidence']:.0%})</div>",
            unsafe_allow_html=True,
        )

        pts_preview = engine.calculate_points(activity_type, duration_min, intensity)
        st.info(f"💰 Vorschau: **+{pts_preview} StuttPunkte** für diese Aktivität")

        if st.button("✅ Aktivität speichern", type="primary", use_container_width=True):
            st.session_state.points += pts_preview
            st.success(f"🎉 +{pts_preview} Punkte gutgeschrieben! Gesamt: {st.session_state.points:,} Pkte".replace(",","."))
            st.balloons()

    with col2:
        st.subheader("Punkterechner")
        st.markdown("Vergleich verschiedener Aktivitäten bei **30 Minuten**:")

        activities = {
            "Laufen": engine.calculate_points("Laufen", 30, "Moderat"),
            "Fahrradfahren": engine.calculate_points("Fahrradfahren", 30, "Moderat"),
            "Schwimmen": engine.calculate_points("Schwimmen", 30, "Moderat"),
            "Walken": engine.calculate_points("Walken", 30, "Moderat"),
            "Fußball": engine.calculate_points("Fußball/Mannschaftssport", 30, "Intensiv"),
            "Yoga": engine.calculate_points("Yoga/Fitness", 30, "Locker"),
        }
        df_acts = pd.DataFrame(
            list(activities.items()), columns=["Aktivität", "Punkte"]
        ).sort_values("Punkte", ascending=True)

        fig = px.bar(
            df_acts, x="Punkte", y="Aktivität", orientation="h",
            color="Punkte",
            color_continuous_scale=["#fcc", "#E8312A"],
            text="Punkte",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=10, b=10),
            height=320,
            xaxis_title="StuttPunkte",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Punkte steigen mit Dauer und Intensität. "
            "Bonuspunkte für 7-Tage-Streak (+20%) und CO₂-Ersparnis."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MOBILITÄT & ÖPNV
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Mobilität & ÖPNV":
    st.header("Mobilität & ÖPNV-Tracking")
    st.caption("Alle nachhaltigen Verkehrsmittel werden belohnt — nur das Auto nicht.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Fahrt erfassen")

        mode = st.selectbox(
            "Verkehrsmittel",
            ["🚶 Zu Fuß", "🚲 Fahrrad", "🚋 Straßenbahn (VVS)", "🚌 Bus (VVS)",
             "🚆 S-Bahn (VVS)", "🛴 E-Scooter", "🚗 Auto (keine Punkte)"],
        )
        km = st.slider("Strecke (km)", 0.5, 50.0, 5.0, step=0.5)
        departure = st.time_input("Abfahrtszeit", value=datetime.now().replace(hour=8, minute=0, second=0))

        # Punkte berechnen
        mob_pts = engine.calculate_mobility_points(mode, km)
        co2_saved = engine.calculate_co2(mode, km)

        if "Auto" in mode:
            st.warning("🚗 Autofahrten werden nicht belohnt. Wechsle auf ÖPNV oder Rad!")
            mob_pts = 0
        else:
            st.info(
                f"💰 **+{mob_pts} StuttPunkte** | "
                f"🌱 **{co2_saved:.2f} kg CO₂** eingespart"
            )

        if st.button("🗺️ Fahrt speichern", type="primary", use_container_width=True):
            if mob_pts > 0:
                st.session_state.points += mob_pts
                new_row = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "mode": mode.split(" ", 1)[1],
                    "km": km,
                    "points": mob_pts,
                    "co2_saved_kg": co2_saved,
                    "departure": str(departure),
                }
                st.session_state.mobility_log = pd.concat(
                    [st.session_state.mobility_log, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                st.success(f"✅ Fahrt gespeichert — +{mob_pts} Punkte!")
            else:
                st.info("Keine Punkte für Autofahrten.")

    with col2:
        st.subheader("Punktetabelle Mobilität")

        mobil_table = pd.DataFrame([
            {"Verkehrsmittel": "🚶 Zu Fuß", "Punkte/km": 4, "Bonus": "Tagesziel +20"},
            {"Verkehrsmittel": "🚲 Fahrrad", "Punkte/km": 5, "Bonus": "Strecken-Bonus"},
            {"Verkehrsmittel": "🚋 Straßenbahn", "Punkte/km": 3, "Bonus": "Taktik-Bonus"},
            {"Verkehrsmittel": "🚌 Bus", "Punkte/km": 3, "Bonus": "—"},
            {"Verkehrsmittel": "🚆 S-Bahn", "Punkte/km": 3, "Bonus": "Pendler-Bonus"},
            {"Verkehrsmittel": "🛴 E-Scooter", "Punkte/km": 2, "Bonus": "—"},
            {"Verkehrsmittel": "🚗 Auto", "Punkte/km": 0, "Bonus": "—"},
        ])
        st.dataframe(mobil_table, use_container_width=True, hide_index=True)

        st.subheader("VVS Live-Abfahrten (Demo)")
        st.markdown(
            """
| Linie | Richtung | Abfahrt | Punkte |
|-------|----------|---------|--------|
| U1 | Fellbach | 3 Min | +15 |
| S1 | Herrenberg | 5 Min | +21 |
| 42 | Botnang | 7 Min | +9 |
| U14 | Remseck | 12 Min | +18 |
"""
        )
        st.caption("In der Live-Version via VVS-API (TRIAS) befüllt.")

    st.markdown("---")

    st.subheader("Mobilitätsverlauf — letzte 7 Tage")
    mob_by_day = (
        st.session_state.mobility_log
        .groupby(["date", "mode"])["km"]
        .sum()
        .reset_index()
    )
    fig3 = px.bar(
        mob_by_day, x="date", y="km", color="mode",
        color_discrete_map={
            "Zu Fuß": "#E8312A", "Fahrrad": "#4a90d9", "Straßenbahn (VVS)": "#3a9e64",
            "Bus (VVS)": "#2e7d32", "S-Bahn (VVS)": "#1b5e20",
            "E-Scooter": "#f5a623", "Auto": "#b0b0b0",
        },
        labels={"km": "Kilometer", "date": "Datum", "mode": "Verkehrsmittel"},
        barmode="stack",
    )
    fig3.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=10), height=300
    )
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ML-ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML-Analyse":
    st.header("🤖 Machine Learning — Analyse & Empfehlungen")
    st.markdown(
        "Zwei ML-Modelle arbeiten im Hintergrund: "
        "**Aktivitätserkennung** (Random Forest Klassifikation) und "
        "**Punkteprognose** (Gradient Boosting Regression)."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Aktivitätserkennung", "Punkteprognose", "Nudging & Empfehlungen"]
    )

    with tab1:
        st.subheader("Aktivitätserkennung via Sensordaten")
        st.markdown(
            "Ein **Random-Forest-Klassifikator** erkennt deine Aktivität anhand von "
            "Herzfrequenz, Schritttempo und Bewegungsintensität — wie ein Wearable."
        )

        c1, c2 = st.columns(2)
        with c1:
            hr = st.slider("Herzfrequenz (bpm)", 55, 195, 145)
            cadence = st.slider("Schrittfrequenz (Schritte/Min)", 0, 200, 160)
            accel = st.slider("Beschleunigung (m/s²)", 0.0, 20.0, 8.5)

        with c2:
            features = {
                "duration_min": 30,
                "heart_rate": hr,
                "intensity": "Moderat",
                "cadence": cadence,
                "acceleration": accel,
            }
            result = st.session_state.ml_classifier.classify_sensor(features)

            st.markdown("**Erkannte Aktivität:**")
            for act, prob in result["probabilities"].items():
                bar_color = "#E8312A" if act == result["activity"] else "#eee"
                st.markdown(
                    f"<div style='margin:4px 0;'>"
                    f"<span style='width:160px;display:inline-block;font-size:13px;'>{act}</span>"
                    f"<div style='display:inline-block;width:{prob*200:.0f}px;height:14px;"
                    f"background:{bar_color};border-radius:3px;vertical-align:middle;'></div>"
                    f" <span style='font-size:12px;color:#666;'>{prob:.0%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div class='ml-box' style='margin-top:12px;'>✅ <b>Bestes Label:</b> "
                f"{result['activity']} ({result['confidence']:.0%} Konfidenz)</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(
            "📌 In der Produktion wird das Modell auf echten Wearable-Daten trainiert "
            "(Apple Watch / Fitbit Export). Hier: synthetische Trainingsdaten zur Demo."
        )

    with tab2:
        st.subheader("Punkteprognose — Gradient Boosting")
        st.markdown(
            "Ein **Gradient-Boosting-Regressionsmodell** prognostiziert deine "
            "erreichbaren Punkte der nächsten 7 Tage anhand deines Verlaufs."
        )

        predictor = st.session_state.points_predictor
        forecast = predictor.forecast_week(st.session_state.week_data)

        fig4 = go.Figure()
        hist = st.session_state.week_data
        fig4.add_trace(go.Scatter(
            x=hist["day"], y=hist["points_earned"],
            mode="lines+markers", name="Historisch",
            line=dict(color="#E8312A", width=2),
            marker=dict(size=8),
        ))
        fig4.add_trace(go.Scatter(
            x=forecast["day"], y=forecast["predicted"],
            mode="lines+markers", name="Prognose (ML)",
            line=dict(color="#4a90d9", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))
        fig4.add_trace(go.Scatter(
            x=list(forecast["day"]) + list(forecast["day"])[::-1],
            y=list(forecast["upper"]) + list(forecast["lower"])[::-1],
            fill="toself", fillcolor="rgba(74,144,217,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Konfidenzband",
        ))
        fig4.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=320, margin=dict(t=10),
            xaxis_title="Tag", yaxis_title="Punkte",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig4, use_container_width=True)

        total_forecast = forecast["predicted"].sum()
        st.info(
            f"📈 Prognose nächste Woche: **{total_forecast:.0f} Punkte** "
            f"(+{total_forecast/st.session_state.week_data['points_earned'].sum()*100-100:.1f}% ggü. Vorwoche)"
        )

    with tab3:
        st.subheader("Personalisiertes Nudging")
        st.markdown(
            "Basierend auf deinem Verhaltensmuster identifiziert das Modell "
            "**Interventionsfenster** — Uhrzeiten, zu denen ein Push-Hinweis "
            "die Wahrscheinlichkeit nachhaltiger Mobilität am stärksten erhöht."
        )

        hours = list(range(6, 22))
        nudge_scores = [
            0.1, 0.3, 0.7, 0.85, 0.6, 0.4, 0.3, 0.25, 0.35, 0.55,
            0.65, 0.8, 0.9, 0.75, 0.5, 0.3,
        ]
        df_nudge = pd.DataFrame({"Uhrzeit": [f"{h}:00" for h in hours], "Nudge-Score": nudge_scores})

        fig5 = px.area(
            df_nudge, x="Uhrzeit", y="Nudge-Score",
            color_discrete_sequence=["#E8312A"],
            labels={"Nudge-Score": "Interventionswahrscheinlichkeit"},
        )
        fig5.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=260, margin=dict(t=10),
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("**Optimale Interventionszeitpunkte heute:**")
        cols = st.columns(3)
        for i, (time, msg) in enumerate([
            ("07:45", "S-Bahn statt Auto → +21 Pkte"),
            ("12:15", "Mittagsspaziergang 20 Min → +16 Pkte"),
            ("18:00", "Mit dem Rad nach Hause → +30 Pkte"),
        ]):
            with cols[i]:
                st.markdown(
                    f"<div class='ml-box'><b>{time} Uhr</b><br>{msg}</div>",
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BELOHNUNGEN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎁 Belohnungen":
    st.header("Belohnungen einlösen")
    st.markdown(f"Dein Guthaben: **{st.session_state.points:,} StuttPunkte**".replace(",","."))

    rewards = [
        {"name": "VVS-Tageskarte Zone 1+2", "partner": "VVS", "cost": 1500, "icon": "🚋", "cat": "Mobilität"},
        {"name": "Wilhelma-Eintritt", "partner": "Wilhelma Stuttgart", "cost": 800, "icon": "🦒", "cat": "Freizeit"},
        {"name": "Staatsgalerie freier Eintritt", "partner": "Staatsgalerie", "cost": 600, "icon": "🖼️", "cat": "Kultur"},
        {"name": "Stadtbibliothek Kaffee", "partner": "Stadtbibliothek", "cost": 200, "icon": "☕", "cat": "Gastronomie"},
        {"name": "StuttCard Plus 1 Tag", "partner": "Stuttgart Tourist", "cost": 1000, "icon": "🗺️", "cat": "Mobilität"},
        {"name": "Mercedes-Benz Museum", "partner": "Mercedes-Benz", "cost": 700, "icon": "🏎️", "cat": "Kultur"},
        {"name": "Leihrad callabike — 1 Stunde", "partner": "DB Connect", "cost": 150, "icon": "🚲", "cat": "Mobilität"},
        {"name": "Biergarten-Gutschein", "partner": "Stuttgarter Weindorf", "cost": 400, "icon": "🍺", "cat": "Gastronomie"},
    ]

    cat_filter = st.multiselect(
        "Kategorie filtern",
        ["Mobilität", "Freizeit", "Kultur", "Gastronomie"],
        default=["Mobilität", "Freizeit", "Kultur", "Gastronomie"],
    )

    for r in rewards:
        if r["cat"] not in cat_filter:
            continue
        affordable = st.session_state.points >= r["cost"]
        border = "2px solid #E8312A" if affordable else "1px solid #eee"
        opacity = "1" if affordable else "0.5"

        with st.container():
            c1, c2, c3 = st.columns([0.6, 3, 1.5])
            with c1:
                st.markdown(f"<span style='font-size:2rem;'>{r['icon']}</span>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{r['name']}**")
                st.caption(f"{r['partner']} · {r['cat']}")
            with c3:
                if affordable:
                    if st.button(f"Einlösen — {r['cost']} Pkte", key=r["name"], type="primary"):
                        st.session_state.points -= r["cost"]
                        st.success(f"✅ {r['name']} eingelöst!")
                        st.rerun()
                else:
                    st.markdown(
                        f"<span style='color:#999;font-size:0.85rem;'>"
                        f"Noch {r['cost'] - st.session_state.points} Pkte fehlen</span>",
                        unsafe_allow_html=True,
                    )
            st.divider()
