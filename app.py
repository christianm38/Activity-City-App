"""
Stuttgart ActiveCity v3 — dunkles Design, kontrastreiche Farben, Stuttgart-Karte
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from models.activity_classifier import ActivityClassifier
from models.points_predictor import PointsPredictor
from utils.data_generator import generate_week_data, generate_mobility_data
from utils.points_engine import PointsEngine

st.set_page_config(
    page_title="Stuttgart ActiveCity",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Farben ────────────────────────────────────────────────────────────────────
R   = "#E8312A"   # Stuttgart-Rot
RL  = "#ff6b6b"   # Hellrot (Text auf Dunkel)
BG  = "#0f1117"   # App-Hintergrund (fast schwarz)
S1  = "#1c1f2e"   # Card-Hintergrund
S2  = "#252840"   # Card-Hintergrund 2
BRD = "#2e3250"   # Rahmen
TXT = "#f0f2ff"   # Haupttext (sehr helles Blaugrau)
MUT = "#8b92b8"   # Gedämpfter Text
GRN = "#4ade80"   # Grün
BLU = "#60a5fa"   # Blau
AMB = "#fbbf24"   # Amber

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG} !important;
    color: {TXT} !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {S1} !important;
    border-right: 1px solid {BRD} !important;
}}
section[data-testid="stSidebar"] * {{ color: {TXT} !important; }}
section[data-testid="stSidebar"] .stRadio label {{ color: {MUT} !important; font-size:11px; text-transform:uppercase; letter-spacing:1px; }}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{ color: {TXT} !important; font-size:14px; text-transform:none; letter-spacing:0; }}
section[data-testid="stSidebar"] .stSelectbox label {{ color: {MUT} !important; font-size:11px; }}
section[data-testid="stSidebar"] .stTextInput label {{ color: {MUT} !important; font-size:11px; }}

/* Alle Widgets auf dunklem Grund */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div {{
    background-color: {S2} !important;
    border-color: {BRD} !important;
    color: {TXT} !important;
}}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{
    color: {TXT} !important;
    background-color: transparent !important;
}}
div[data-baseweb="popover"] ul {{
    background-color: {S2} !important;
    border: 1px solid {BRD} !important;
}}
div[data-baseweb="popover"] li:hover {{ background-color: {BRD} !important; }}
div[data-baseweb="popover"] li span {{ color: {TXT} !important; }}

input, textarea {{
    background-color: {S2} !important;
    color: {TXT} !important;
    border-color: {BRD} !important;
}}

/* Slider */
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{
    background-color: {R} !important;
    border-color: {R} !important;
}}
div[data-testid="stSlider"] div[data-baseweb="slider"] div:first-child div:first-child {{
    background-color: {R} !important;
}}

/* Buttons */
button[kind="primary"], .stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {R}, #c0201a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 4px 15px rgba(232,49,42,0.35) !important;
}}
button[kind="primary"]:hover {{ transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(232,49,42,0.5) !important; }}
.stButton > button:not([kind="primary"]) {{
    background: {S2} !important; color: {TXT} !important;
    border: 1px solid {BRD} !important; border-radius: 10px !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    background: {S1} !important; color: {MUT} !important;
    border-radius: 8px 8px 0 0 !important; font-size: 13px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background: {R} !important; color: white !important; font-weight: 600 !important;
}}
div[data-baseweb="tab-panel"] {{
    background: {S1} !important; border: 1px solid {BRD} !important;
    border-radius: 0 8px 8px 8px !important; padding: 1.2rem !important;
}}

/* Progress */
div[data-testid="stProgressBar"] > div {{ background: {BRD} !important; border-radius: 10px; }}
div[data-testid="stProgressBar"] > div > div {{ background: linear-gradient(90deg, {R}, {RL}) !important; border-radius: 10px; }}

/* Alerts — eigene Farben */
div[data-testid="stAlert"] {{ border-radius: 10px !important; }}
div[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {{
    background: rgba(96,165,250,0.12) !important; border: 1px solid rgba(96,165,250,0.4) !important;
}}
div[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {{
    background: rgba(74,222,128,0.12) !important; border: 1px solid rgba(74,222,128,0.4) !important;
}}
div[data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {{
    background: rgba(251,191,36,0.12) !important; border: 1px solid rgba(251,191,36,0.4) !important;
}}
div[data-testid="stAlert"] p {{ color: {TXT} !important; }}

/* DataFrames */
div[data-testid="stDataFrameResizable"] {{ background: {S1} !important; border: 1px solid {BRD} !important; border-radius: 10px; }}
div[data-testid="stDataFrameResizable"] * {{ color: {TXT} !important; background-color: transparent !important; }}
div[class*="dvn-scroller"] {{ background: {S1} !important; }}

/* Selectbox dropdown arrow */
div[data-baseweb="select"] svg {{ fill: {MUT} !important; }}

/* Divider */
hr {{ border: none; border-top: 1px solid {BRD} !important; margin: 1rem 0; }}

/* Markdown text */
.stMarkdown p, .stMarkdown li, .element-container p {{ color: {TXT} !important; }}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 {{ color: {TXT} !important; }}
[data-testid="stCaptionContainer"] p {{ color: {MUT} !important; }}

/* Multiselect tags */
span[data-baseweb="tag"] {{ background: {R} !important; }}
span[data-baseweb="tag"] span {{ color: white !important; }}

/* Time input */
div[data-testid="stTimeInput"] input {{ background: {S2} !important; color: {TXT} !important; border: 1px solid {BRD} !important; }}

/* Headers */
h1,h2,h3 {{ color: {TXT} !important; }}
label {{ color: {MUT} !important; }}

/* Number input */
div[data-testid="stNumberInput"] input {{ background: {S2} !important; color: {TXT} !important; border: 1px solid {BRD} !important; border-radius: 8px !important; }}
div[data-testid="stNumberInput"] button {{ background: {S2} !important; color: {TXT} !important; border-color: {BRD} !important; }}

/* Select slider */
div[data-testid="stSlickSlider"] span {{ color: {TXT} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme helper ──────────────────────────────────────────────────
def dk(fig, h=300, legend_h=False):
    upd = dict(
        plot_bgcolor=S1, paper_bgcolor=S1,
        font=dict(color=TXT, family="Inter, sans-serif", size=12),
        height=h, margin=dict(t=24, b=8, l=8, r=8),
        xaxis=dict(gridcolor=BRD, zerolinecolor=BRD, tickfont=dict(color=MUT)),
        yaxis=dict(gridcolor=BRD, zerolinecolor=BRD, tickfont=dict(color=MUT)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TXT)),
    )
    if legend_h:
        upd["legend"] = dict(orientation="h", yanchor="bottom", y=1.02,
                              bgcolor="rgba(0,0,0,0)", font=dict(color=TXT))
    fig.update_layout(**upd)
    return fig

# ── HTML-Komponenten ──────────────────────────────────────────────────────────
def card(content): return f"<div style='background:{S1};border:1px solid {BRD};border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:.8rem;'>{content}</div>"
def metric(val, lbl): return f"<div style='background:{S1};border:1px solid {BRD};border-radius:12px;padding:1.2rem;text-align:center;'><div style='font-size:1.8rem;font-weight:700;color:{RL};line-height:1.1;'>{val}</div><div style='font-size:.72rem;color:{MUT};margin-top:5px;text-transform:uppercase;letter-spacing:.6px;'>{lbl}</div></div>"
def mlcard(label, val, sub=""): return f"<div style='background:{S2};border:1px solid {R};border-radius:12px;padding:1rem 1.2rem;margin:.4rem 0;'><div style='font-size:.7rem;color:{MUT};text-transform:uppercase;letter-spacing:.8px;font-weight:600;'>{label}</div><div style='font-size:1.4rem;font-weight:700;color:{RL};margin-top:4px;'>{val}</div>{f'<div style=font-size:.85rem;color:{TXT};margin-top:2px;>{sub}</div>' if sub else ''}</div>"
def sec(txt): return f"<p style='color:{MUT};font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin:1rem 0 .5rem;'>{txt}</p>"
def badge(txt, color=R): return f"<span style='background:rgba(232,49,42,.18);color:{RL};border:1px solid rgba(232,49,42,.4);border-radius:20px;padding:3px 11px;font-size:.75rem;font-weight:600;'>{txt}</span>"

# ── Session State ─────────────────────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{RL};margin:0 0 2px;'>Stuttgart ActiveCity</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};font-size:.8rem;margin:0 0 12px;'>Smart City Bewegungs-App</p>", unsafe_allow_html=True)
    st.divider()

    name  = st.text_input("Dein Name", value="Christian M.")
    level = engine.get_level(st.session_state.points)
    st.markdown(badge(f"Level {level['level']} — {level['title']}"), unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.4rem;font-weight:700;color:{RL};margin:8px 0 4px;'>{st.session_state.points:,} Pkte</p>".replace(",","."), unsafe_allow_html=True)
    prog = (st.session_state.points - level["min"]) / max(1, level["next"] - level["min"])
    st.progress(min(prog, 1.0), text=f"→ Level {level['level']+1}: {level['next']:,} Pkte".replace(",","."))

    st.divider()
    st.markdown(sec("Health-Sync"), unsafe_allow_html=True)
    sync = st.selectbox("Quelle", ["Apple Health","Google Fit","Garmin Connect","Fitbit"], label_visibility="collapsed")
    st.success(f" {sync} · heute 06:32")

    st.divider()
    st.markdown(sec("Navigation"), unsafe_allow_html=True)
    page = st.radio("nav", [
        "Dashboard", "Aktivitat loggen", "Mobilitat & OPNV",
        "Stuttgart-Karte", "ML-Analyse", "Belohnungen"
    ], label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#9b0000,{R});border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.5rem;border:1px solid rgba(255,255,255,.1);'>
        <h2 style='color:white;margin:0;font-size:1.6rem;'>Guten Morgen, {name.split()[0]}</h2>
        <p style='color:rgba(255,255,255,.8);margin:6px 0 0;font-size:.9rem;'>Stuttgart ActiveCity — nachhaltige Mobilität wird belohnt</p>
    </div>""", unsafe_allow_html=True)

    today = st.session_state.week_data.iloc[-1]
    co2   = st.session_state.mobility_log["co2_saved_kg"].sum()
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(metric(f"{st.session_state.points:,}".replace(",","."), "StuttPunkte gesamt"), unsafe_allow_html=True)
    c2.markdown(metric(f"{int(today['steps']):,}".replace(",","."), "Schritte heute"), unsafe_allow_html=True)
    c3.markdown(metric(f"{today['points_earned']:.0f}", "Punkte heute"), unsafe_allow_html=True)
    c4.markdown(metric(f"{co2:.1f} kg", "CO₂ eingespart"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cL, cR = st.columns([3,2])

    with cL:
        st.markdown(sec("Punkte — letzte 7 Tage"), unsafe_allow_html=True)
        fig = px.bar(st.session_state.week_data, x="day", y="points_earned",
                     color="points_earned",
                     color_continuous_scale=[[0,"#7f0000"],[.5,R],[1,RL]],
                     text="points_earned")
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                          textfont=dict(color=TXT, size=12))
        fig.update_coloraxes(showscale=False)
        dk(fig, 300)
        st.plotly_chart(fig, use_container_width=True)

    with cR:
        st.markdown(sec("Mobilitätsmix"), unsafe_allow_html=True)
        mob = st.session_state.mobility_log.groupby("mode")["km"].sum().reset_index()
        fig2 = px.pie(mob, values="km", names="mode", hole=.58,
            color_discrete_map={"Zu Fuß":R,"Fahrrad":BLU,"Straßenbahn (VVS)":GRN,
                                "Bus (VVS)":"#34d399","S-Bahn (VVS)":"#2dd4bf",
                                "E-Scooter":AMB,"Auto":"#475569"})
        fig2.update_traces(textfont=dict(color="white", size=12))
        fig2.update_layout(plot_bgcolor=S1, paper_bgcolor=S1, height=300,
                           margin=dict(t=10,b=10,l=10,r=10),
                           font=dict(color=TXT),
                           legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TXT)))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown(sec(" ML-Prognose für morgen"), unsafe_allow_html=True)
    pred = st.session_state.points_predictor.predict_tomorrow(st.session_state.week_data)
    c1,c2,c3 = st.columns(3)
    c1.markdown(mlcard("Erwartete Punkte", f"{pred['points']:.0f} Pkte"), unsafe_allow_html=True)
    c2.markdown(mlcard("Empfohlene Aktivität", pred['recommended_activity']), unsafe_allow_html=True)
    c3.markdown(mlcard("Optimale Abfahrt", f"{pred['best_departure']} Uhr"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  AKTIVITÄT LOGGEN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Aktivitat loggen":
    st.markdown(f"<h2 style='color:{TXT};'>Aktivität loggen</h2>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        act  = st.selectbox("Aktivitätstyp", ["Laufen","Walken","Fahrradfahren","Schwimmen",
                             "Fußball/Mannschaftssport","Yoga/Fitness","Klettern","Inline-Skating"])
        dur  = st.slider("Dauer (Minuten)", 5, 180, 30, step=5)
        intens = st.select_slider("Intensität", ["Locker","Moderat","Intensiv"], "Moderat")
        hr   = st.number_input("Ø Herzfrequenz (bpm)", 60, 200, 130)

        ml_r = st.session_state.ml_classifier.classify({"duration_min":dur,"heart_rate":hr,"intensity":intens})
        pts  = engine.calculate_points(act, dur, intens)

        st.markdown(mlcard(" ML-Erkennung", ml_r['activity'], f"Konfidenz: {ml_r['confidence']:.0%}"), unsafe_allow_html=True)
        st.info(f"Vorschau: **+{pts} StuttPunkte**")
        if st.button("Aktivitat speichern", type="primary", use_container_width=True):
            st.session_state.points += pts
            st.success(f" +{pts} Punkte! Gesamt: {st.session_state.points:,} Pkte".replace(",","."))
            st.balloons()

    with c2:
        st.markdown(sec("Punktevergleich bei 30 Minuten"), unsafe_allow_html=True)
        acts_d = {
            "Laufen":    engine.calculate_points("Laufen",30,"Moderat"),
            "Fahrrad":   engine.calculate_points("Fahrradfahren",30,"Moderat"),
            "Schwimmen": engine.calculate_points("Schwimmen",30,"Moderat"),
            "Walken":    engine.calculate_points("Walken",30,"Moderat"),
            "Fußball":   engine.calculate_points("Fußball/Mannschaftssport",30,"Intensiv"),
            "Yoga":      engine.calculate_points("Yoga/Fitness",30,"Locker"),
        }
        df_a = pd.DataFrame(list(acts_d.items()),columns=["Aktivität","Punkte"]).sort_values("Punkte")
        fig3 = px.bar(df_a,x="Punkte",y="Aktivität",orientation="h",
                      color="Punkte",color_continuous_scale=[[0,"#7f0000"],[1,RL]],text="Punkte")
        fig3.update_traces(textposition="outside",textfont=dict(color=TXT))
        fig3.update_coloraxes(showscale=False)
        dk(fig3, 340)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MOBILITÄT & ÖPNV
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Mobilitat & OPNV":
    st.markdown(f"<h2 style='color:{TXT};'>Mobilität & ÖPNV-Tracking</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};'>Alle nachhaltigen Verkehrsmittel werden belohnt — nur das Auto nicht.</p>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        mode = st.selectbox("Verkehrsmittel", ["Zu Fuß","Fahrrad","Strassenbahn (VVS)",
                            "Bus (VVS)","S-Bahn (VVS)","E-Scooter","Auto (keine Punkte)"])
        km   = st.slider("Strecke (km)", 0.5, 50.0, 5.0, step=0.5)
        dep  = st.time_input("Abfahrtszeit", value=datetime.now().replace(hour=8,minute=0,second=0,microsecond=0))
        mp   = engine.calculate_mobility_points(mode, km)
        co2s = engine.calculate_co2(mode, km)

        if "Auto" in mode:
            st.warning("Auto Keine Punkte für Autofahrten. Wechsle auf ÖPNV oder Rad!")
            mp = 0
        else:
            st.info(f" **+{mp} StuttPunkte** ·  **{co2s:.2f} kg CO₂** eingespart")

        if st.button("Fahrt speichern", type="primary", use_container_width=True):
            if mp > 0:
                st.session_state.points += mp
                new = {"date":datetime.now().strftime("%Y-%m-%d"),"mode":mode.split(" ",1)[1],
                       "km":km,"points":mp,"co2_saved_kg":co2s,"departure":str(dep)}
                st.session_state.mobility_log = pd.concat(
                    [st.session_state.mobility_log, pd.DataFrame([new])], ignore_index=True)
                st.success(f" +{mp} Punkte gespeichert!")
            else:
                st.info("Keine Punkte für Autofahrten.")

    with c2:
        st.markdown(sec("Punktetabelle"), unsafe_allow_html=True)
        df_mob = pd.DataFrame([
            {"Verkehrsmittel":"Zu Fuß","Pkte/km":4,"CO₂/km":"0.21 kg"},
            {"Verkehrsmittel":"Fahrrad","Pkte/km":5,"CO₂/km":"0.21 kg"},
            {"Verkehrsmittel":"Strassenbahn Straßenbahn","Pkte/km":3,"CO₂/km":"0.17 kg"},
            {"Verkehrsmittel":"Bus Bus","Pkte/km":3,"CO₂/km":"0.12 kg"},
            {"Verkehrsmittel":"S-Bahn S-Bahn","Pkte/km":3,"CO₂/km":"0.18 kg"},
            {"Verkehrsmittel":"E-Scooter","Pkte/km":2,"CO₂/km":"0.10 kg"},
            {"Verkehrsmittel":"Auto Auto","Pkte/km":0,"CO₂/km":"—"},
        ])
        st.dataframe(df_mob, use_container_width=True, hide_index=True)
        st.markdown(sec("VVS Live-Abfahrten (Demo)"), unsafe_allow_html=True)
        st.markdown("| Linie | Richtung | Abfahrt | Punkte |\n|---|---|---|---|\n| U1 | Fellbach | 3 Min | +15 |\n| S1 | Herrenberg | 5 Min | +21 |\n| 42 | Botnang | 7 Min | +9 |\n| U14 | Remseck | 12 Min | +18 |")

    st.divider()
    st.markdown(sec("Mobilitätsverlauf — letzte 7 Tage"), unsafe_allow_html=True)
    md = st.session_state.mobility_log.groupby(["date","mode"])["km"].sum().reset_index()
    fm = px.bar(md,x="date",y="km",color="mode",barmode="stack",
        color_discrete_map={"Zu Fuß":R,"Fahrrad":BLU,"Straßenbahn (VVS)":GRN,
                            "Bus (VVS)":"#34d399","S-Bahn (VVS)":"#2dd4bf","E-Scooter":AMB,"Auto":"#475569"},
        labels={"km":"km","date":"Datum","mode":"Verkehrsmittel"})
    dk(fm, 280)
    st.plotly_chart(fm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STUTTGART-KARTE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Stuttgart-Karte":
    st.markdown(f"<h2 style='color:{TXT};'>Stuttgart — Aktivitätskarte</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};'>Aktivitätsdichte nach Stadtbezirk · simulierte Echtzeit-Daten</p>", unsafe_allow_html=True)

    np.random.seed(42)
    bezirke = pd.DataFrame([
        {"bezirk":"Stuttgart-Mitte",   "lat":48.7758,"lon":9.1829, "nutzer":1240,"co2":48.2,"schritte":9800,"oepnv":82},
        {"bezirk":"Stuttgart-Nord",    "lat":48.7987,"lon":9.1760, "nutzer":890, "co2":32.1,"schritte":8200,"oepnv":71},
        {"bezirk":"Stuttgart-Süd",     "lat":48.7580,"lon":9.1740, "nutzer":760, "co2":27.4,"schritte":7600,"oepnv":68},
        {"bezirk":"Stuttgart-Ost",     "lat":48.7810,"lon":9.2100, "nutzer":680, "co2":24.9,"schritte":7100,"oepnv":61},
        {"bezirk":"Stuttgart-West",    "lat":48.7730,"lon":9.1560, "nutzer":920, "co2":33.8,"schritte":8500,"oepnv":74},
        {"bezirk":"Bad Cannstatt",     "lat":48.8060,"lon":9.2200, "nutzer":1100,"co2":40.3,"schritte":9200,"oepnv":78},
        {"bezirk":"Zuffenhausen",      "lat":48.8370,"lon":9.1750, "nutzer":540, "co2":19.8,"schritte":6400,"oepnv":55},
        {"bezirk":"Feuerbach",         "lat":48.8170,"lon":9.1550, "nutzer":610, "co2":22.4,"schritte":6800,"oepnv":59},
        {"bezirk":"Münster",           "lat":48.8230,"lon":9.2180, "nutzer":320, "co2":11.7,"schritte":5200,"oepnv":44},
        {"bezirk":"Hedelfingen",       "lat":48.7700,"lon":9.2450, "nutzer":280, "co2":10.3,"schritte":4900,"oepnv":41},
        {"bezirk":"Untertürkheim",     "lat":48.7840,"lon":9.2490, "nutzer":360, "co2":13.2,"schritte":5500,"oepnv":48},
        {"bezirk":"Wangen",            "lat":48.7890,"lon":9.2310, "nutzer":290, "co2":10.6,"schritte":5000,"oepnv":43},
        {"bezirk":"Degerloch",         "lat":48.7440,"lon":9.1860, "nutzer":480, "co2":17.6,"schritte":6200,"oepnv":52},
        {"bezirk":"Möhringen",         "lat":48.7230,"lon":9.1710, "nutzer":620, "co2":22.7,"schritte":7000,"oepnv":60},
        {"bezirk":"Vaihingen",         "lat":48.7310,"lon":9.1410, "nutzer":700, "co2":25.7,"schritte":7300,"oepnv":63},
        {"bezirk":"Sillenbuch",        "lat":48.7420,"lon":9.2230, "nutzer":390, "co2":14.3,"schritte":5700,"oepnv":49},
        {"bezirk":"Plieningen",        "lat":48.7080,"lon":9.2100, "nutzer":260, "co2":9.5, "schritte":4700,"oepnv":39},
        {"bezirk":"Botnang",           "lat":48.7800,"lon":9.1330, "nutzer":340, "co2":12.5,"schritte":5300,"oepnv":46},
        {"bezirk":"Stammheim",         "lat":48.8530,"lon":9.1870, "nutzer":290, "co2":10.6,"schritte":5000,"oepnv":43},
        {"bezirk":"Mühlhausen",        "lat":48.8440,"lon":9.2070, "nutzer":350, "co2":12.8,"schritte":5400,"oepnv":47},
        {"bezirk":"Weilimdorf",        "lat":48.8160,"lon":9.1220, "nutzer":520, "co2":19.1,"schritte":6500,"oepnv":56},
    ])

    cL, cR = st.columns([2, 1])

    with cR:
        st.markdown(sec("Kartenansicht"), unsafe_allow_html=True)
        mode_map = st.radio("Anzeige", ["Aktive Nutzer","CO₂ eingespart","ÖPNV-Quote (%)"], label_visibility="collapsed")
        col_map  = {"Aktive Nutzer":"nutzer","CO₂ eingespart":"co2","ÖPNV-Quote (%)":"oepnv"}[mode_map]
        lbl_map  = {"Aktive Nutzer":"Nutzer","CO₂ eingespart":"CO₂ kg","ÖPNV-Quote (%)":"ÖPNV %"}[mode_map]

        st.divider()
        st.markdown(sec("Top 5 Bezirke"), unsafe_allow_html=True)
        top5 = bezirke.nlargest(5, col_map)[["bezirk", col_map]]
        for _, row in top5.iterrows():
            v = f"{int(row[col_map]):,}".replace(",",".")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:6px 0;border-bottom:1px solid {BRD};'>"
                f"<span style='font-size:.82rem;color:{TXT};'>{row['bezirk']}</span>"
                f"<span style='font-size:.82rem;color:{RL};font-weight:700;'>{v}</span></div>",
                unsafe_allow_html=True)

        st.divider()
        total_u = bezirke["nutzer"].sum()
        total_c = bezirke["co2"].sum()
        st.markdown(mlcard("Gesamtstadt", f"{total_u:,}".replace(",","."), "aktive Nutzer"), unsafe_allow_html=True)
        st.markdown(mlcard("CO₂ gespart", f"{total_c:.0f} kg", "diese Woche"), unsafe_allow_html=True)

    with cL:
        # Farbskala: dunkelrot → rot → hellrot → gelb (Heatmap-artig)
        color_scale = [
            [0.0,  "#1a0a0a"],
            [0.25, "#6b0f0f"],
            [0.5,  R],
            [0.75, RL],
            [1.0,  AMB],
        ]
        fig_map = px.scatter_mapbox(
            bezirke,
            lat="lat", lon="lon",
            size=col_map,
            color=col_map,
            color_continuous_scale=color_scale,
            hover_name="bezirk",
            hover_data={"nutzer":True,"co2":":.1f","oepnv":True,"lat":False,"lon":False},
            size_max=60,
            zoom=11.0,
            center={"lat":48.775,"lon":9.182},
            mapbox_style="carto-darkmatter",
            labels={"nutzer":"Aktive Nutzer","co2":"CO₂ (kg)","oepnv":"ÖPNV %"},
        )
        fig_map.update_layout(
            paper_bgcolor=S1,
            margin=dict(t=0,b=0,l=0,r=0),
            height=520,
            coloraxis_colorbar=dict(
                title=dict(text=lbl_map, font=dict(color=TXT, size=12)),
                tickfont=dict(color=TXT),
                bgcolor=S1,
                bordercolor=BRD,
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.markdown(sec("Aktivitätsdichte nach Uhrzeit — Stuttgart gesamt"), unsafe_allow_html=True)
    hours = list(range(5,24))
    dens  = [50,120,380,820,1100,980,720,650,700,760,890,1050,870,680,720,980,1200,1080,760]
    df_h  = pd.DataFrame({"Uhrzeit":[f"{h:02d}:00" for h in hours],"Aktive Nutzer":dens})
    fh    = px.area(df_h, x="Uhrzeit", y="Aktive Nutzer", color_discrete_sequence=[R])
    fh.update_traces(fillcolor="rgba(232,49,42,.18)", line=dict(color=RL, width=2))
    dk(fh, 200)
    st.plotly_chart(fh, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ML-ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML-Analyse":
    st.markdown(f"<h2 style='color:{TXT};'> Machine Learning</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};'>Random Forest Klassifikation · Gradient Boosting Regression · Nudging-Modell</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Aktivitätserkennung","Punkteprognose","Nudging"])

    with tab1:
        st.markdown(f"<p style='color:{TXT};'>Random Forest erkennt deine Aktivität aus Wearable-Sensordaten in Echtzeit.</p>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            hr2  = st.slider("Herzfrequenz (bpm)", 55, 195, 145)
            cad  = st.slider("Schrittfrequenz (Schritte/Min)", 0, 200, 160)
            acc  = st.slider("Beschleunigung (m/s²)", 0.0, 20.0, 8.5)
        with c2:
            res  = st.session_state.ml_classifier.classify_sensor({"heart_rate":hr2,"cadence":cad,"acceleration":acc})
            st.markdown(sec("Erkannte Aktivität — Klassenwahrscheinlichkeiten"), unsafe_allow_html=True)
            for act2, prob in res["probabilities"].items():
                bw   = int(prob * 200)
                bc   = R if act2 == res["activity"] else BRD
                tc   = TXT if act2 == res["activity"] else MUT
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0;'>"
                    f"<span style='width:145px;font-size:13px;color:{tc};'>{act2}</span>"
                    f"<div style='width:{bw}px;height:12px;background:{bc};border-radius:3px;transition:width .3s;'></div>"
                    f"<span style='font-size:12px;color:{MUT};'>{prob:.0%}</span></div>",
                    unsafe_allow_html=True)
            st.markdown(mlcard("Ergebnis", res['activity'], f"Konfidenz: {res['confidence']:.0%}"), unsafe_allow_html=True)
        st.caption("Trainiert auf synthetischen Sensor-Profilen. Produktion: Apple Watch / Fitbit-Exports.")

    with tab2:
        st.markdown(f"<p style='color:{TXT};'>Gradient Boosting prognostiziert deine Punkte der nächsten 7 Tage.</p>", unsafe_allow_html=True)
        fc   = st.session_state.points_predictor.forecast_week(st.session_state.week_data)
        hist = st.session_state.week_data
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=hist["day"],y=hist["points_earned"],mode="lines+markers",
            name="Historisch",line=dict(color=R,width=2.5),marker=dict(size=8,color=RL)))
        fig4.add_trace(go.Scatter(x=fc["day"],y=fc["predicted"],mode="lines+markers",
            name="Prognose (ML)",line=dict(color=BLU,width=2.5,dash="dash"),
            marker=dict(size=8,symbol="diamond",color=BLU)))
        fig4.add_trace(go.Scatter(
            x=list(fc["day"])+list(fc["day"])[::-1],
            y=list(fc["upper"])+list(fc["lower"])[::-1],
            fill="toself",fillcolor="rgba(96,165,250,0.1)",
            line=dict(color="rgba(0,0,0,0)"),name="Konfidenzband"))
        dk(fig4, 340, legend_h=True)
        st.plotly_chart(fig4, use_container_width=True)
        tf = fc["predicted"].sum()
        st.info(f"Prognose nächste Woche: **{tf:.0f} Punkte** (+{tf/hist['points_earned'].sum()*100-100:.1f}% ggü. Vorwoche)")

    with tab3:
        st.markdown(f"<p style='color:{TXT};'>Optimale Uhrzeiten für Push-Benachrichtigungen zur nachhaltigen Mobilität.</p>", unsafe_allow_html=True)
        h2   = list(range(6,22))
        ns   = [0.1,0.3,0.7,0.85,0.6,0.4,0.3,0.25,0.35,0.55,0.65,0.8,0.9,0.75,0.5,0.3]
        df_n = pd.DataFrame({"Uhrzeit":[f"{h}:00" for h in h2],"Score":ns})
        fn   = px.area(df_n,x="Uhrzeit",y="Score",color_discrete_sequence=[R])
        fn.update_traces(fillcolor="rgba(232,49,42,.15)",line=dict(color=RL,width=2))
        dk(fn, 230)
        st.plotly_chart(fn, use_container_width=True)
        c1,c2,c3 = st.columns(3)
        for col,t,msg in zip([c1,c2,c3],["07:45","12:15","18:00"],
            ["S-Bahn statt Auto → +21 Pkte","Mittagsspaziergang → +16 Pkte","Mit dem Rad heim → +30 Pkte"]):
            with col:
                col.markdown(mlcard(f"{t} Uhr","",msg), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  BELOHNUNGEN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Belohnungen":
    st.markdown(f"<h2 style='color:{TXT};'>Belohnungen einlösen</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};'>Guthaben: <span style='color:{RL};font-weight:700;font-size:1.1rem;'>{st.session_state.points:,} StuttPunkte</span></p>".replace(",","."), unsafe_allow_html=True)

    rewards = [
        {"name":"VVS-Tageskarte Zone 1+2","partner":"VVS","cost":1500,"icon":"VVS","cat":"Mobilität"},
        {"name":"Wilhelma-Eintritt","partner":"Wilhelma Stuttgart","cost":800,"icon":"Tier","cat":"Freizeit"},
        {"name":"Staatsgalerie — freier Eintritt","partner":"Staatsgalerie","cost":600,"icon":"Kunst","cat":"Kultur"},
        {"name":"Stadtbibliothek Kaffee","partner":"Stadtbibliothek","cost":200,"icon":"Kaffee","cat":"Gastronomie"},
        {"name":"StuttCard Plus 1 Tag","partner":"Stuttgart Tourist","cost":1000,"icon":"Karte","cat":"Mobilität"},
        {"name":"Mercedes-Benz Museum","partner":"Mercedes-Benz","cost":700,"icon":"Auto","cat":"Kultur"},
        {"name":"Leihrad callabike 1 Std.","partner":"DB Connect","cost":150,"icon":"Rad","cat":"Mobilität"},
        {"name":"Biergarten-Gutschein","partner":"Stuttgarter Weindorf","cost":400,"icon":"Getrank","cat":"Gastronomie"},
    ]
    cats = st.multiselect("Kategorie", ["Mobilität","Freizeit","Kultur","Gastronomie"],
                          default=["Mobilität","Freizeit","Kultur","Gastronomie"])

    for r in rewards:
        if r["cat"] not in cats: continue
        ok  = st.session_state.points >= r["cost"]
        bc  = R if ok else BRD
        c1,c2,c3 = st.columns([.5, 3.5, 1.5])
        with c1:
            icon_lbl = r['icon']
            ic = COLORS["text_muted"]; ibg = COLORS["border"]
            st.markdown(f"<span style='font-size:0.82rem;font-weight:600;color:{ic};padding:3px 8px;background:{ibg};border-radius:6px;'>{icon_lbl}</span>", unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div style='padding:4px 0;'>"
                f"<div style='font-size:.95rem;font-weight:600;color:{TXT};'>{r['name']}</div>"
                f"<div style='font-size:.78rem;color:{MUT};margin-top:2px;'>{r['partner']} · {badge(r['cat'])}</div>"
                f"</div>", unsafe_allow_html=True)
        with c3:
            if ok:
                if st.button(f"Einlösen — {r['cost']} Pkte", key=r["name"], type="primary"):
                    st.session_state.points -= r["cost"]
                    st.success(f"{r['name']} eingelöst!")
                    st.rerun()
            else:
                st.markdown(f"<span style='color:{MUT};font-size:.82rem;'>Noch {r['cost']-st.session_state.points} Pkte</span>", unsafe_allow_html=True)
        st.markdown(f"<hr style='border:none;border-top:1px solid {BRD};margin:4px 0;'>", unsafe_allow_html=True)
