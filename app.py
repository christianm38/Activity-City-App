"""
Stuttgart ActiveCity v4
Fokus: Nachhaltige Mobilitat — Tracking, Punkte, ML-Stadtanalyse
Keine Sportarten. ML: Auslastungsprognose + Mobilitatsanalyse.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor

from utils.data_generator import generate_mobility_data
from utils.points_engine import PointsEngine

st.set_page_config(
    page_title="Stuttgart ActiveCity",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Farben
R   = "#E8312A"
RL  = "#ff6b6b"
BG  = "#0f1117"
S1  = "#1c1f2e"
S2  = "#252840"
BRD = "#2e3250"
TXT = "#f0f2ff"
MUT = "#8b92b8"
GRN = "#4ade80"
BLU = "#60a5fa"
AMB = "#fbbf24"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG} !important;
    color: {TXT} !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {S1} !important;
    border-right: 1px solid {BRD} !important;
}}
section[data-testid="stSidebar"] * {{ color: {TXT} !important; }}
section[data-testid="stSidebar"] .stRadio label {{ color: {MUT} !important; font-size:11px; text-transform:uppercase; letter-spacing:1px; }}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{ color: {TXT} !important; font-size:14px; text-transform:none; letter-spacing:0; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] > div {{
    background-color: {S2} !important; border-color: {BRD} !important; color: {TXT} !important;
}}
div[data-baseweb="select"] span, div[data-baseweb="select"] div {{ color: {TXT} !important; background-color: transparent !important; }}
div[data-baseweb="popover"] ul {{ background-color: {S2} !important; border: 1px solid {BRD} !important; }}
div[data-baseweb="popover"] li:hover {{ background-color: {BRD} !important; }}
div[data-baseweb="popover"] li span {{ color: {TXT} !important; }}
input, textarea {{ background-color: {S2} !important; color: {TXT} !important; border-color: {BRD} !important; }}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{ background-color: {R} !important; border-color: {R} !important; }}
div[data-testid="stSlider"] div[data-baseweb="slider"] div:first-child div:first-child {{ background-color: {R} !important; }}
button[kind="primary"], .stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {R}, #c0201a) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 14px !important; padding: 10px 20px !important;
    box-shadow: 0 4px 15px rgba(232,49,42,0.35) !important;
}}
.stButton > button:not([kind="primary"]) {{
    background: {S2} !important; color: {TXT} !important;
    border: 1px solid {BRD} !important; border-radius: 10px !important;
}}
button[data-baseweb="tab"] {{ background: {S1} !important; color: {MUT} !important; border-radius: 8px 8px 0 0 !important; font-size: 13px !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ background: {R} !important; color: white !important; font-weight: 600 !important; }}
div[data-baseweb="tab-panel"] {{ background: {S1} !important; border: 1px solid {BRD} !important; border-radius: 0 8px 8px 8px !important; padding: 1.2rem !important; }}
div[data-testid="stProgressBar"] > div {{ background: {BRD} !important; border-radius: 10px; }}
div[data-testid="stProgressBar"] > div > div {{ background: linear-gradient(90deg, {R}, {RL}) !important; border-radius: 10px; }}
div[data-testid="stAlert"] {{ border-radius: 10px !important; }}
div[data-testid="stAlert"] p {{ color: {TXT} !important; }}
div[data-testid="stDataFrameResizable"] {{ background: {S1} !important; border: 1px solid {BRD} !important; border-radius: 10px; }}
div[data-testid="stDataFrameResizable"] * {{ color: {TXT} !important; background-color: transparent !important; }}
hr {{ border: none; border-top: 1px solid {BRD} !important; margin: 1rem 0; }}
.stMarkdown p, .stMarkdown li, .element-container p {{ color: {TXT} !important; }}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 {{ color: {TXT} !important; }}
[data-testid="stCaptionContainer"] p {{ color: {MUT} !important; }}
span[data-baseweb="tag"] {{ background: {R} !important; }}
span[data-baseweb="tag"] span {{ color: white !important; }}
div[data-testid="stTimeInput"] input {{ background: {S2} !important; color: {TXT} !important; border: 1px solid {BRD} !important; }}
h1,h2,h3 {{ color: {TXT} !important; }}
label {{ color: {MUT} !important; }}
div[data-testid="stNumberInput"] input {{ background: {S2} !important; color: {TXT} !important; border: 1px solid {BRD} !important; border-radius: 8px !important; }}
div[data-testid="stNumberInput"] button {{ background: {S2} !important; color: {TXT} !important; border-color: {BRD} !important; }}
</style>
""", unsafe_allow_html=True)

# Plotly-Helfer
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

# HTML-Komponenten
def metric(val, lbl):
    return (f"<div style='background:{S1};border:1px solid {BRD};border-radius:12px;"
            f"padding:1.2rem;text-align:center;'>"
            f"<div style='font-size:1.8rem;font-weight:700;color:{RL};line-height:1.1;'>{val}</div>"
            f"<div style='font-size:.72rem;color:{MUT};margin-top:5px;text-transform:uppercase;"
            f"letter-spacing:.6px;'>{lbl}</div></div>")

def mlcard(label, val, sub=""):
    sub_html = f"<div style='font-size:.85rem;color:{TXT};margin-top:2px;'>{sub}</div>" if sub else ""
    return (f"<div style='background:{S2};border:1px solid {R};border-radius:12px;"
            f"padding:1rem 1.2rem;margin:.4rem 0;'>"
            f"<div style='font-size:.7rem;color:{MUT};text-transform:uppercase;"
            f"letter-spacing:.8px;font-weight:600;'>{label}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{RL};margin-top:4px;'>{val}</div>"
            f"{sub_html}</div>")

def sec(txt):
    return (f"<p style='color:{MUT};font-size:.72rem;text-transform:uppercase;"
            f"letter-spacing:.8px;font-weight:600;margin:1rem 0 .5rem;'>{txt}</p>")

def badge(txt):
    return (f"<span style='background:rgba(232,49,42,.18);color:{RL};"
            f"border:1px solid rgba(232,49,42,.4);border-radius:20px;"
            f"padding:3px 11px;font-size:.75rem;font-weight:600;'>{txt}</span>")

# Mobilitatsfarben
MOB_COLORS = {
    "Zu Fuß":            R,
    "Fahrrad":           BLU,
    "Straßenbahn (VVS)": GRN,
    "Bus (VVS)":         "#34d399",
    "S-Bahn (VVS)":      "#2dd4bf",
    "E-Scooter":         AMB,
    "Auto":              "#475569",
}

# Session State
if "points" not in st.session_state:
    st.session_state.points = 1240
if "mobility_log" not in st.session_state:
    st.session_state.mobility_log = generate_mobility_data()
if "model_auslastung" not in st.session_state:
    np.random.seed(0)
    _n = 7 * 24
    _h_all = np.arange(_n, dtype=float)
    _wd    = (_h_all // 24) % 7
    _hod   = _h_all % 24
    _base  = np.array([50,120,380,820,1100,980,720,650,700,760,
                        890,1050,870,680,720,980,1200,1080,760,430,
                        280,160,80,50], dtype=float)
    _y = np.array([
        _base[int(h) % 24] * (1.2 if d >= 5 else 1.0) + np.random.normal(0, 60)
        for h, d in zip(_hod, _wd)
    ])
    _X = np.column_stack([_hod, np.sin(2*np.pi*_hod/24), np.cos(2*np.pi*_hod/24),
                           _wd,  np.sin(2*np.pi*_wd/7)])
    _m = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                    learning_rate=0.08, random_state=42)
    _m.fit(_X, _y)
    st.session_state.model_auslastung = _m

engine = PointsEngine()

# Sidebar
with st.sidebar:
    st.markdown(f"<h3 style='color:{RL};margin:0 0 2px;'>Stuttgart ActiveCity</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};font-size:.8rem;margin:0 0 12px;'>Nachhaltige Mobilitat — Stuttgart</p>", unsafe_allow_html=True)
    st.divider()

    name  = st.text_input("Dein Name", value="Christian M.")
    level = engine.get_level(st.session_state.points)
    st.markdown(badge(f"Level {level['level']} — {level['title']}"), unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.4rem;font-weight:700;color:{RL};margin:8px 0 4px;'>{st.session_state.points:,} Pkte</p>".replace(",", "."), unsafe_allow_html=True)
    prog = (st.session_state.points - level["min"]) / max(1, level["next"] - level["min"])
    st.progress(min(prog, 1.0), text=f"Level {level['level']+1}: {level['next']:,} Pkte".replace(",", "."))

    st.divider()
    st.markdown(sec("Mobility-Sync"), unsafe_allow_html=True)
    sync = st.selectbox("Quelle", ["Apple Health", "Google Fit", "Garmin Connect", "Fitbit"], label_visibility="collapsed")
    st.success(f"{sync} verbunden · 06:32")

    st.divider()
    st.markdown(sec("Navigation"), unsafe_allow_html=True)
    page = st.radio("nav", [
        "Dashboard", "Fahrt erfassen", "Mobilitat & OPNV",
        "Stuttgart-Karte", "ML-Analyse", "Belohnungen"
    ], label_visibility="collapsed")

# Stadtbezirke (wiederverwendet auf Karte + ML)
BEZIRKE = pd.DataFrame([
    {"bezirk": "Stuttgart-Mitte",   "lat": 48.7758, "lon": 9.1829,  "nutzer": 1240, "co2": 48.2, "oepnv_pct": 82, "rad_pct": 12, "fuss_pct": 6},
    {"bezirk": "Stuttgart-Nord",    "lat": 48.7987, "lon": 9.1760,  "nutzer": 890,  "co2": 32.1, "oepnv_pct": 71, "rad_pct": 18, "fuss_pct": 11},
    {"bezirk": "Stuttgart-Süd",     "lat": 48.7580, "lon": 9.1740,  "nutzer": 760,  "co2": 27.4, "oepnv_pct": 68, "rad_pct": 20, "fuss_pct": 12},
    {"bezirk": "Stuttgart-Ost",     "lat": 48.7810, "lon": 9.2100,  "nutzer": 680,  "co2": 24.9, "oepnv_pct": 61, "rad_pct": 22, "fuss_pct": 17},
    {"bezirk": "Stuttgart-West",    "lat": 48.7730, "lon": 9.1560,  "nutzer": 920,  "co2": 33.8, "oepnv_pct": 74, "rad_pct": 16, "fuss_pct": 10},
    {"bezirk": "Bad Cannstatt",     "lat": 48.8060, "lon": 9.2200,  "nutzer": 1100, "co2": 40.3, "oepnv_pct": 78, "rad_pct": 14, "fuss_pct": 8},
    {"bezirk": "Zuffenhausen",      "lat": 48.8370, "lon": 9.1750,  "nutzer": 540,  "co2": 19.8, "oepnv_pct": 55, "rad_pct": 25, "fuss_pct": 20},
    {"bezirk": "Feuerbach",         "lat": 48.8170, "lon": 9.1550,  "nutzer": 610,  "co2": 22.4, "oepnv_pct": 59, "rad_pct": 23, "fuss_pct": 18},
    {"bezirk": "Münster",           "lat": 48.8230, "lon": 9.2180,  "nutzer": 320,  "co2": 11.7, "oepnv_pct": 44, "rad_pct": 30, "fuss_pct": 26},
    {"bezirk": "Hedelfingen",       "lat": 48.7700, "lon": 9.2450,  "nutzer": 280,  "co2": 10.3, "oepnv_pct": 41, "rad_pct": 32, "fuss_pct": 27},
    {"bezirk": "Untertürkheim",     "lat": 48.7840, "lon": 9.2490,  "nutzer": 360,  "co2": 13.2, "oepnv_pct": 48, "rad_pct": 28, "fuss_pct": 24},
    {"bezirk": "Wangen",            "lat": 48.7890, "lon": 9.2310,  "nutzer": 290,  "co2": 10.6, "oepnv_pct": 43, "rad_pct": 31, "fuss_pct": 26},
    {"bezirk": "Degerloch",         "lat": 48.7440, "lon": 9.1860,  "nutzer": 480,  "co2": 17.6, "oepnv_pct": 52, "rad_pct": 27, "fuss_pct": 21},
    {"bezirk": "Möhringen",         "lat": 48.7230, "lon": 9.1710,  "nutzer": 620,  "co2": 22.7, "oepnv_pct": 60, "rad_pct": 24, "fuss_pct": 16},
    {"bezirk": "Vaihingen",         "lat": 48.7310, "lon": 9.1410,  "nutzer": 700,  "co2": 25.7, "oepnv_pct": 63, "rad_pct": 22, "fuss_pct": 15},
    {"bezirk": "Sillenbuch",        "lat": 48.7420, "lon": 9.2230,  "nutzer": 390,  "co2": 14.3, "oepnv_pct": 49, "rad_pct": 29, "fuss_pct": 22},
    {"bezirk": "Plieningen",        "lat": 48.7080, "lon": 9.2100,  "nutzer": 260,  "co2": 9.5,  "oepnv_pct": 39, "rad_pct": 33, "fuss_pct": 28},
    {"bezirk": "Botnang",           "lat": 48.7800, "lon": 9.1330,  "nutzer": 340,  "co2": 12.5, "oepnv_pct": 46, "rad_pct": 30, "fuss_pct": 24},
    {"bezirk": "Stammheim",         "lat": 48.8530, "lon": 9.1870,  "nutzer": 290,  "co2": 10.6, "oepnv_pct": 43, "rad_pct": 31, "fuss_pct": 26},
    {"bezirk": "Mühlhausen",        "lat": 48.8440, "lon": 9.2070,  "nutzer": 350,  "co2": 12.8, "oepnv_pct": 47, "rad_pct": 29, "fuss_pct": 24},
    {"bezirk": "Weilimdorf",        "lat": 48.8160, "lon": 9.1220,  "nutzer": 520,  "co2": 19.1, "oepnv_pct": 56, "rad_pct": 26, "fuss_pct": 18},
])

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#9b0000,{R});border-radius:16px;"
        f"padding:1.6rem 2rem;margin-bottom:1.5rem;border:1px solid rgba(255,255,255,.1);'>"
        f"<h2 style='color:white;margin:0;font-size:1.6rem;'>Guten Morgen, {name.split()[0]}</h2>"
        f"<p style='color:rgba(255,255,255,.8);margin:6px 0 0;font-size:.9rem;'>"
        f"Stuttgart ActiveCity — nachhaltige Mobilitat wird belohnt</p></div>",
        unsafe_allow_html=True,
    )

    co2  = st.session_state.mobility_log["co2_saved_kg"].sum()
    km_w = st.session_state.mobility_log["km"].sum()
    mob_pts_today = st.session_state.mobility_log.iloc[-1]["points"] if len(st.session_state.mobility_log) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric(f"{st.session_state.points:,}".replace(",", "."), "StuttPunkte gesamt"), unsafe_allow_html=True)
    c2.markdown(metric(f"{km_w:.1f} km", "Kilometer diese Woche"), unsafe_allow_html=True)
    c3.markdown(metric(f"{co2:.1f} kg", "CO2 eingespart"), unsafe_allow_html=True)
    c4.markdown(metric(f"{int(mob_pts_today)}", "Punkte letzte Fahrt"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cL, cR = st.columns([3, 2])

    with cL:
        st.markdown(sec("Kilometer pro Tag nach Verkehrsmittel"), unsafe_allow_html=True)
        md = st.session_state.mobility_log.groupby(["date", "mode"])["km"].sum().reset_index()
        fm = px.bar(md, x="date", y="km", color="mode", barmode="stack",
                    color_discrete_map=MOB_COLORS,
                    labels={"km": "km", "date": "Datum", "mode": "Verkehrsmittel"})
        dk(fm, 300)
        st.plotly_chart(fm, use_container_width=True)

    with cR:
        st.markdown(sec("Mobilitatsmix — Kilometer"), unsafe_allow_html=True)
        mob = st.session_state.mobility_log.groupby("mode")["km"].sum().reset_index()
        fig2 = px.pie(mob, values="km", names="mode", hole=.58,
                      color_discrete_map=MOB_COLORS)
        fig2.update_traces(textfont=dict(color="white", size=12))
        fig2.update_layout(plot_bgcolor=S1, paper_bgcolor=S1, height=300,
                           margin=dict(t=10, b=10, l=10, r=10),
                           font=dict(color=TXT),
                           legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TXT)))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    # ML-Schnellprognose im Dashboard
    st.markdown(sec("ML-Prognose Stadtauslastung — nachste 3 Stunden"), unsafe_allow_html=True)
    now_h = datetime.now().hour
    future_hours = [(now_h + i) % 24 for i in range(1, 4)]
    base = [820, 1100, 980, 720, 650, 700, 760, 890, 1050, 870, 680, 720,
            980, 1200, 1080, 760, 430, 280, 160, 80, 50, 50, 120, 380]
    c1, c2, c3 = st.columns(3)
    for col, h in zip([c1, c2, c3], future_hours):
        val = base[h] + np.random.randint(-40, 40)
        trend = "steigend" if h in [7, 8, 9, 17, 18] else "fallend" if h in [10, 11, 20, 21] else "stabil"
        col.markdown(mlcard(f"{h:02d}:00 Uhr", f"{val:,}".replace(",", "."), f"Nutzer aktiv · {trend}"), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# FAHRT ERFASSEN
# ════════════════════════════════════════════════════════════════════════════
elif page == "Fahrt erfassen":
    st.markdown(f"<h2 style='color:{TXT};'>Fahrt erfassen</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};'>Alle nachhaltigen Verkehrsmittel sammeln StuttPunkte — nur das Auto nicht.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        mode = st.selectbox("Verkehrsmittel", [
            "Zu Fuß", "Fahrrad", "Straßenbahn (VVS)",
            "Bus (VVS)", "S-Bahn (VVS)", "E-Scooter", "Auto (keine Punkte)",
        ])
        km   = st.slider("Strecke (km)", 0.5, 50.0, 5.0, step=0.5)
        dep  = st.time_input("Abfahrtszeit", value=datetime.now().replace(hour=8, minute=0, second=0, microsecond=0))

        mp   = engine.calculate_mobility_points(mode, km)
        co2s = engine.calculate_co2(mode, km)

        if "Auto" in mode:
            st.warning("Autofahrten werden nicht belohnt. Wechsle auf OPNV, Rad oder zu Fuss!")
            mp = 0
        else:
            st.info(f"Vorschau: **+{mp} StuttPunkte** · **{co2s:.2f} kg CO2** eingespart")

        if st.button("Fahrt speichern", type="primary", use_container_width=True):
            if mp > 0:
                st.session_state.points += mp
                mode_clean = mode.replace(" (keine Punkte)", "")
                new = {
                    "date":        datetime.now().strftime("%Y-%m-%d"),
                    "mode":        mode_clean,
                    "km":          km,
                    "points":      mp,
                    "co2_saved_kg": co2s,
                    "departure":   str(dep),
                }
                st.session_state.mobility_log = pd.concat(
                    [st.session_state.mobility_log, pd.DataFrame([new])], ignore_index=True)
                st.success(f"+{mp} Punkte gespeichert! Gesamt: {st.session_state.points:,} Pkte".replace(",", "."))
            else:
                st.info("Keine Punkte fur Autofahrten.")

    with c2:
        st.markdown(sec("Punktetabelle Mobilitat"), unsafe_allow_html=True)
        df_mob = pd.DataFrame([
            {"Verkehrsmittel": "Zu Fuss",         "Pkte/km": 4, "CO2-Ersparnis": "0.21 kg/km"},
            {"Verkehrsmittel": "Fahrrad",           "Pkte/km": 5, "CO2-Ersparnis": "0.21 kg/km"},
            {"Verkehrsmittel": "Strassenbahn (VVS)","Pkte/km": 3, "CO2-Ersparnis": "0.17 kg/km"},
            {"Verkehrsmittel": "Bus (VVS)",          "Pkte/km": 3, "CO2-Ersparnis": "0.12 kg/km"},
            {"Verkehrsmittel": "S-Bahn (VVS)",       "Pkte/km": 3, "CO2-Ersparnis": "0.18 kg/km"},
            {"Verkehrsmittel": "E-Scooter",           "Pkte/km": 2, "CO2-Ersparnis": "0.10 kg/km"},
            {"Verkehrsmittel": "Auto",                "Pkte/km": 0, "CO2-Ersparnis": "—"},
        ])
        st.dataframe(df_mob, use_container_width=True, hide_index=True)
        st.caption("Streckenbonus: uber 10 km pro Fahrt gibt es 20% mehr Punkte.")

        st.markdown(sec("VVS Nachste Abfahrten (Demo)"), unsafe_allow_html=True)
        st.markdown(
            "| Linie | Richtung | Abfahrt | Punkte |\n"
            "|---|---|---|---|\n"
            "| U1 | Fellbach | 3 Min | +15 |\n"
            "| S1 | Herrenberg | 5 Min | +21 |\n"
            "| 42 | Botnang | 7 Min | +9 |\n"
            "| U14 | Remseck | 12 Min | +18 |"
        )

# ════════════════════════════════════════════════════════════════════════════
# MOBILITAT & OPNV
# ════════════════════════════════════════════════════════════════════════════
elif page == "Mobilitat & OPNV":
    st.markdown(f"<h2 style='color:{TXT};'>Mobilitat & OPNV — Auswertung</h2>", unsafe_allow_html=True)

    log = st.session_state.mobility_log

    # KPIs
    total_km  = log["km"].sum()
    oepnv_km  = log[log["mode"].str.contains("Bahn|Bus|bahn", na=False)]["km"].sum()
    rad_km    = log[log["mode"] == "Fahrrad"]["km"].sum()
    fuss_km   = log[log["mode"] == "Zu Fuß"]["km"].sum()
    auto_km   = log[log["mode"].str.contains("Auto", na=False)]["km"].sum()
    nachhaltig_pct = (total_km - auto_km) / max(total_km, 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric(f"{total_km:.1f} km", "Gesamt diese Woche"), unsafe_allow_html=True)
    c2.markdown(metric(f"{oepnv_km:.1f} km", "OPNV gefahren"), unsafe_allow_html=True)
    c3.markdown(metric(f"{rad_km:.1f} km", "Rad gefahren"), unsafe_allow_html=True)
    c4.markdown(metric(f"{nachhaltig_pct:.0f}%", "Nachhaltige Mobilitat"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cL, cR = st.columns([3, 2])

    with cL:
        st.markdown(sec("Verkehrsmittel nach Tag — Kilometer"), unsafe_allow_html=True)
        md = log.groupby(["date", "mode"])["km"].sum().reset_index()
        fm = px.bar(md, x="date", y="km", color="mode", barmode="stack",
                    color_discrete_map=MOB_COLORS,
                    labels={"km": "km", "date": "Datum", "mode": "Verkehrsmittel"})
        dk(fm, 300)
        st.plotly_chart(fm, use_container_width=True)

    with cR:
        st.markdown(sec("Anteil Verkehrsmittel"), unsafe_allow_html=True)
        mob_grp = log.groupby("mode")["km"].sum().reset_index()
        fig_pie = px.pie(mob_grp, values="km", names="mode", hole=.55,
                         color_discrete_map=MOB_COLORS)
        fig_pie.update_traces(textfont=dict(color="white", size=12))
        fig_pie.update_layout(plot_bgcolor=S1, paper_bgcolor=S1, height=300,
                              margin=dict(t=10, b=10, l=10, r=10),
                              font=dict(color=TXT),
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TXT)))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    cL2, cR2 = st.columns(2)
    with cL2:
        st.markdown(sec("Punkte pro Verkehrsmittel — kumuliert"), unsafe_allow_html=True)
        pts_grp = log.groupby("mode")["points"].sum().reset_index().sort_values("points")
        fig_pts = px.bar(pts_grp, x="points", y="mode", orientation="h",
                         color="mode", color_discrete_map=MOB_COLORS,
                         labels={"points": "StuttPunkte", "mode": ""},
                         text="points")
        fig_pts.update_traces(textposition="outside", textfont=dict(color=TXT))
        fig_pts.update_layout(showlegend=False)
        dk(fig_pts, 280)
        st.plotly_chart(fig_pts, use_container_width=True)

    with cR2:
        st.markdown(sec("CO2-Einsparung nach Verkehrsmittel"), unsafe_allow_html=True)
        co2_grp = log.groupby("mode")["co2_saved_kg"].sum().reset_index().sort_values("co2_saved_kg")
        fig_co2 = px.bar(co2_grp, x="co2_saved_kg", y="mode", orientation="h",
                         color="mode", color_discrete_map=MOB_COLORS,
                         labels={"co2_saved_kg": "kg CO2 eingespart", "mode": ""},
                         text="co2_saved_kg")
        fig_co2.update_traces(texttemplate="%{text:.1f} kg", textposition="outside",
                              textfont=dict(color=TXT))
        fig_co2.update_layout(showlegend=False)
        dk(fig_co2, 280)
        st.plotly_chart(fig_co2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# STUTTGART-KARTE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Stuttgart-Karte":
    st.markdown(f"<h2 style='color:{TXT};'>Stuttgart — Mobilitats-Heatmap</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};'>Aktive Nutzer und Mobilitatsmuster nach Stadtbezirk · simulierte Echtzeit-Daten</p>", unsafe_allow_html=True)

    cL, cR = st.columns([2, 1])

    with cR:
        st.markdown(sec("Kartenansicht"), unsafe_allow_html=True)
        mode_map = st.radio("Anzeige", [
            "Aktive Nutzer", "CO2 eingespart (kg)", "OPNV-Quote (%)", "Rad-Anteil (%)"
        ], label_visibility="collapsed")
        col_map = {
            "Aktive Nutzer":        "nutzer",
            "CO2 eingespart (kg)":  "co2",
            "OPNV-Quote (%)":       "oepnv_pct",
            "Rad-Anteil (%)":       "rad_pct",
        }[mode_map]
        lbl_map = {
            "Aktive Nutzer":        "Nutzer",
            "CO2 eingespart (kg)":  "CO2 kg",
            "OPNV-Quote (%)":       "OPNV %",
            "Rad-Anteil (%)":       "Rad %",
        }[mode_map]

        st.divider()
        st.markdown(sec("Top 5 Bezirke"), unsafe_allow_html=True)
        top5 = BEZIRKE.nlargest(5, col_map)[["bezirk", col_map]]
        for _, row in top5.iterrows():
            v = f"{int(row[col_map]):,}".replace(",", ".")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:6px 0;border-bottom:1px solid {BRD};'>"
                f"<span style='font-size:.82rem;color:{TXT};'>{row['bezirk']}</span>"
                f"<span style='font-size:.82rem;color:{RL};font-weight:700;'>{v}</span></div>",
                unsafe_allow_html=True,
            )
        st.divider()
        total_u = BEZIRKE["nutzer"].sum()
        total_c = BEZIRKE["co2"].sum()
        st.markdown(mlcard("Gesamtstadt", f"{total_u:,}".replace(",", "."), "aktive Nutzer"), unsafe_allow_html=True)
        st.markdown(mlcard("CO2 gespart", f"{total_c:.0f} kg", "diese Woche"), unsafe_allow_html=True)

    with cL:
        color_scale = [[0.0, "#1a0a0a"], [0.25, "#6b0f0f"], [0.5, R], [0.75, RL], [1.0, AMB]]
        fig_map = px.scatter_mapbox(
            BEZIRKE, lat="lat", lon="lon",
            size=col_map, color=col_map,
            color_continuous_scale=color_scale,
            hover_name="bezirk",
            hover_data={"nutzer": True, "co2": ":.1f", "oepnv_pct": True, "rad_pct": True, "lat": False, "lon": False},
            size_max=60, zoom=11.0,
            center={"lat": 48.775, "lon": 9.182},
            mapbox_style="carto-darkmatter",
            labels={"nutzer": "Aktive Nutzer", "co2": "CO2 (kg)", "oepnv_pct": "OPNV %", "rad_pct": "Rad %"},
        )
        fig_map.update_layout(
            paper_bgcolor=S1, margin=dict(t=0, b=0, l=0, r=0), height=520,
            coloraxis_colorbar=dict(
                title=dict(text=lbl_map, font=dict(color=TXT, size=12)),
                tickfont=dict(color=TXT), bgcolor=S1, bordercolor=BRD,
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.markdown(sec("Aktive Nutzer nach Uhrzeit — Stuttgart gesamt"), unsafe_allow_html=True)
    hours = list(range(5, 24))
    dens  = [50, 120, 380, 820, 1100, 980, 720, 650, 700, 760, 890, 1050, 870, 680, 720, 980, 1200, 1080, 760]
    df_h  = pd.DataFrame({"Uhrzeit": [f"{h:02d}:00" for h in hours], "Aktive Nutzer": dens})
    fh    = px.area(df_h, x="Uhrzeit", y="Aktive Nutzer", color_discrete_sequence=[R])
    fh.update_traces(fillcolor="rgba(232,49,42,.18)", line=dict(color=RL, width=2))
    dk(fh, 200)
    st.plotly_chart(fh, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# ML-ANALYSE
# ════════════════════════════════════════════════════════════════════════════
elif page == "ML-Analyse":
    st.markdown(f"<h2 style='color:{TXT};'>Machine Learning — Stadtanalyse</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUT};'>Auslastungsprognose (Gradient Boosting) · Mobilitatsanalyse nach Bezirk und Verkehrsmittel</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Auslastungsprognose", "Mobilitatsanalyse"])

    # ── Tab 1: Auslastungskarte ────────────────────────────────────────────
    with tab1:
        st.markdown(
            f"<p style='color:{TXT};'>Gradient-Boosting-Prognose der aktiven Nutzer "
            f"je Stuttgarter Stadtbezirk — basierend auf Wochentag und Wetter.</p>",
            unsafe_allow_html=True,
        )

        # Modell + Steuerung
        model_auslastung = st.session_state.model_auslastung
        ctrl1, ctrl2 = st.columns([1, 1])
        with ctrl1:
            prog_tag = st.selectbox("Prognose-Tag", [
                "Heute (Montag)", "Morgen (Dienstag)", "Mittwoch",
                "Donnerstag", "Freitag", "Samstag", "Sonntag",
            ])
            wd = {"Heute (Montag)": 0, "Morgen (Dienstag)": 1, "Mittwoch": 2,
                  "Donnerstag": 3, "Freitag": 4, "Samstag": 5, "Sonntag": 6}[prog_tag]
        with ctrl2:
            wetter = st.slider("Wetter-Score (0 = Regen, 10 = Sonnig)", 0, 10, 7)
            st.caption("In Produktion via DWD-API befullt.")

        # ML-Prognose berechnen
        h24    = np.arange(24, dtype=float)
        wd_arr = np.full(24, float(wd), dtype=float)
        X_pred = np.empty((24, 5), dtype=float)
        X_pred[:, 0] = h24
        X_pred[:, 1] = np.sin(2 * np.pi * h24 / 24)
        X_pred[:, 2] = np.cos(2 * np.pi * h24 / 24)
        X_pred[:, 3] = wd_arr
        X_pred[:, 4] = np.sin(2 * np.pi * wd_arr / 7)
        y_pred = model_auslastung.predict(X_pred)
        y_pred = np.clip(y_pred * (0.85 + float(wetter) * 0.03), 0, None).round()

        # Aktuelle Stunde auf Bezirke verteilen
        current_h  = min(datetime.now().hour, 23)
        total_now  = int(y_pred[current_h])
        weights    = BEZIRKE["nutzer"].values / BEZIRKE["nutzer"].sum()
        rng        = np.random.default_rng(seed=current_h + wd * 24)
        variation  = rng.uniform(0.85, 1.15, size=len(weights))
        nutzer_now = np.maximum((weights * total_now * variation).round().astype(int), 5)

        bezirke_now = BEZIRKE.copy()
        bezirke_now["aktiv_jetzt"] = nutzer_now

        # Karte
        st.markdown(
            f"<p style='color:{MUT};font-size:.82rem;margin-top:.5rem;'>"
            f"Aktuell ca. <b style='color:{TXT};'>{total_now:,}</b> Nutzer "
            f"unterwegs in Stuttgart — {int(current_h):02d}:00 Uhr.</p>".replace(",", "."),
            unsafe_allow_html=True,
        )
        map_col, stat_col = st.columns([3, 1])
        with map_col:
            fig_live = px.scatter_mapbox(
                bezirke_now,
                lat="lat", lon="lon",
                size="aktiv_jetzt",
                color="aktiv_jetzt",
                color_continuous_scale=[
                    [0.0, "#1a0a0a"], [0.3, "#6b0f0f"],
                    [0.6, R], [0.85, RL], [1.0, AMB],
                ],
                hover_name="bezirk",
                hover_data={"aktiv_jetzt": True, "oepnv_pct": True,
                            "rad_pct": True, "lat": False, "lon": False},
                size_max=55, zoom=11.0,
                center={"lat": 48.775, "lon": 9.182},
                mapbox_style="carto-darkmatter",
                labels={"aktiv_jetzt": "Aktiv jetzt", "oepnv_pct": "OPNV %", "rad_pct": "Rad %"},
            )
            fig_live.update_layout(
                paper_bgcolor=S1, margin=dict(t=0, b=0, l=0, r=0), height=500,
                coloraxis_colorbar=dict(
                    title=dict(text="Aktiv jetzt", font=dict(color=TXT, size=12)),
                    tickfont=dict(color=TXT), bgcolor=S1, bordercolor=BRD,
                ),
            )
            st.plotly_chart(fig_live, use_container_width=True)

        with stat_col:
            st.markdown(mlcard("Stadtgesamt",
                f"{total_now:,}".replace(",", "."),
                f"aktiv um {int(current_h):02d}:00 Uhr"), unsafe_allow_html=True)
            st.markdown(mlcard("Aktivster Bezirk",
                bezirke_now.loc[bezirke_now["aktiv_jetzt"].idxmax(), "bezirk"],
                f"{bezirke_now['aktiv_jetzt'].max():,} Nutzer".replace(",", ".")),
                unsafe_allow_html=True)
            st.markdown(mlcard("Ruhigster Bezirk",
                bezirke_now.loc[bezirke_now["aktiv_jetzt"].idxmin(), "bezirk"],
                f"{bezirke_now['aktiv_jetzt'].min():,} Nutzer".replace(",", ".")),
                unsafe_allow_html=True)
            st.caption("Aktualisiert bei Anderung von Tag oder Wetter.")

    # ── Tab 2: Mobilitatsanalyse ──────────────────────────────────────────
    with tab2:
        st.markdown(
            f"<p style='color:{TXT};'>Analyse der Mobilitatsmuster nach Stadtbezirk: "
            f"Wie viel wird gelaufen, mit dem Rad gefahren oder der Bahn genutzt — und wo?</p>",
            unsafe_allow_html=True,
        )

        # Stacked-Bar: OPNV vs Rad vs Fuss pro Bezirk
        st.markdown(sec("Mobilitatsanteil nach Bezirk (%)"), unsafe_allow_html=True)
        df_bez = BEZIRKE[["bezirk", "oepnv_pct", "rad_pct", "fuss_pct"]].copy()
        df_bez = df_bez.sort_values("oepnv_pct", ascending=True)
        df_melt = df_bez.melt(id_vars="bezirk",
                               value_vars=["oepnv_pct", "rad_pct", "fuss_pct"],
                               var_name="Modus", value_name="Anteil (%)")
        df_melt["Modus"] = df_melt["Modus"].map({
            "oepnv_pct": "OPNV", "rad_pct": "Fahrrad", "fuss_pct": "Zu Fuss"
        })
        fig_bez = px.bar(df_melt, x="Anteil (%)", y="bezirk", color="Modus",
                         barmode="stack", orientation="h",
                         color_discrete_map={"OPNV": GRN, "Fahrrad": BLU, "Zu Fuss": R},
                         labels={"bezirk": ""})
        dk(fig_bez, 580)
        st.plotly_chart(fig_bez, use_container_width=True)

        st.divider()
        cL, cR = st.columns(2)

        with cL:
            st.markdown(sec("OPNV-Quote vs. Rad-Anteil — Streudiagramm"), unsafe_allow_html=True)
            fig_scatter = px.scatter(
                BEZIRKE, x="oepnv_pct", y="rad_pct",
                size="nutzer", color="co2",
                hover_name="bezirk",
                color_continuous_scale=[[0, "#7f0000"], [1, RL]],
                labels={
                    "oepnv_pct": "OPNV-Quote (%)",
                    "rad_pct":   "Rad-Anteil (%)",
                    "nutzer":    "Aktive Nutzer",
                    "co2":       "CO2 eingespart (kg)",
                },
            )
            fig_scatter.update_traces(marker=dict(line=dict(color=BRD, width=1)))
            dk(fig_scatter, 320)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Grossere Punkte = mehr aktive Nutzer. Farbe = CO2-Einsparung.")

        with cR:
            st.markdown(sec("CO2-Einsparung nach Bezirk"), unsafe_allow_html=True)
            df_co2_bez = BEZIRKE[["bezirk", "co2"]].sort_values("co2", ascending=True)
            fig_co2_bez = px.bar(df_co2_bez, x="co2", y="bezirk", orientation="h",
                                  color="co2",
                                  color_continuous_scale=[[0, "#7f0000"], [1, RL]],
                                  labels={"co2": "CO2 eingespart (kg)", "bezirk": ""},
                                  text="co2")
            fig_co2_bez.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                       textfont=dict(color=TXT))
            fig_co2_bez.update_coloraxes(showscale=False)
            dk(fig_co2_bez, 580)
            st.plotly_chart(fig_co2_bez, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# BELOHNUNGEN
# ════════════════════════════════════════════════════════════════════════════
elif page == "Belohnungen":
    st.markdown(f"<h2 style='color:{TXT};'>Belohnungen einlosen</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{MUT};'>Guthaben: "
        f"<span style='color:{RL};font-weight:700;font-size:1.1rem;'>"
        f"{st.session_state.points:,} StuttPunkte</span></p>".replace(",", "."),
        unsafe_allow_html=True,
    )

    rewards = [
        {"name": "VVS-Tageskarte Zone 1+2",      "partner": "VVS",                   "cost": 1500, "icon": "VVS",     "cat": "Mobilitat"},
        {"name": "Wilhelma-Eintritt",              "partner": "Wilhelma Stuttgart",     "cost": 800,  "icon": "Freizeit","cat": "Freizeit"},
        {"name": "Staatsgalerie — freier Eintritt","partner": "Staatsgalerie",          "cost": 600,  "icon": "Kultur",  "cat": "Kultur"},
        {"name": "Stadtbibliothek Kaffee",         "partner": "Stadtbibliothek",        "cost": 200,  "icon": "Kaffee",  "cat": "Gastronomie"},
        {"name": "StuttCard Plus 1 Tag",           "partner": "Stuttgart Tourist",      "cost": 1000, "icon": "Karte",   "cat": "Mobilitat"},
        {"name": "Mercedes-Benz Museum",           "partner": "Mercedes-Benz",          "cost": 700,  "icon": "Kultur",  "cat": "Kultur"},
        {"name": "Leihrad callabike 1 Std.",       "partner": "DB Connect",             "cost": 150,  "icon": "Rad",     "cat": "Mobilitat"},
        {"name": "Biergarten-Gutschein",           "partner": "Stuttgarter Weindorf",   "cost": 400,  "icon": "Getrank", "cat": "Gastronomie"},
    ]

    cats = st.multiselect(
        "Kategorie",
        ["Mobilitat", "Freizeit", "Kultur", "Gastronomie"],
        default=["Mobilitat", "Freizeit", "Kultur", "Gastronomie"],
    )

    for r in rewards:
        if r["cat"] not in cats:
            continue
        ok = st.session_state.points >= r["cost"]
        c1, c2, c3 = st.columns([0.5, 3.5, 1.5])
        with c1:
            st.markdown(
                f"<span style='font-size:.82rem;font-weight:600;color:{MUT};"
                f"padding:3px 8px;background:{BRD};border-radius:6px;'>{r['icon']}</span>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div style='padding:4px 0;'>"
                f"<div style='font-size:.95rem;font-weight:600;color:{TXT};'>{r['name']}</div>"
                f"<div style='font-size:.78rem;color:{MUT};margin-top:2px;'>"
                f"{r['partner']} · {badge(r['cat'])}</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            if ok:
                if st.button(f"Einlosen — {r['cost']} Pkte", key=r["name"], type="primary"):
                    st.session_state.points -= r["cost"]
                    st.success(f"{r['name']} eingelost!")
                    st.rerun()
            else:
                st.markdown(
                    f"<span style='color:{MUT};font-size:.82rem;'>"
                    f"Noch {r['cost'] - st.session_state.points} Pkte</span>",
                    unsafe_allow_html=True,
                )
        st.markdown(f"<hr style='border:none;border-top:1px solid {BRD};margin:4px 0;'>", unsafe_allow_html=True)
