# Stuttgart ActiveCity

**Prototyp einer Smart-City-App der Landeshauptstadt Stuttgart**
Nachhaltige Mobilitat und OPNV-Nutzung werden mit StuttPunkten belohnt — unterstutzt durch Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikitlearn)](https://scikit-learn.org/)


---

## Idee

Stuttgart ActiveCity setzt einen einfachen Anreiz: Wer zu Fuss geht, Rad fahrt oder den VVS nutzt,
sammelt StuttPunkte und kann diese gegen echte lokale Belohnungen eintauschen —
VVS-Tageskarten, Wilhelma-Eintritte, Stadtbibliothek-Gutscheine u. v. m.

Machine Learning macht das Erlebnis datengetrieben: Das System prognostiziert die stadtweite
Mobilitatsauslastung stundengenau und analysiert Mobilitatsmuster nach Stuttgarter Stadtbezirk.

---

## Screens

| Screen | Inhalt |
|---|---|
| Dashboard | Wochenuberblick, Kilometer, CO2-Einsparung, ML-Schnellprognose |
| Fahrt erfassen | Verkehrsmittel loggen, Punkte berechnen, VVS-Abfahrten |
| Mobilitat & OPNV | Detailauswertung: km, Punkte und CO2 nach Verkehrsmittel |
| Stuttgart-Karte | Interaktive Heatmap aller 21 Stadtbezirke |
| ML-Analyse | Auslastungsprognose + Mobilitatsanalyse (siehe unten) |
| Belohnungen | StuttPunkte gegen Stuttgarter Angebote einlosen |

---

## Machine Learning

### Auslastungsprognose — Gradient Boosting Regressor

```
Ziel:     Prognose aktiver Nutzer im Stuttgarter Stadtgebiet (stundlich, 24h)
Modell:   GradientBoostingRegressor (scikit-learn)
          200 Estimatoren · max_depth 4 · learning_rate 0.08
Features: Uhrzeit, Uhrzeit (sin/cos-kodiert), Wochentag, Wochentag (sin-kodiert)
Output:   Stundliche Nutzerzahl + Konfidenzband + Spitzenstunden
```

Das Modell wird einmalig beim App-Start auf synthetischen Wochenstundendaten trainiert
und im Session State gecacht — kein erneutes Training bei jedem Seitenaufruf.
Der Wetter-Score (0 = Regen, 10 = Sonnig) skaliert die Prognose post-hoc,
in der Produktion via DWD-API befullt.

### Mobilitatsanalyse — Stadtbezirke Stuttgart

```
Ziel:     Visualisierung von Mobilitatsmustern nach Bezirk
Daten:    21 Stuttgarter Stadtbezirke mit OPNV-Quote, Rad-Anteil,
          Fuss-Anteil, aktiven Nutzern, CO2-Einsparung
Analyse:  Gestackter Balken (Mobilitatsanteil), Streudiagramm
          (OPNV vs. Rad), CO2-Ranking
```

In der Produktion: anonymisierte GPS-Aggregatdaten + VVS-Fahrgastzahlen (TRIAS-API).

---

## Punktesystem Mobilitat

| Verkehrsmittel | Punkte/km | CO2-Ersparnis vs. PKW |
|---|---|---|
| Zu Fuss | 4 | 0,21 kg/km |
| Fahrrad | 5 | 0,21 kg/km |
| Strassenbahn (VVS) | 3 | 0,17 kg/km |
| S-Bahn (VVS) | 3 | 0,18 kg/km |
| Bus (VVS) | 3 | 0,12 kg/km |
| E-Scooter | 2 | 0,10 kg/km |
| Auto | 0 | — |

Streckenbonus: uber 10 km pro Fahrt gibt es 20% mehr Punkte.

### Level-System

| Level | Titel | ab Punkten |
|---|---|---|
| 1 | Spazierganger | 0 |
| 2 | Stadtlaufer | 200 |
| 3 | Radpionier | 500 |
| 4 | OPNV-Champion | 1.000 |
| 5 | Klimaheld | 2.000 |
| 6 | Klimaheld+ | 3.500 |
| 7 | ActiveCity Pro | 5.000 |

---

## Installation & Start

```bash
git clone https://github.com/christianm38/activity-city-app.git
cd activity-city-app

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Die App offnet sich unter `http://localhost:8501`.

---

## Projektstruktur

```
activity-city-app/
├── app.py                      # Streamlit-Hauptanwendung
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   ├── activity_classifier.py  # (Erweiterung: Verkehrsmittelerkennung)
│   └── points_predictor.py     # (Erweiterung: individuelle Punkteprognose)
└── utils/
    ├── __init__.py
    ├── points_engine.py        # Punkte- und Level-Berechnung
    └── data_generator.py       # Synthetische Mobilitatsdaten
```

---

## Produktionsarchitektur (Vision)

```
Datenquellen                Backend                  Datenbank
────────────────            ─────────────────────    ──────────────
Apple HealthKit    ──┐      FastAPI                  PostgreSQL
Google Fit         ──┼──►  ML-Inferenz-Service  ──► TimescaleDB
VVS TRIAS-API      ──┤      Anonymisierung           Redis (Cache)
DWD Wetter-API     ──┘
                            ↕
                       Streamlit / React Native App
```

**VVS-Integration**: Live-Abfahrten und Fahrgastzahlen via
[VVS TRIAS-API](https://www.openvvs.de/dataset/trias).
OPNV-Nutzung wird automatisch via NFC-Ticket-Scan erkannt und gutgeschrieben.

---

## Datenschutz

Alle Bewegungsdaten verbleiben im Demo-Modus lokal in der Session.
Produktiv: DSGVO-konforme Datenspeicherung, Ende-zu-Ende-Verschlusselung,
explizite Einwilligung vor Mobility-Sync.

---

## Autor

**Christian Mann** 
GitHub: [github.com/christianm38](https://github.com/christianm38)

---

