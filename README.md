# Activity-City-App
# Stuttgart ActiveCity

**Prototyp einer Smart-City-App der Landeshauptstadt Stuttgart**  
Bewegung, nachhaltige Mobilität und ÖPNV-Nutzung werden mit StuttPunkten belohnt — unterstützt durch Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikitlearn)](https://scikit-learn.org/)

---

## Idee

Stuttgart ActiveCity setzt einen einfachen Anreiz: Wer zu Fuß geht, Rad fährt oder den VVS nutzt,
sammelt StuttPunkte und kann diese gegen echte lokale Belohnungen eintauschen —  
VVS-Tageskarten, Wilhelma-Eintritte, Stadtbibliothek-Gutscheine u. v. m.

Machine Learning macht das Erlebnis personalisiert: Das System erkennt Aktivitäten automatisch
aus Sensordaten und prognostiziert, wann und wie du morgen am meisten Punkte sammeln kannst.

---

## Features

| Bereich | Funktion |
|---|---|
| Dashboard | Tagesziele, Punkte, CO₂-Einsparung, ML-Prognose |
| Aktivität | Laufen, Rad, Schwimmen, Fußball u. a. manuell loggen |
| Mobilität & ÖPNV | Alle Verkehrsmittel tracken; Auto = 0 Punkte |
| ML-Analyse | Aktivitätserkennung + Punkteprognose + Nudging |
| Belohnungen | Punkte gegen Stuttgarter Angebote einlösen |

---

## Machine Learning

### 1. Aktivitätserkennung — Random Forest Klassifikator

```
Eingabe: Herzfrequenz, Schrittfrequenz, Beschleunigung (Wearable-Sensordaten)
Ausgabe: Erkannte Aktivität + Konfidenzwert
Modell:  RandomForestClassifier (sklearn), 100 Estimatoren, 8 Aktivitätsklassen
```

Trainiert auf synthetischen Sensordaten mit realistischen physiologischen Profilen
(Laufen, Radfahren, Walken, Schwimmen, Fußball, Yoga, ÖPNV-Fahrt, Ruhe).  
In der Produktion: Training auf anonymisierten Apple Health / Fitbit Exports.

### 2. Punkteprognose — Gradient Boosting Regressor

```
Eingabe: Lag-Features (Vortag, Vorvortagswert), rollender 3-Tage-Schnitt,
         Streak-Länge, Wochentag (sin/cos-kodiert)
Ausgabe: Prognostizierte Punkte + 7-Tage-Forecast mit Konfidenzband
Modell:  GradientBoostingRegressor (sklearn), 200 Estimatoren
```

### 3. Nudging-Analyse — Interventionsmodell

Auf Basis des individuellen Verhaltensmusters berechnet das Modell optimale
Uhrzeiten für Push-Benachrichtigungen, um nachhaltige Mobilitätsentscheidungen
zu fördern (z. B. "S-Bahn statt Auto — +21 Pkte").

---

## Punktesystem

### Aktivitäten

| Aktivität | Punkte/Minute | Intensitätsmultiplikator |
|---|---|---|
| Laufen | 2,5 | 0,8 – 1,3 |
| Fahrradfahren | 1,8 | 0,8 – 1,3 |
| Fußball | 2,8 | 0,8 – 1,3 |
| Schwimmen | 2,0 | 0,8 – 1,3 |
| Walken | 1,2 | 0,8 – 1,3 |
| Yoga | 1,0 | 0,8 – 1,3 |

### Mobilität (Punkte/km)

| Verkehrsmittel | Punkte/km | CO₂-Ersparnis ggü. PKW |
|---|---|---|
|  Zu Fuß | 4 | 0,21 kg/km |
|  Fahrrad | 5 | 0,21 kg/km |
|  Straßenbahn (VVS) | 3 | 0,17 kg/km |
|  S-Bahn (VVS) | 3 | 0,18 kg/km |
|  Bus (VVS) | 3 | 0,12 kg/km |
|  E-Scooter | 2 | 0,10 kg/km |
|  Auto | 0 | — |

Streckenbonus: >10 km → +20 % Punkte

### Level-System

| Level | Titel | ab Punkten |
|---|---|---|
| 1 | Spaziergänger | 0 |
| 2 | Stadtläufer | 200 |
| 3 | Radpionier | 500 |
| 4 | ÖPNV-Champion | 1.000 |
| 5 | Klimaheld | 2.000 |
| 6 | Klimaheld+ | 3.500 |
| 7 | ActiveCity Pro | 5.000 |

---

## Installation & Start

```bash
# Repository klonen
git clone https://github.com/christianm38/stuttgart-activecity.git
cd stuttgart-activecity

# Virtuelle Umgebung (empfohlen)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run app.py
```

Die App öffnet sich automatisch unter `http://localhost:8501`.

---

## Projektstruktur

```
stuttgart-activecity/
├── app.py                          # Streamlit-Hauptanwendung
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   ├── activity_classifier.py      # Random Forest — Aktivitätserkennung
│   └── points_predictor.py         # Gradient Boosting — Punkteprognose
└── utils/
    ├── __init__.py
    ├── points_engine.py            # Punkte- & Level-Berechnung
    └── data_generator.py           # Synthetische Demodaten
```

---

## Produktionsarchitektur (Vision)

```
Health-Datenquellen          Backend                    Datenbank
────────────────────         ────────────────────       ──────────────
Apple HealthKit    ──┐       FastAPI / Django    ──┐    PostgreSQL
Google Fit         ──┤──►   ML-Inferenz-Service ──┤──► TimescaleDB
Garmin Connect     ──┘       VVS TRIAS API       ──┘    Redis (Cache)
                             ↕
                        Streamlit / React Native App
```

**VVS-Integration**: Live-Abfahrten via [VVS TRIAS-API](https://www.openvvs.de/dataset/trias) —
ÖPNV-Nutzung wird automatisch via NFC-Ticket-Scan erkannt und gutgeschrieben.

---

## Datenschutz

Alle Gesundheits- und Bewegungsdaten verbleiben lokal auf dem Gerät (Demo-Modus).
Für den Produktiveinsatz: DSGVO-konforme Datenspeicherung, Ende-zu-Ende-Verschlüsselung,
explizite Einwilligung vor Health-Sync.

---

## Autor

**Christian Mann** 
GitHub: [github.com/christianm38](https://github.com/christianm38)

---

## Lizenz

MIT License — siehe [LICENSE](LICENSE)
