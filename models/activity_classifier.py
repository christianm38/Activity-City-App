"""
Aktivitätsklassifikator — Random Forest
Erkennt Aktivitätstyp aus Sensordaten (Herzfrequenz, Schritttempo, Beschleunigung)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


ACTIVITY_LABELS = [
    "Laufen", "Fahrradfahren", "Walken", "Schwimmen",
    "Fußball", "Yoga", "ÖPNV-Fahrt", "Ruhe",
]

# Typische Sensormuster je Aktivität:
# [heart_rate, cadence (steps/min), acceleration (m/s²)]
ACTIVITY_PROFILES = {
    "Laufen":        (155, 165, 9.5),
    "Fahrradfahren": (130, 80,  6.0),
    "Walken":        (105, 110, 4.0),
    "Schwimmen":     (140, 40,  7.0),
    "Fußball":       (165, 140, 12.0),
    "Yoga":          (80,  20,  1.5),
    "ÖPNV-Fahrt":    (75,  10,  2.0),
    "Ruhe":          (65,  0,   0.5),
}


class ActivityClassifier:
    """Random-Forest-Klassifikator für Aktivitätserkennung."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=8
        )
        self.le = LabelEncoder()
        self._train()

    def _generate_training_data(self, n_per_class: int = 300):
        X, y = [], []
        for activity, (hr_mean, cad_mean, acc_mean) in ACTIVITY_PROFILES.items():
            for _ in range(n_per_class):
                hr  = np.random.normal(hr_mean,  hr_mean  * 0.08)
                cad = np.random.normal(cad_mean, cad_mean * 0.12 + 1)
                acc = np.random.normal(acc_mean, acc_mean * 0.15 + 0.3)
                X.append([hr, cad, acc])
                y.append(activity)
        return np.array(X), np.array(y)

    def _train(self):
        X, y = self._generate_training_data()
        y_enc = self.le.fit_transform(y)
        self.model.fit(X, y_enc)

    def classify(self, features: dict) -> dict:
        """Klassifikation aus einfachen Eingaben (für Manuallog)."""
        intensity_map = {"Locker": 0.7, "Moderat": 1.0, "Intensiv": 1.3}
        scale = intensity_map.get(features.get("intensity", "Moderat"), 1.0)
        hr  = features["heart_rate"] * scale
        cad = min(hr * 0.95, 200)
        acc = hr / 16

        return self._predict_from_sensors(hr, cad, acc)

    def classify_sensor(self, features: dict) -> dict:
        """Klassifikation aus rohen Sensordaten."""
        hr  = features.get("heart_rate",  120)
        cad = features.get("cadence",     80)
        acc = features.get("acceleration", 5.0)
        return self._predict_from_sensors(hr, cad, acc)

    def _predict_from_sensors(self, hr: float, cad: float, acc: float) -> dict:
        X = np.array([[hr, cad, acc]])
        probs_enc = self.model.predict_proba(X)[0]

        # Map zurück auf Aktivitätsnamen
        probabilities = {}
        for idx, prob in enumerate(probs_enc):
            label = self.le.inverse_transform([idx])[0]
            probabilities[label] = float(prob)

        best_label = max(probabilities, key=probabilities.get)
        return {
            "activity":     best_label,
            "confidence":   probabilities[best_label],
            "probabilities": dict(
                sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            ),
        }
