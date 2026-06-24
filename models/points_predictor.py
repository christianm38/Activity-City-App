"""
Punkteprognose — Gradient Boosting Regressor
Prognostiziert erreichbare StuttPunkte für die nächsten 7 Tage
basierend auf dem bisherigen Aktivitätsverlauf.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta


class PointsPredictor:
    """
    Gradient-Boosting-Modell zur Punkteprognose.
    Features: Wochentag, rollender 3-Tage-Durchschnitt, Streak-Länge,
              vorheriger Tagespunktewert, Saisonalität.
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42,
        )
        self.is_fitted = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        pts = df["points_earned"].values
        features = []
        for i in range(len(pts)):
            dow         = i % 7
            lag1        = pts[i - 1] if i > 0 else pts[0]
            lag2        = pts[i - 2] if i > 1 else pts[0]
            roll3       = np.mean(pts[max(0, i - 3) : i + 1])
            streak      = min(i + 1, 7)
            weekday_sin = np.sin(2 * np.pi * dow / 7)
            weekday_cos = np.cos(2 * np.pi * dow / 7)
            features.append([lag1, lag2, roll3, streak, weekday_sin, weekday_cos])
        return np.array(features)

    def fit(self, df: pd.DataFrame):
        if len(df) < 4:
            return
        X = self._build_features(df)
        y = df["points_earned"].values
        # Leave-last-out: trainiere auf allen außer letztem Tag
        self.model.fit(X[:-1], y[:-1])
        self.is_fitted = True

    def predict_tomorrow(self, df: pd.DataFrame) -> dict:
        pts  = df["points_earned"].values
        lag1 = pts[-1]
        lag2 = pts[-2] if len(pts) > 1 else pts[-1]
        roll3 = np.mean(pts[-3:])
        streak = min(len(pts), 7)
        dow = (len(pts)) % 7
        X = np.array([[lag1, lag2, roll3, streak,
                        np.sin(2 * np.pi * dow / 7),
                        np.cos(2 * np.pi * dow / 7)]])

        if self.is_fitted:
            pred_pts = float(self.model.predict(X)[0])
        else:
            pred_pts = float(np.mean(pts[-3:]) * 1.05)

        pred_pts = max(10, round(pred_pts))

        activities = ["Laufen", "Fahrradfahren", "ÖPNV + Walken", "Schwimmen", "Yoga + Rad"]
        departures = ["07:30", "08:00", "07:45", "12:00", "17:45"]
        idx = dow % len(activities)

        return {
            "points":               pred_pts,
            "recommended_activity": activities[idx],
            "best_departure":       departures[idx],
        }

    def forecast_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prognostiziert die nächsten 7 Tage mit Konfidenzband."""
        pts = list(df["points_earned"].values)
        last_date = datetime.now()

        rows = []
        for i in range(7):
            future_date = last_date + timedelta(days=i + 1)
            dow  = future_date.weekday()
            lag1 = pts[-1]
            lag2 = pts[-2] if len(pts) > 1 else pts[-1]
            roll3 = np.mean(pts[-3:])
            streak = min(len(pts), 7)
            X = np.array([[lag1, lag2, roll3, streak,
                           np.sin(2 * np.pi * dow / 7),
                           np.cos(2 * np.pi * dow / 7)]])

            if self.is_fitted:
                pred = float(self.model.predict(X)[0])
            else:
                pred = float(np.mean(pts[-3:]) * (1 + 0.03 * i))

            pred = max(10, round(pred))
            noise = pred * 0.15

            rows.append({
                "day":       future_date.strftime("%a %d.%m."),
                "predicted": pred,
                "lower":     max(0, pred - noise),
                "upper":     pred + noise,
            })
            pts.append(pred)

        return pd.DataFrame(rows)
