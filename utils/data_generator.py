"""
Synthetische Datengenerierung für Demonstrations- und Entwicklungszwecke.
In der Produktion: ersetzt durch Health-API + VVS-TRIAS-API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_week_data(n_days: int = 7) -> pd.DataFrame:
    """Generiert synthetische Aktivitätsdaten für die letzten n_days Tage."""
    random.seed(42)
    np.random.seed(42)

    base_steps = 7200
    base_points = 65

    rows = []
    for i in range(n_days):
        date = (datetime.now() - timedelta(days=n_days - 1 - i)).strftime("%Y-%m-%d")
        day_label = (datetime.now() - timedelta(days=n_days - 1 - i)).strftime("%a")

        # Wochenende-Boost
        weekday = (datetime.now() - timedelta(days=n_days - 1 - i)).weekday()
        weekend_factor = 1.35 if weekday >= 5 else 1.0

        steps        = int(np.random.normal(base_steps * weekend_factor, 1200))
        active_min   = int(np.random.normal(45 * weekend_factor, 12))
        points_earned = int(np.random.normal(base_points * weekend_factor, 15))
        calories     = int(steps * 0.065 + np.random.normal(0, 30))

        rows.append({
            "date":          date,
            "day":           day_label,
            "steps":         max(500, steps),
            "active_min":    max(5, active_min),
            "points_earned": max(5, points_earned),
            "calories":      max(50, calories),
        })

    return pd.DataFrame(rows)


def generate_mobility_data(n_days: int = 7) -> pd.DataFrame:
    """Generiert synthetische Mobilitätsdaten für die letzten n_days Tage."""
    random.seed(7)
    np.random.seed(7)

    modes = [
        ("Zu Fuß", 4, 0.21),
        ("Fahrrad", 5, 0.21),
        ("Straßenbahn (VVS)", 3, 0.17),
        ("Bus (VVS)", 3, 0.12),
        ("S-Bahn (VVS)", 3, 0.18),
        ("E-Scooter", 2, 0.10),
        ("Auto", 0, 0.0),
    ]

    rows = []
    for i in range(n_days):
        date = (datetime.now() - timedelta(days=n_days - 1 - i)).strftime("%Y-%m-%d")
        # 2–4 Fahrten pro Tag
        n_trips = random.randint(2, 4)
        for _ in range(n_trips):
            mode_choice = random.choices(
                modes,
                weights=[0.2, 0.2, 0.15, 0.1, 0.15, 0.05, 0.15],
                k=1,
            )[0]
            mode_name, ppm, co2_rate = mode_choice
            km = round(random.uniform(0.5, 15.0), 1)
            pts = round(ppm * km * (1.2 if km > 10 else 1.0))
            co2 = round(co2_rate * km, 3)

            rows.append({
                "date":        date,
                "mode":        mode_name,
                "km":          km,
                "points":      pts,
                "co2_saved_kg": co2,
                "departure":   f"{random.randint(6,20):02d}:{random.choice(['00','15','30','45'])}",
            })

    return pd.DataFrame(rows)
