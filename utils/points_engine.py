"""
Punktelogik — PointsEngine
Zentrale Berechnung aller StuttPunkte für Aktivitäten und Mobilität.
"""


ACTIVITY_BASE_POINTS = {
    "Laufen":                    2.5,   # Pkte/Minute
    "Walken":                    1.2,
    "Fahrradfahren":             1.8,
    "Schwimmen":                 2.0,
    "Fußball/Mannschaftssport":  2.8,
    "Yoga/Fitness":              1.0,
    "Klettern":                  2.2,
    "Inline-Skating":            1.6,
}

INTENSITY_MULTIPLIER = {
    "Locker":   0.8,
    "Moderat":  1.0,
    "Intensiv": 1.3,
}

# Punkte pro Kilometer je Verkehrsmittel
MOBILITY_POINTS_PER_KM = {
    " Zu Fuß":           4,
    " Fahrrad":           5,
    " Straßenbahn (VVS)": 3,
    " Bus (VVS)":         3,
    " S-Bahn (VVS)":      3,
    " E-Scooter":         2,
    " Auto (keine Punkte)": 0,
}

# CO₂-Ersparnis ggü. PKW (kg/km)
CO2_SAVED_PER_KM = {
    "🚶 Zu Fuß":           0.21,
    "🚲 Fahrrad":           0.21,
    "🚋 Straßenbahn (VVS)": 0.17,
    "🚌 Bus (VVS)":         0.12,
    "🚆 S-Bahn (VVS)":      0.18,
    "🛴 E-Scooter":         0.10,
    "🚗 Auto (keine Punkte)": 0.0,
}

LEVELS = [
    {"level": 1, "title": "Spaziergänger",  "min": 0,    "next": 200},
    {"level": 2, "title": "Stadtläufer",    "min": 200,  "next": 500},
    {"level": 3, "title": "Radpionier",     "min": 500,  "next": 1000},
    {"level": 4, "title": "ÖPNV-Champion",  "min": 1000, "next": 2000},
    {"level": 5, "title": "Stadtläufer",    "min": 2000, "next": 3500},
    {"level": 6, "title": "Klimaheld",      "min": 3500, "next": 5000},
    {"level": 7, "title": "ActiveCity Pro", "min": 5000, "next": 9999},
]


class PointsEngine:
    """Berechnet Punkte, Level und CO₂-Einsparung."""

    def calculate_points(
        self,
        activity: str,
        duration_min: int,
        intensity: str = "Moderat",
        streak_bonus: bool = False,
    ) -> int:
        base   = ACTIVITY_BASE_POINTS.get(activity, 1.5)
        mult   = INTENSITY_MULTIPLIER.get(intensity, 1.0)
        points = base * duration_min * mult
        if streak_bonus:
            points *= 1.20
        return max(1, round(points))

    def calculate_mobility_points(self, mode: str, km: float) -> int:
        ppm = MOBILITY_POINTS_PER_KM.get(mode, 0)
        # Streckenbonus: >10 km → +20 %
        bonus = 1.2 if km > 10 else 1.0
        return max(0, round(ppm * km * bonus))

    def calculate_co2(self, mode: str, km: float) -> float:
        return CO2_SAVED_PER_KM.get(mode, 0.0) * km

    def get_level(self, points: int) -> dict:
        for lvl in reversed(LEVELS):
            if points >= lvl["min"]:
                return lvl
        return LEVELS[0]
