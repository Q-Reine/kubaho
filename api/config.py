"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'kubaho.db'}"))

    mqtt_host: str | None = field(default_factory=lambda: os.getenv("MQTT_HOST") or None)
    mqtt_port: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    mqtt_username: str | None = field(default_factory=lambda: os.getenv("MQTT_USERNAME") or None)
    mqtt_password: str | None = field(default_factory=lambda: os.getenv("MQTT_PASSWORD") or None)
    mqtt_tls: bool = field(default_factory=lambda: _bool("MQTT_TLS"))

    # Africa's Talking SMS
    at_username: str = field(default_factory=lambda: os.getenv("AT_USERNAME", "sandbox"))
    at_api_key: str | None = field(default_factory=lambda: os.getenv("AT_API_KEY") or None)
    at_sender_id: str | None = field(default_factory=lambda: os.getenv("AT_SENDER_ID") or None)

    escalation_minutes: float = field(default_factory=lambda: float(os.getenv("ESCALATION_MINUTES", "5")))
    bathroom_dwell_minutes: float = field(default_factory=lambda: float(os.getenv("BATHROOM_DWELL_MINUTES", "30")))
    fall_dedupe_seconds: int = 60
    restless_energy_threshold: float = field(default_factory=lambda: float(os.getenv("RESTLESS_ENERGY_THRESHOLD", "0.3")))
    restless_z: float = 2.0
    restless_nights: int = 3
    baseline_nights: int = 7
    tz_offset_hours: int = 2  # Africa/Kigali, UTC+2, no daylight saving

    model_path: Path = field(default_factory=lambda: Path(os.getenv("MODEL_PATH", ROOT / "models" / "kubaho_lora_int8.onnx")))
    model_card_path: Path = field(default_factory=lambda: Path(os.getenv("MODEL_CARD", ROOT / "models" / "model_card.json")))
    cors_origins: list[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))


settings = Settings()
