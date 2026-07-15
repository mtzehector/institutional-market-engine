from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    fmp_api_key: str
    fmp_rate_limit_per_minute: int = 250
    fmp_workers: int = 12
    fmp_cache_hours: int = 18
    smart_money_length: int = 14
    smart_money_volume_length: int = 20
    smart_money_ema: int = 5
    minimum_delta_smart_money: float = 0.25

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.getenv("FMP_API_KEY", "").strip()
        if not key:
            raise RuntimeError("FMP_API_KEY no está configurada en .env")

        rate_limit = int(os.getenv("FMP_RATE_LIMIT_PER_MINUTE", "250"))
        if not 1 <= rate_limit <= 250:
            raise ValueError("FMP_RATE_LIMIT_PER_MINUTE debe estar entre 1 y 250")

        return cls(
            fmp_api_key=key,
            fmp_rate_limit_per_minute=rate_limit,
            fmp_workers=int(os.getenv("FMP_WORKERS", "12")),
            fmp_cache_hours=int(os.getenv("FMP_CACHE_HOURS", "18")),
            smart_money_length=int(os.getenv("SMART_MONEY_LENGTH", "14")),
            smart_money_volume_length=int(os.getenv("SMART_MONEY_VOLUME_LENGTH", "20")),
            smart_money_ema=int(os.getenv("SMART_MONEY_EMA", "5")),
            minimum_delta_smart_money=float(os.getenv("MIN_DELTA_SM", "0.25")),
        )
