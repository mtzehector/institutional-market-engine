from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from market_engine.config import PROJECT_ROOT, Settings


class RateLimiter:
    def __init__(self, calls_per_minute: int) -> None:
        self.interval = 60.0 / calls_per_minute
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self) -> None:
        with self.lock:
            remaining = self.interval - (time.monotonic() - self.last_call)
            if remaining > 0:
                time.sleep(remaining)
            self.last_call = time.monotonic()


@dataclass
class FMPProvider:
    settings: Settings

    def __post_init__(self) -> None:
        self.base_url = "https://financialmodelingprep.com/stable"
        self.limiter = RateLimiter(self.settings.fmp_rate_limit_per_minute)
        self.cache_dir = PROJECT_ROOT / "cache_fmp"
        self.cache_dir.mkdir(exist_ok=True)

    def _get(self, endpoint: str, params: dict[str, Any]) -> Any:
        query = dict(params)
        query["apikey"] = self.settings.fmp_api_key
        last_error: Exception | None = None
        for attempt in range(1, 5):
            self.limiter.wait()
            try:
                response = requests.get(
                    f"{self.base_url}/{endpoint}", params=query, timeout=30
                )
                if response.status_code == 429:
                    time.sleep(61)
                    continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and (
                    payload.get("Error Message") or payload.get("error")
                ):
                    raise RuntimeError(str(payload))
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 4:
                    time.sleep(min(16, 2**attempt))
        raise RuntimeError(f"FMP falló: {last_error}")

    def historical_eod(
        self,
        symbol: str,
        start: date,
        end: date,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        cache = self.cache_dir / f"{symbol}_{start}_{end}.csv"
        fresh = cache.exists() and (
            time.time() - cache.stat().st_mtime
            <= self.settings.fmp_cache_hours * 3600
        )
        if fresh and not force_refresh:
            return pd.read_csv(cache, parse_dates=["date"])

        payload = self._get(
            "historical-price-eod/full",
            {"symbol": symbol, "from": str(start), "to": str(end)},
        )
        if not isinstance(payload, list) or not payload:
            raise RuntimeError(f"Sin histórico EOD para {symbol}")

        rows = [
            {
                "date": item.get("date"),
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close"),
                "volume": item.get("volume"),
                "vwap": item.get("vwap"),
            }
            for item in payload
            if item.get("date")
        ]
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        for column in ["open", "high", "low", "close", "volume", "vwap"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
        frame = frame.sort_values("date").reset_index(drop=True)
        frame.to_csv(cache, index=False)
        return frame
