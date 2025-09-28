import json
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CityConfig:
	name: str
	symbol: str
	api_url: str
	bulk_url: str
	documents_url: str
	date_params: Dict[str, str]
	rate_limit: Dict[str, Any]


def load_cities_config(config_path: str) -> Dict[str, CityConfig]:
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Config not found: {config_path}")
	with open(config_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	cities: Dict[str, CityConfig] = {}
	for symbol, cfg in data.items():
		cities[symbol] = CityConfig(
			name=cfg.get("name", symbol),
			symbol=symbol,
			api_url=cfg.get("api_url", ""),
			bulk_url=cfg.get("bulk_url", ""),
			documents_url=cfg.get("documents_url", ""),
			date_params=cfg.get("date_params", {}),
			rate_limit=cfg.get("rate_limit", {"delay_seconds": 1.0, "max_retries": 3}),
		)
	return cities

