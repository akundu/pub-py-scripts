from typing import Any, Dict, Iterable, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
import os
import time
from ..config import CityConfig
from ..engine import rate_limit_sleep


@dataclass
class PermitDocument:
	url: str
	filename: Optional[str] = None


class BaseCityScraper:
	def __init__(self, config: CityConfig, headless: bool = True, debug: bool = False):
		self.config = config
		self.headless = headless
		self.debug = debug

	def search(self, start_date: str, end_date: str, limit: int = 100) -> Iterable[Dict[str, Any]]:
		raise NotImplementedError

	def parse_permit(self, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PermitDocument]]:
		raise NotImplementedError

	def polite_wait(self):
		delay = float(self.config.rate_limit.get("delay_seconds", 1.0))
		rate_limit_sleep(delay)

	# Async wrappers
	async def async_search(self, start_date: str, end_date: str, limit: int = 100) -> Iterable[Dict[str, Any]]:
		return await asyncio.to_thread(self.search, start_date, end_date, limit)

	async def async_parse_permit(self, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PermitDocument]]:
		return await asyncio.to_thread(self.parse_permit, raw)

	def save_debug(self, name: str, content: str) -> Optional[str]:
		if not self.debug:
			return None
		# project_root/data/debug/<city>
		project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
		base = os.path.join(project_root, "data", "debug", self.config.symbol)
		os.makedirs(base, exist_ok=True)
		ts = int(time.time())
		path = os.path.join(base, f"{ts}_{name}")
		with open(path, "w", encoding="utf-8") as f:
			f.write(content)
		return path
