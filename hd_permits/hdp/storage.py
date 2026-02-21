import os
import time
import pathlib
from typing import Optional
from urllib.request import urlopen, Request
import asyncio

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def build_permit_dir(city: str, issue_date: Optional[str], permit_type: str, permit_number: str) -> str:
	date_part = issue_date or time.strftime("%Y-%m-%d")
	path = os.path.join(BASE_DATA_DIR, city, date_part, permit_type or "unknown", permit_number)
	os.makedirs(path, exist_ok=True)
	return path


def download_file(url: str, dest_dir: str, filename: Optional[str] = None, timeout: int = 30) -> Optional[str]:
	try:
		local_name = filename or url.split("/")[-1].split("?")[0]
		if not local_name:
			local_name = f"download_{int(time.time()*1000)}"
		path = os.path.join(dest_dir, local_name)
		req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
		with urlopen(req, timeout=timeout) as r, open(path, "wb") as f:
			while True:
				chunk = r.read(8192)
				if not chunk:
					break
				f.write(chunk)
		return path
	except Exception:
		return None


async def download_file_async(url: str, dest_dir: str, filename: Optional[str] = None, timeout: int = 30) -> Optional[str]:
	return await asyncio.to_thread(download_file, url, dest_dir, filename, timeout)


def file_size(path: str) -> Optional[int]:
	try:
		return pathlib.Path(path).stat().st_size
	except Exception:
		return None

