from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urljoin

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException

from .base import BaseCityScraper, PermitDocument
from ..engine import browser, soup, wait_for
from .util import normalize_row_to_permit


class LAScraper(BaseCityScraper):
	def _fill_dates_and_submit(self, drv, start_date: str, end_date: str) -> None:
		start_name = self.config.date_params.get("start", "startDate")
		end_name = self.config.date_params.get("end", "endDate")
		for by, sel in [(By.NAME, start_name), (By.ID, start_name), (By.NAME, "from")]:
			try:
				el = drv.find_element(by, sel)
				el.clear(); el.send_keys(start_date)
				break
			except Exception:
				continue
		for by, sel in [(By.NAME, end_name), (By.ID, end_name), (By.NAME, "to")]:
			try:
				el = drv.find_element(by, sel)
				el.clear(); el.send_keys(end_date); el.send_keys(Keys.ENTER)
				break
			except Exception:
				continue
		for sel in [
			(By.CSS_SELECTOR, "input[type='submit']"),
			(By.XPATH, "//button[contains(., 'Search') or contains(., 'Submit')]"),
		]:
			try:
				drv.find_element(*sel).click(); break
			except Exception:
				continue

	def _parse_table(self, html: str) -> List[Dict[str, Any]]:
		page = soup(html)
		tables = page.find_all("table")
		if not tables:
			return []
		chosen = None
		for t in tables:
			head = t.find("th")
			if not head:
				continue
			text = head.get_text(" ", strip=True).lower()
			if any(k in text for k in ["permit", "number", "address", "issue"]):
				chosen = t; break
		if not chosen:
			chosen = tables[0]
		headers = [th.get_text(" ", strip=True) for th in chosen.find_all("th")]
		rows: List[Dict[str, Any]] = []
		for tr in chosen.find_all("tr"):
			tds = tr.find_all("td")
			if not tds:
				continue
			row = {headers[i] if i < len(headers) else f"col{i}": tds[i].get_text(" ", strip=True) for i in range(len(tds))}
			link = None
			for td in tds:
				link = td.find("a")
				if link:
					break
			if link and link.get("href"):
				row["detail_url"] = urljoin(self.config.documents_url, link.get("href"))
			rows.append(row)
		return rows

	def _goto_next_page(self, drv) -> bool:
		try:
			el = drv.find_element(By.LINK_TEXT, "Next")
			el.click(); return True
		except Exception:
			return False

	def search(self, start_date: str, end_date: str, limit: int = 100) -> Iterable[Dict[str, Any]]:
		results: List[Dict[str, Any]] = []
		with browser(headless=self.headless) as drv:
			drv.get(self.config.search_url)
			if self.debug:
				self.save_debug("initial_page.html", drv.page_source)
			self.polite_wait()
			self._fill_dates_and_submit(drv, start_date, end_date)
			if self.debug:
				self.save_debug("after_submit.html", drv.page_source)
			while True:
				try:
					wait_for(drv, By.TAG_NAME, "table", timeout=20)
				except Exception:
					break
				if self.debug:
					self.save_debug("results_page.html", drv.page_source)
				rows = self._parse_table(drv.page_source)
				for r in rows:
					results.append(r)
					if len(results) >= limit:
						return results
				if not self._goto_next_page(drv):
					break
				self.polite_wait()
		return results

	def parse_permit(self, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PermitDocument]]:
		permit = normalize_row_to_permit(self.config.symbol, raw)
		docs: List[PermitDocument] = []
		detail = raw.get("detail_url")
		if detail:
			try:
				with browser(headless=self.headless) as drv:
					drv.get(detail)
					h = drv.page_source
				s = soup(h)
				for a in s.find_all("a"):
					href = a.get("href")
					if not href:
						continue
					url = urljoin(self.config.documents_url, href)
					if any(ext in url.lower() for ext in [".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png"]):
						docs.append(PermitDocument(url=url))
			except Exception:
				pass
		return permit, docs

