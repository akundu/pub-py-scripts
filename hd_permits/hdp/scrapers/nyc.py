from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urljoin
import json
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from .base import BaseCityScraper, PermitDocument
from ..engine import browser, soup, wait_for
from .util import normalize_row_to_permit


class NYCScraper(BaseCityScraper):
	def _navigate_to_search(self, drv) -> None:
		# Start at the main BIS page and navigate to the search
		main_url = "https://a810-bisweb.nyc.gov/bisweb/bispi00.jsp"
		drv.get(main_url)
		self.polite_wait()
		
		# Look for the "Building Information Search" link or similar
		try:
			search_link = drv.find_element(By.LINK_TEXT, "Building Information Search")
			search_link.click()
			self.polite_wait()
		except Exception:
			# Try alternative selectors
			for selector in [
				(By.PARTIAL_LINK_TEXT, "Building Information"),
				(By.PARTIAL_LINK_TEXT, "Search"),
				(By.XPATH, "//a[contains(@href, 'bispi')]")
			]:
				try:
					drv.find_element(*selector).click()
					self.polite_wait()
					break
				except Exception:
					continue

	def _fill_dates_and_submit(self, drv, start_date: str, end_date: str) -> None:
		# Try multiple approaches to find date fields
		date_selectors = [
			# Common NYC BIS date field patterns
			("from_date", "to_date"),
			("fromdate", "todate"), 
			("from", "to"),
			("start_date", "end_date"),
			("startdate", "enddate"),
			("datefrom", "dateto"),
			("date_from", "date_to")
		]
		
		start_filled = False
		end_filled = False
		
		for start_name, end_name in date_selectors:
			try:
				# Try by name first
				start_el = drv.find_element(By.NAME, start_name)
				start_el.clear()
				start_el.send_keys(start_date)
				start_filled = True
				break
			except Exception:
				continue
		
		if not start_filled:
			# Try by ID
			for start_name, end_name in date_selectors:
				try:
					start_el = drv.find_element(By.ID, start_name)
					start_el.clear()
					start_el.send_keys(start_date)
					start_filled = True
					break
				except Exception:
					continue
		
		for start_name, end_name in date_selectors:
			try:
				end_el = drv.find_element(By.NAME, end_name)
				end_el.clear()
				end_el.send_keys(end_date)
				end_filled = True
				break
			except Exception:
				continue
		
		if not end_filled:
			for start_name, end_name in date_selectors:
				try:
					end_el = drv.find_element(By.ID, end_name)
					end_el.clear()
					end_el.send_keys(end_date)
					end_filled = True
					break
				except Exception:
					continue
		
		# Try to submit the form
		submit_selectors = [
			(By.CSS_SELECTOR, "input[type='submit']"),
			(By.CSS_SELECTOR, "input[type='button'][value*='Search']"),
			(By.CSS_SELECTOR, "input[type='button'][value*='Query']"),
			(By.XPATH, "//input[@type='submit' or @type='button'][contains(@value,'Search') or contains(@value,'Query')]"),
			(By.XPATH, "//button[contains(text(),'Search') or contains(text(),'Query')]")
		]
		
		for selector in submit_selectors:
			try:
				drv.find_element(*selector).click()
				self.polite_wait()
				break
			except Exception:
				continue

	def _parse_table(self, html: str) -> List[Dict[str, Any]]:
		page = soup(html)
		
		# Check if we got an error page
		if "error" in html.lower() or "not found" in html.lower():
			if self.debug:
				self.save_debug("error_page.html", html)
			return []
		
		tables = page.find_all("table")
		if not tables:
			return []
		
		# Look for a table with actual data (not navigation/header tables)
		chosen = None
		for t in tables:
			# Skip tables that are clearly navigation/header
			text = t.get_text(" ", strip=True).lower()
			if any(skip in text for skip in ["menu", "navigation", "header", "footer", "logo"]):
				continue
			
			# Look for table with headers that suggest permit data
			headers = t.find_all("th")
			if headers:
				header_text = " ".join([h.get_text(" ", strip=True) for h in headers]).lower()
				if any(k in header_text for k in ["permit", "job", "number", "address", "date", "type", "applicant"]):
					chosen = t
					break
		
		if not chosen:
			# Fallback to largest table that's not clearly navigation
			largest_table = None
			largest_size = 0
			for t in tables:
				text = t.get_text(" ", strip=True).lower()
				if not any(skip in text for skip in ["menu", "navigation", "header", "footer", "logo"]):
					size = len(t.find_all("tr"))
					if size > largest_size:
						largest_size = size
						largest_table = t
			chosen = largest_table
		
		if not chosen:
			return []
		
		headers = [th.get_text(" ", strip=True) for th in chosen.find_all("th")]
		if self.debug:
			self.save_debug("results_headers.txt", "\n".join(headers))
		
		rows: List[Dict[str, Any]] = []
		for tr in chosen.find_all("tr"):
			tds = tr.find_all("td")
			if not tds or len(tds) < 2:  # Skip header rows and empty rows
				continue
			
			row = {headers[i] if i < len(headers) else f"col{i}": tds[i].get_text(" ", strip=True) for i in range(len(tds))}
			
			# Look for detail links
			link = None
			for td in tds:
				link = td.find("a")
				if link and link.get("href"):
					break
			
			if link and link.get("href"):
				row["detail_url"] = urljoin(self.config.documents_url, link.get("href"))
			
			rows.append(row)
		
		if self.debug and rows:
			self.save_debug("results_sample_row.json", json.dumps(rows[0], indent=2))
		
		return rows

	def _goto_next_page(self, drv) -> bool:
		# Look for pagination links
		next_selectors = [
			(By.LINK_TEXT, "Next"),
			(By.PARTIAL_LINK_TEXT, "Next"),
			(By.XPATH, "//a[contains(text(),'Next')]"),
			(By.XPATH, "//a[contains(text(),'>')]"),
			(By.XPATH, "//a[contains(text(),'More')]")
		]
		
		for selector in next_selectors:
			try:
				drv.find_element(*selector).click()
				self.polite_wait()
				return True
			except Exception:
				continue
		return False

	def search(self, start_date: str, end_date: str, limit: int = 100) -> Iterable[Dict[str, Any]]:
		results: List[Dict[str, Any]] = []
		with browser(headless=self.headless) as drv:
			# Navigate to search page first
			self._navigate_to_search(drv)
			
			# Fill dates and submit
			self._fill_dates_and_submit(drv, start_date, end_date)
			
			# Process results
			while True:
				try:
					wait_for(drv, By.TAG_NAME, "table", timeout=20)
				except Exception:
					break
				
				if self.debug:
					self.save_debug("results_page.html", drv.page_source)
				
				page_rows = self._parse_table(drv.page_source)
				for row in page_rows:
					results.append(row)
					if len(results) >= limit:
						return results
				
				# Try to go to next page
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
					try:
						wait_for(drv, By.TAG_NAME, "a", timeout=10)
					except Exception:
						pass
					if self.debug:
						self.save_debug("detail_page.html", drv.page_source)
					h = drv.page_source
				s = soup(h)
				for a in s.find_all("a"):
					href = a.get("href")
					if not href:
						continue
					url = urljoin(self.config.documents_url, href)
					text = a.get_text(" ", strip=True).lower()
					if any(ext in url.lower() for ext in [".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png"]) or "document" in text:
						docs.append(PermitDocument(url=url))
			except Exception:
				pass
		return permit, docs