import time
from contextlib import contextmanager
from typing import Iterator, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


def make_driver(headless: bool = True, user_agent: Optional[str] = None) -> webdriver.Chrome:
	options = Options()
	if headless:
		options.add_argument("--headless=new")
	options.add_argument("--no-sandbox")
	options.add_argument("--disable-dev-shm-usage")
	options.add_argument("--window-size=1920,1080")
	if user_agent:
		options.add_argument(f"--user-agent={user_agent}")
	driver = webdriver.Chrome(options=options)
	return driver


@contextmanager
def browser(headless: bool = True) -> Iterator[webdriver.Chrome]:
	drv = make_driver(headless=headless)
	try:
		yield drv
	finally:
		drv.quit()


def wait_for(drv: webdriver.Chrome, by: By, selector: str, timeout: int = 20):
	return WebDriverWait(drv, timeout).until(EC.presence_of_element_located((by, selector)))


def soup(html: str) -> BeautifulSoup:
	return BeautifulSoup(html, "html.parser")


def rate_limit_sleep(delay_seconds: float) -> None:
	time.sleep(max(0.0, delay_seconds))

