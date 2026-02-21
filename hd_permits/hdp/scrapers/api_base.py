from typing import Any, Dict, Iterable, List, Tuple, Optional
import requests
import csv
import io
from datetime import datetime, timedelta
from urllib.parse import urlencode

from .base import BaseCityScraper, PermitDocument


class APICityScraper(BaseCityScraper):
	"""Base class for API-based city scrapers using open data portals"""
	
	def __init__(self, config, headless: bool = True, debug: bool = False):
		super().__init__(config, headless, debug)
		self.session = requests.Session()
		self.session.headers.update({
			'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
		})

	def _download_bulk_data(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
		"""Download bulk CSV data from open data portal"""
		if not hasattr(self.config, 'bulk_url') or not self.config.bulk_url:
			return []
		
		try:
			response = self.session.get(self.config.bulk_url, timeout=30)
			response.raise_for_status()
			
			# Parse CSV data
			csv_data = []
			reader = csv.DictReader(io.StringIO(response.text))
			for row in reader:
				csv_data.append(row)
			
			if self.debug:
				self.save_debug("bulk_data_sample.json", str(csv_data[:3]))
			
			return csv_data
		except Exception as e:
			if self.debug:
				self.save_debug("bulk_download_error.txt", str(e))
			return []

	def _filter_by_date_range(self, data: List[Dict[str, Any]], start_date: str, end_date: str) -> List[Dict[str, Any]]:
		"""Filter data by date range"""
		filtered = []
		start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
		end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
		
		for i, row in enumerate(data):
			# Try different date field names
			date_field = None
			for field in ['issue_date', 'issued_date', 'permit_issue_date', 'date_issued', 'created_date', 'filing_date', 'latest_action_date', 'pre__filing_date', 'paid', 'fully_permitted', 'approved', 'dobrundate', 'Latest Action Date', 'Pre- Filing Date', 'Paid', 'Fully Permitted', 'Approved', 'DOBRunDate']:
				if field in row and row[field]:
					date_field = field
					break
			
			if not date_field:
				if self.debug and i < 3:  # Debug first few rows
					self.save_debug("no_date_field.txt", f"Row {i}: No date field found. Available fields: {list(row.keys())[:10]}")
				continue
			
			try:
				# Parse various date formats
				row_date = None
				date_str = str(row[date_field]).strip()
				if not date_str:
					continue
				
				# Try different date formats
				for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
					try:
						row_date = datetime.strptime(date_str.split('T')[0], fmt).date()
						break
					except ValueError:
						continue
				
				if row_date and start_dt <= row_date <= end_dt:
					filtered.append(row)
				elif self.debug and len(filtered) < 3:  # Debug first few attempts
					self.save_debug("date_debug.txt", f"Date: {date_str}, Parsed: {row_date}, Start: {start_dt}, End: {end_dt}, Match: {row_date and start_dt <= row_date <= end_dt if row_date else False}")
			except Exception as e:
				if self.debug:
					self.save_debug("date_error.txt", f"Error parsing date {date_str}: {e}")
				continue
		
		return filtered

	def search(self, start_date: str, end_date: str, limit: int = 100) -> Iterable[Dict[str, Any]]:
		"""Search for permits using bulk download"""
		# Download bulk data
		bulk_data = self._download_bulk_data(start_date, end_date)
		if self.debug:
			self.save_debug("bulk_data_count.txt", f"Downloaded {len(bulk_data)} records")
		
		if not bulk_data:
			return []
		
		# Filter by date range
		filtered_data = self._filter_by_date_range(bulk_data, start_date, end_date)
		if self.debug:
			self.save_debug("filtered_data_count.txt", f"Filtered to {len(filtered_data)} records for {start_date} to {end_date}")
		
		# Limit results
		return filtered_data[:limit]

	def parse_permit(self, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PermitDocument]]:
		"""Parse raw permit data into standardized format"""
		# Map common field names to standardized fields
		field_mapping = {
			'permit_number': ['permit_number', 'permit_id', 'id', 'permit_no', 'job_number'],
			'issue_date': ['issue_date', 'issued_date', 'permit_issue_date', 'date_issued', 'created_date'],
			'address': ['address', 'property_address', 'location', 'street_address', 'site_address'],
			'permit_type': ['permit_type', 'work_type', 'permit_class', 'type', 'description'],
			'project_value': ['project_value', 'estimated_value', 'cost', 'value', 'estimated_cost'],
			'contractor': ['contractor', 'applicant', 'contractor_name', 'applicant_name', 'licensee']
		}
		
		permit = {
			'city': self.config.symbol,
			'permit_number': 'unknown',
			'issue_date': None,
			'address': '',
			'permit_type': 'unknown',
			'project_value': None,
			'contractor': '',
			'details_json': None
		}
		
		# Map fields
		for standard_field, possible_fields in field_mapping.items():
			for field in possible_fields:
				if field in raw and raw[field]:
					permit[standard_field] = str(raw[field]).strip()
					break
		
		# Store raw data as JSON
		permit['details_json'] = str(raw)
		
		# No documents available from bulk API
		docs: List[PermitDocument] = []
		
		return permit, docs
