from typing import Any, Dict, Iterable, List, Tuple
from .api_base import APICityScraper, PermitDocument


class NYCAPIScraper(APICityScraper):
	"""NYC DOB permits via Open Data API"""
	
	def parse_permit(self, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PermitDocument]]:
		"""Parse NYC-specific permit data"""
		# Map NYC field names to our standard format (CSV uses underscores)
		permit = {
			'city': self.config.symbol,
			'permit_number': raw.get('job__', raw.get('job_s1_no', 'unknown')),
			'issue_date': self._find_best_date(raw),
			'address': f"{raw.get('house__', '')} {raw.get('street_name', '')} {raw.get('borough', '')}".strip(),
			'permit_type': raw.get('job_type', raw.get('job_description', 'unknown')),
			'project_value': self._parse_value(raw.get('initial_cost', raw.get('total_est__fee'))),
			'contractor': f"{raw.get('applicant_s_first_name', '')} {raw.get('applicant_s_last_name', '')}".strip(),
			'details_json': str(raw)
		}
		
		# Clean up address
		if permit['address'] and permit['address'] != ' ':
			permit['address'] = ' '.join(permit['address'].split())
		else:
			permit['address'] = ''
		
		docs: List[PermitDocument] = []
		return permit, docs
	
	def _find_best_date(self, raw: Dict[str, Any]) -> str:
		"""Find the best date field from NYC data"""
		date_fields = [
			'latest_action_date',
			'pre__filing_date', 
			'paid',
			'fully_permitted',
			'approved',
			'dobrundate',
			'Latest Action Date',
			'Pre- Filing Date', 
			'Paid',
			'Fully Permitted',
			'Approved',
			'DOBRunDate'
		]
		
		for field in date_fields:
			if field in raw and raw[field]:
				date_str = str(raw[field]).strip()
				if date_str and date_str != '':
					return date_str
		return None
	
	def _parse_value(self, value_str: str) -> float:
		"""Parse monetary value from string"""
		if not value_str:
			return None
		try:
			# Remove common currency symbols and commas
			cleaned = str(value_str).replace('$', '').replace(',', '').strip()
			return float(cleaned) if cleaned else None
		except (ValueError, TypeError):
			return None
