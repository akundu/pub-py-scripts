from typing import Any, Dict, Iterable, List, Tuple
from .api_base import APICityScraper, PermitDocument


class LAAPIScraper(APICityScraper):
	"""Los Angeles building permits via Open Data API"""
	
	def parse_permit(self, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PermitDocument]]:
		"""Parse LA-specific permit data"""
		permit = {
			'city': self.config.symbol,
			'permit_number': raw.get('permit_number', raw.get('id', 'unknown')),
			'issue_date': raw.get('issue_date', raw.get('permit_issue_date')),
			'address': raw.get('address', raw.get('property_address', '')),
			'permit_type': raw.get('permit_type', raw.get('work_description', 'unknown')),
			'project_value': self._parse_value(raw.get('project_value', raw.get('estimated_cost'))),
			'contractor': raw.get('contractor', raw.get('applicant', '')),
			'details_json': str(raw)
		}
		
		docs: List[PermitDocument] = []
		return permit, docs
	
	def _parse_value(self, value_str: str) -> float:
		"""Parse monetary value from string"""
		if not value_str:
			return None
		try:
			cleaned = str(value_str).replace('$', '').replace(',', '').strip()
			return float(cleaned) if cleaned else None
		except (ValueError, TypeError):
			return None
