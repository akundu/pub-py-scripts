from typing import Any, Dict, Iterable, List, Tuple
from .api_base import APICityScraper, PermitDocument


class ChicagoAPIScraper(APICityScraper):
	"""Chicago building permits via Open Data API"""
	
	def parse_permit(self, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PermitDocument]]:
		"""Parse Chicago-specific permit data"""
		permit = {
			'city': self.config.symbol,
			'permit_number': raw.get('id', raw.get('permit_number', 'unknown')),
			'issue_date': raw.get('issue_date', raw.get('permit_issue_date')),
			'address': raw.get('street_number', '') + ' ' + raw.get('street_direction', '') + ' ' + raw.get('street_name', '') + ' ' + raw.get('suffix', ''),
			'permit_type': raw.get('permit_type', raw.get('work_description', 'unknown')),
			'project_value': self._parse_value(raw.get('estimated_cost', raw.get('total_fees'))),
			'contractor': raw.get('contractor', raw.get('applicant', '')),
			'details_json': str(raw)
		}
		
		# Clean up address
		if permit['address']:
			permit['address'] = ' '.join(permit['address'].split())
		else:
			permit['address'] = ''
		
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
