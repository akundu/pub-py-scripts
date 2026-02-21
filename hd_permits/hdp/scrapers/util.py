from typing import Dict, Any, Optional
import re

# Heuristic maps for header detection
HEADER_ALIASES = {
	"permit_number": [r"^permit\b", r"^job\b", r"\bnumber\b", r"^app(?:lication)?\s*#?$"] ,
	"issue_date": [r"issue\s*date", r"issued", r"date"],
	"address": [r"address", r"location", r"house\s*#", r"street"],
	"permit_type": [r"type", r"permit\s*type", r"work\s*type"],
	"project_value": [r"(job|project|estimated)\s*value", r"valuation", r"cost"],
	"contractor": [r"contractor", r"applicant", r"licensee", r"builder"],
}


def _find_key(row: Dict[str, Any], patterns) -> Optional[str]:
	for k in row.keys():
		lk = (k or "").strip().lower()
		for pat in patterns:
			if re.search(pat, lk):
				return k
	return None


def normalize_row_to_permit(city_symbol: str, row: Dict[str, Any]) -> Dict[str, Any]:
	permit_number = None
	key = _find_key(row, HEADER_ALIASES["permit_number"]) ; permit_number = row.get(key) if key else None
	issue_date = None
	key = _find_key(row, HEADER_ALIASES["issue_date"]) ; issue_date = row.get(key) if key else None
	address = None
	key = _find_key(row, HEADER_ALIASES["address"]) ; address = row.get(key) if key else None
	permit_type = None
	key = _find_key(row, HEADER_ALIASES["permit_type"]) ; permit_type = row.get(key) if key else None
	project_value = None
	key = _find_key(row, HEADER_ALIASES["project_value"]) ; project_value = row.get(key) if key else None
	contractor = None
	key = _find_key(row, HEADER_ALIASES["contractor"]) ; contractor = row.get(key) if key else None

	return {
		"city": city_symbol,
		"permit_number": (permit_number or "unknown").strip() or "unknown",
		"issue_date": (issue_date or None),
		"address": (address or "").strip(),
		"permit_type": (permit_type or "unknown").strip() or "unknown",
		"project_value": _coerce_value(project_value),
		"contractor": (contractor or "").strip(),
		"details_json": None,
	}


def _coerce_value(val: Any) -> Optional[float]:
	if val is None:
		return None
	try:
		s = str(val).replace(",", "").replace("$", "").strip()
		return float(s) if s else None
	except Exception:
		return None
