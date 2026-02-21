import os
import sqlite3
from contextlib import contextmanager
from typing import Dict, Any, Iterable, Optional
import asyncio

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "permits.sqlite3")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS permits (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	city TEXT NOT NULL,
	permit_number TEXT NOT NULL,
	issue_date TEXT,
	address TEXT,
	permit_type TEXT,
	project_value REAL,
	contractor TEXT,
	details_json TEXT,
	unique(city, permit_number)
);

CREATE TABLE IF NOT EXISTS documents (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	permit_id INTEGER NOT NULL,
	filename TEXT NOT NULL,
	file_ext TEXT NOT NULL,
	file_size_bytes INTEGER,
	url TEXT,
	FOREIGN KEY (permit_id) REFERENCES permits(id) ON DELETE CASCADE
);
"""


def ensure_db(db_path: Optional[str] = None) -> str:
	path = db_path or DB_FILE
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	with sqlite3.connect(path) as conn:
		conn.executescript(SCHEMA_SQL)
		conn.commit()
	return path


@contextmanager
def get_conn(db_path: Optional[str] = None):
	path = db_path or DB_FILE
	conn = sqlite3.connect(path)
	try:
		yield conn
	finally:
		conn.close()


def upsert_permit(conn: sqlite3.Connection, permit: Dict[str, Any]) -> int:
	cur = conn.cursor()
	cur.execute(
		"""
		INSERT INTO permits (city, permit_number, issue_date, address, permit_type, project_value, contractor, details_json)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(city, permit_number) DO UPDATE SET
			issue_date=excluded.issue_date,
			address=excluded.address,
			permit_type=excluded.permit_type,
			project_value=excluded.project_value,
			contractor=excluded.contractor,
			details_json=excluded.details_json
		""",
		(
			permit.get("city"),
			permit.get("permit_number"),
			permit.get("issue_date"),
			permit.get("address"),
			permit.get("permit_type"),
			permit.get("project_value"),
			permit.get("contractor"),
			permit.get("details_json"),
		),
	)
	conn.commit()
	return cur.lastrowid or cur.execute(
		"SELECT id FROM permits WHERE city=? AND permit_number=?",
		(permit.get("city"), permit.get("permit_number")),
	).fetchone()[0]


def insert_documents(conn: sqlite3.Connection, permit_id: int, docs: Iterable[Dict[str, Any]]) -> None:
	cur = conn.cursor()
	cur.executemany(
		"""
		INSERT INTO documents (permit_id, filename, file_ext, file_size_bytes, url)
		VALUES (?, ?, ?, ?, ?)
		""",
		[
			(
				permit_id,
				d.get("filename"),
				d.get("file_ext", ""),
				d.get("file_size_bytes"),
				d.get("url"),
			)
			for d in docs
		],
	)
	conn.commit()


# Transaction helpers

def upsert_permit_tx(permit: Dict[str, Any]) -> int:
	ensure_db()
	with get_conn() as conn:
		return upsert_permit(conn, permit)


def insert_documents_tx(permit_id: int, docs: Iterable[Dict[str, Any]]) -> None:
	ensure_db()
	with get_conn() as conn:
		insert_documents(conn, permit_id, docs)


# Async wrappers

async def upsert_permit_tx_async(permit: Dict[str, Any]) -> int:
	return await asyncio.to_thread(upsert_permit_tx, permit)


async def insert_documents_tx_async(permit_id: int, docs: Iterable[Dict[str, Any]]) -> None:
	await asyncio.to_thread(insert_documents_tx, permit_id, docs)

