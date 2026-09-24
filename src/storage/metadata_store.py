import sqlite3
import json
from typing import List, Dict, Any, Optional
import contextlib

class MetadataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextlib.contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS code_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    repository_id TEXT,
                    file_path TEXT,
                    language TEXT,
                    chunk_type TEXT,
                    class_name TEXT,
                    function_name TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    imports TEXT,
                    calls TEXT,
                    code TEXT,
                    representation TEXT,
                    commit_hash TEXT
                )
            ''')
            # Create indexes for fast lookup as required
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON code_chunks(file_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_function_name ON code_chunks(function_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_class_name ON code_chunks(class_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_commit_hash ON code_chunks(commit_hash)')
            conn.commit()

    def insert_chunk(self, metadata: Dict[str, Any], representation: str):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO code_chunks (
                    chunk_id, repository_id, file_path, language, chunk_type, 
                    class_name, function_name, start_line, end_line, 
                    imports, calls, code, representation, commit_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.get("chunk_id"),
                metadata.get("repository_id"),
                metadata.get("file_path"),
                metadata.get("language"),
                metadata.get("chunk_type"),
                metadata.get("class_name"),
                metadata.get("function_name"),
                metadata.get("start_line"),
                metadata.get("end_line"),
                json.dumps(metadata.get("imports", [])),
                json.dumps(metadata.get("function_calls", [])),
                metadata.get("code"),
                representation,
                metadata.get("git_commit")
            ))
            conn.commit()
            
    def insert_chunks(self, metadata_list: List[Dict[str, Any]], representations: List[str]):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            rows = []
            for metadata, rep in zip(metadata_list, representations):
                rows.append((
                    metadata.get("chunk_id"),
                    metadata.get("repository_id"),
                    metadata.get("file_path"),
                    metadata.get("language"),
                    metadata.get("chunk_type"),
                    metadata.get("class_name"),
                    metadata.get("function_name"),
                    metadata.get("start_line"),
                    metadata.get("end_line"),
                    json.dumps(metadata.get("imports", [])),
                    json.dumps(metadata.get("function_calls", [])),
                    metadata.get("code"),
                    rep,
                    metadata.get("git_commit")
                ))
                
            cursor.executemany('''
                INSERT OR REPLACE INTO code_chunks (
                    chunk_id, repository_id, file_path, language, chunk_type, 
                    class_name, function_name, start_line, end_line, 
                    imports, calls, code, representation, commit_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)
            conn.commit()

    def get_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM code_chunks WHERE chunk_id = ?', (chunk_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def search_by_function(self, function_name: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM code_chunks WHERE function_name = ?', (function_name,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def search_by_class(self, class_name: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM code_chunks WHERE class_name = ?', (class_name,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def search_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM code_chunks WHERE file_path = ?', (file_path,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["imports"] = json.loads(d["imports"]) if d["imports"] else []
        
        if "calls" in d:
            d["function_calls"] = json.loads(d["calls"]) if d["calls"] else []
            del d["calls"]
            
        d["git_commit"] = d["commit_hash"]
        return d
