# ============================================================================
# POSTGRESQL DOCUMENT REPOSITORY (backend/services/document_repository.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Manages database storage for sales playbooks and PDF policy documents.
#
# KEY FEATURES:
# 1. Binary PDF Storage: Stores original file content in PostgreSQL BYTEA format
#    so users can view and download original files from the UI at any time.
# 2. Metadata Tracking: Records filename, file size, page count, total characters,
#    and processing status ('processing', 'completed', 'failed').
# 3. Vector Tracking: Stores generated Pinecone vector IDs so when a document
#    is deleted, its vectors can be cleanly purged from the vector database.
# 4. Auto Table Creation: Ensures 'documents' table exists automatically.
# ============================================================================

import os
import json
import uuid
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables from backend/.env if available
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

from services.lead_repository import (
    DatabaseError,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseOperationError,
)

logger = logging.getLogger(__name__)



class DocumentRepository:
    """
    PostgreSQL repository for storing and managing uploaded PDF documents.
    Table: 'documents'
    Stores original binary PDF files (BYTEA), metadata, extracted text, and Pinecone vector tracking IDs.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dbname: Optional[str] = None,
    ):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.host = host or os.getenv("PGHOST", "127.0.0.1")
        self.port = port or int(os.getenv("PGPORT", 5432))
        self.user = user or os.getenv("PGUSER", "postgres")
        self.password = password if password is not None else os.getenv("PGPASSWORD", "")
        self.dbname = dbname or os.getenv("PGDATABASE", "voice_sales_copilot")

    def get_connection(self):
        """
        Establishes and returns a connection to the PostgreSQL database.
        """
        try:
            if self.host and self.user and self.dbname:
                return psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    dbname=self.dbname,
                )
            elif self.database_url:
                return psycopg2.connect(self.database_url)
            else:
                raise DatabaseConfigurationError("No valid PostgreSQL credentials or DATABASE_URL provided.")
        except psycopg2.OperationalError as oe:
            logger.error(f"PostgreSQL connection error: {str(oe)}")
            raise DatabaseConnectionError(f"Failed to connect to PostgreSQL database: {str(oe)}") from oe
        except DatabaseConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected database connection failure: {str(e)}")
            raise DatabaseConfigurationError(f"Invalid database configuration: {str(e)}") from e

    def ensure_documents_table(self) -> None:
        """
        Ensures the 'documents' table exists in PostgreSQL.
        Creates it automatically with all required fields and indexes if missing.
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            document_uid VARCHAR(100) UNIQUE NOT NULL,
            filename VARCHAR(255) NOT NULL,
            mime_type VARCHAR(100) NOT NULL DEFAULT 'application/pdf',
            file_size BIGINT NOT NULL,
            file_data BYTEA NOT NULL,
            upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            extracted_text TEXT,
            total_pages INTEGER DEFAULT 0,
            total_characters INTEGER DEFAULT 0,
            total_chunks INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}'::jsonb,
            pinecone_namespace VARCHAR(100),
            pinecone_doc_ids TEXT[],
            pinecone_upserted_count INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_documents_uid ON documents(document_uid);
        CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
            logger.info("Successfully verified 'documents' table in PostgreSQL.")
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error initializing 'documents' table: {str(e)}")
            raise DatabaseOperationError(f"Failed to initialize 'documents' table: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    def create_document(
        self,
        filename: str,
        file_bytes: bytes,
        mime_type: str = "application/pdf",
        document_uid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Inserts a new PDF document into PostgreSQL with its binary content.

        Args:
            filename: Original uploaded file name
            file_bytes: Raw binary bytes of the PDF file
            mime_type: File MIME type (default 'application/pdf')
            document_uid: Optional unique identifier string (generated if omitted)
            metadata: Optional initial metadata dictionary

        Returns:
            Dict containing document metadata and unique ID (excluding heavy raw bytes).
        """
        if not file_bytes:
            raise ValueError("File bytes cannot be empty.")

        clean_filename = (filename or "document.pdf").strip()
        uid = document_uid or f"doc_{uuid.uuid4().hex[:12]}"
        meta_json = json.dumps(metadata or {})
        file_size = len(file_bytes)

        insert_sql = """
        INSERT INTO documents (
            document_uid,
            filename,
            mime_type,
            file_size,
            file_data,
            metadata
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, document_uid, filename, mime_type, file_size, upload_date, created_at;
        """

        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    insert_sql,
                    (
                        uid,
                        clean_filename,
                        mime_type or "application/pdf",
                        file_size,
                        psycopg2.Binary(file_bytes),
                        meta_json,
                    ),
                )
                row = cur.fetchone()
            conn.commit()

            if not row:
                raise DatabaseOperationError("Failed to insert document: no row returned.")

            # Format timestamps to ISO strings
            res = dict(row)
            for k in ["upload_date", "created_at"]:
                if isinstance(res.get(k), datetime):
                    res[k] = res[k].isoformat()

            logger.info(f"Persisted document to PostgreSQL: id={res['id']}, uid={res['document_uid']}, file={clean_filename}")
            return res

        except DatabaseError:
            if conn:
                conn.rollback()
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error persisting document '{clean_filename}': {str(e)}")
            raise DatabaseOperationError(f"Failed to persist document to PostgreSQL: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    def update_document_extraction(
        self,
        document_id: int,
        extracted_text: str,
        total_pages: int,
        total_characters: int,
        total_chunks: int,
        metadata: Optional[Dict[str, Any]] = None,
        pinecone_namespace: Optional[str] = None,
        pinecone_doc_ids: Optional[List[str]] = None,
        pinecone_upserted_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Updates an existing document record with extracted text, chunk statistics,
        and Pinecone vector tracking metadata.
        """
        update_sql = """
        UPDATE documents SET
            extracted_text = %s,
            total_pages = %s,
            total_characters = %s,
            total_chunks = %s,
            metadata = %s,
            pinecone_namespace = %s,
            pinecone_doc_ids = %s,
            pinecone_upserted_count = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        RETURNING id, document_uid, filename, mime_type, file_size, total_pages, total_chunks,
                  pinecone_namespace, pinecone_doc_ids, pinecone_upserted_count, upload_date, updated_at;
        """

        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    update_sql,
                    (
                        extracted_text,
                        total_pages,
                        total_characters,
                        total_chunks,
                        json.dumps(metadata or {}),
                        pinecone_namespace,
                        pinecone_doc_ids or [],
                        pinecone_upserted_count,
                        document_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()

            if not row:
                raise DatabaseOperationError(f"Document with id {document_id} was not found.")

            res = dict(row)
            for k in ["upload_date", "updated_at"]:
                if isinstance(res.get(k), datetime):
                    res[k] = res[k].isoformat()

            logger.info(f"Updated document id={document_id} with extraction & Pinecone metadata (vectors: {len(pinecone_doc_ids or [])}).")
            return res

        except DatabaseError:
            if conn:
                conn.rollback()
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error updating document id={document_id}: {str(e)}")
            raise DatabaseOperationError(f"Failed to update document extraction: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    def get_document(self, document_id: int, include_data: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieves document metadata and optionally raw binary bytes by document ID.
        """
        select_sql = """
        SELECT id, document_uid, filename, mime_type, file_size, upload_date,
               extracted_text, total_pages, total_characters, total_chunks, metadata,
               pinecone_namespace, pinecone_doc_ids, pinecone_upserted_count,
               created_at, updated_at
               {data_clause}
        FROM documents
        WHERE id = %s;
        """.format(data_clause=", file_data" if include_data else "")

        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (document_id,))
                row = cur.fetchone()

            if not row:
                return None

            res = dict(row)
            for k in ["upload_date", "created_at", "updated_at"]:
                if isinstance(res.get(k), datetime):
                    res[k] = res[k].isoformat()

            if include_data and "file_data" in res and isinstance(res["file_data"], memoryview):
                res["file_data"] = res["file_data"].tobytes()

            return res

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error fetching document id={document_id}: {str(e)}")
            raise DatabaseOperationError(f"Failed to fetch document: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    def get_document_by_uid(self, document_uid: str, include_data: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieves document metadata and optionally raw binary bytes by unique document UID.
        """
        select_sql = """
        SELECT id, document_uid, filename, mime_type, file_size, upload_date,
               extracted_text, total_pages, total_characters, total_chunks, metadata,
               pinecone_namespace, pinecone_doc_ids, pinecone_upserted_count,
               created_at, updated_at
               {data_clause}
        FROM documents
        WHERE document_uid = %s;
        """.format(data_clause=", file_data" if include_data else "")

        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (document_uid,))
                row = cur.fetchone()

            if not row:
                return None

            res = dict(row)
            for k in ["upload_date", "created_at", "updated_at"]:
                if isinstance(res.get(k), datetime):
                    res[k] = res[k].isoformat()

            if include_data and "file_data" in res and isinstance(res["file_data"], memoryview):
                res["file_data"] = res["file_data"].tobytes()

            return res

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error fetching document uid={document_uid}: {str(e)}")
            raise DatabaseOperationError(f"Failed to fetch document: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    def get_documents(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieves a list of uploaded documents ordered by creation date descending.
        Excludes the heavy file_data binary column for efficient listing.
        """
        select_sql = """
        SELECT id, document_uid, filename, mime_type, file_size, upload_date,
               total_pages, total_characters, total_chunks,
               pinecone_namespace, pinecone_upserted_count,
               created_at, updated_at
        FROM documents
        ORDER BY id DESC
        LIMIT %s OFFSET %s;
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (max(1, min(limit, 200)), max(0, offset)))
                rows = cur.fetchall()

            results = []
            for r in rows:
                doc = dict(r)
                for k in ["upload_date", "created_at", "updated_at"]:
                    if isinstance(doc.get(k), datetime):
                        doc[k] = doc[k].isoformat()
                results.append(doc)

            return results

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise DatabaseOperationError(f"Failed to list documents: {str(e)}") from e
        finally:
            if conn:
                conn.close()
