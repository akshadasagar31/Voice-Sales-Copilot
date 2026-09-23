# ============================================================================
# POSTGRESQL LEAD REPOSITORY (backend/services/lead_repository.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Manages database interactions for the sales CRM leads table in PostgreSQL.
#
# KEY RESPONSIBILITIES:
# 1. Database Connections: Connects to PostgreSQL using DATABASE_URL or host/port/user credentials.
# 2. Schema Auto-Initialization: Automatically creates the 'leads' table with proper columns
#    and types if it doesn't exist yet (`ensure_leads_table()`).
# 3. CRUD Operations:
#    - create_lead(): Inserts a validated Pydantic Lead and returns the auto-generated id.
#    - update_lead(): Updates an existing lead in-place during multi-turn calls.
#    - get_lead_by_id() / list_leads(): Fetches leads for display in the frontend CRM drawer.
#    - delete_lead(): Removes obsolete or test records.
# 4. Safe Dictionary Mapping: Uses psycopg2's RealDictCursor so rows map cleanly to JSON.
# ============================================================================

import os
import logging
from typing import Optional, Dict, Any, List
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

from services.lead_extractor import Lead

logger = logging.getLogger(__name__)



class DatabaseError(Exception):
    """Base exception for PostgreSQL database errors."""
    pass


class DatabaseConfigurationError(DatabaseError):
    """Raised when database credentials or connection parameters are missing or invalid."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when unable to establish a connection to PostgreSQL."""
    pass


class DatabaseOperationError(DatabaseError):
    """Raised when an SQL query or transaction execution fails."""
    pass


class LeadRepository:
    """
    PostgreSQL repository for persisting and querying validated sales leads.
    Table: 'leads'
    """
    _pool: Optional[Any] = None

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

    @classmethod
    def get_pool(cls, host=None, port=None, user=None, password=None, dbname=None, database_url=None):
        if cls._pool is None:
            try:
                from psycopg2.pool import ThreadedConnectionPool
                h = host or os.getenv("PGHOST", "127.0.0.1")
                p = port or int(os.getenv("PGPORT", 5432))
                u = user or os.getenv("PGUSER", "postgres")
                pwd = password if password is not None else os.getenv("PGPASSWORD", "")
                db = dbname or os.getenv("PGDATABASE", "voice_sales_copilot")
                url = database_url or os.getenv("DATABASE_URL")
                if h and u and db:
                    cls._pool = ThreadedConnectionPool(1, 10, host=h, port=p, user=u, password=pwd, dbname=db)
                elif url:
                    cls._pool = ThreadedConnectionPool(1, 10, url)
            except Exception as pool_err:
                logger.warning(f"[LeadRepository] Pool initialization note: {pool_err}")
                cls._pool = None
        return cls._pool

    def get_connection(self):
        """
        Establishes and returns a connection to the PostgreSQL database.
        Raises DatabaseConfigurationError or DatabaseConnectionError if connection fails.
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

    def _acquire_conn(self):
        pool = self.get_pool(self.host, self.port, self.user, self.password, self.dbname, self.database_url)
        if pool:
            try:
                return pool.getconn(), True
            except Exception:
                pass
        return self.get_connection(), False

    def _release_conn(self, conn, is_pooled: bool):
        if not conn:
            return
        if is_pooled:
            pool = self.get_pool(self.host, self.port, self.user, self.password, self.dbname, self.database_url)
            if pool:
                try:
                    pool.putconn(conn)
                    return
                except Exception:
                    pass
        try:
            conn.close()
        except Exception:
            pass

    def ensure_leads_table(self) -> None:
        """
        Ensures the 'leads' table exists in the database.
        Creates it automatically with all required fields if missing.
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS leads (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            phone VARCHAR(50),
            email VARCHAR(255),
            company VARCHAR(255),
            role VARCHAR(100),
            loan_type VARCHAR(100),
            loan_amount DOUBLE PRECISION,
            tenure_months INTEGER,
            notes TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
            logger.info("Successfully verified 'leads' table in PostgreSQL.")
        except DatabaseError:
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to ensure 'leads' table exists: {str(e)}")
            raise DatabaseOperationError(f"Failed to create/verify leads table: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------------------
    # METHOD: create_lead
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Executes SQL INSERT into PostgreSQL 'leads' table and returns
    #   the newly generated row with its database ID and created_at timestamp.
    # • INPUTS:
    #     - lead (Lead): Validated Pydantic Lead object with customer fields.
    # • OUTPUT: Dictionary containing the persisted lead row including generated 'id'.
    # • WHY IT IS USED: Commits validated sales contacts to permanent storage.
    # • WHERE IT FITS IN THE FLOW:
    #     [stream_lead_turn in lead_extractor.py] -> [create_lead] -> [PostgreSQL Database]
    # ------------------------------------------------------------------------
    def create_lead(self, lead: Lead) -> Dict[str, Any]:
        """
        Inserts a validated Pydantic Lead into the 'leads' table.

        Returns the saved record dictionary containing the auto-generated id and timestamps.
        """
        if not isinstance(lead, Lead):
            lead = Lead.model_validate(lead)

        insert_sql = """
        INSERT INTO leads (
            name, phone, email, company, role,
            loan_type, loan_amount, tenure_months, notes,
            created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        RETURNING id, name, phone, email, company, role, loan_type, loan_amount, tenure_months, notes, created_at, updated_at;
        """

        conn = None
        is_pooled = False
        try:
            conn, is_pooled = self._acquire_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    insert_sql,
                    (
                        lead.name,
                        lead.phone,
                        lead.email,
                        lead.company,
                        lead.role,
                        lead.loan_type,
                        lead.loan_amount,
                        lead.tenure_months,
                        lead.notes,
                    ),
                )
                row = cur.fetchone()
            conn.commit()

            if not row:
                raise DatabaseOperationError("Insert succeeded but no record was returned.")

            # Convert datetime to ISO string for JSON serialization
            record = dict(row)
            if isinstance(record.get("created_at"), datetime):
                record["created_at"] = record["created_at"].isoformat()
            if isinstance(record.get("updated_at"), datetime):
                record["updated_at"] = record["updated_at"].isoformat()

            return record

        except DatabaseError:
            if conn:
                conn.rollback()
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to insert lead into database: {str(e)}")
            raise DatabaseOperationError(f"Database insert error: {str(e)}") from e
        finally:
            self._release_conn(conn, is_pooled)

    def get_leads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves recent saved leads from PostgreSQL sorted by created_at DESC.
        """
        select_sql = """
        SELECT id, name, phone, email, company, role,
               loan_type, loan_amount, tenure_months, notes,
               created_at, updated_at
        FROM leads
        ORDER BY created_at DESC
        LIMIT %s;
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (limit,))
                rows = cur.fetchall()

            records = []
            for r in rows:
                rec = dict(r)
                if isinstance(rec.get("created_at"), datetime):
                    rec["created_at"] = rec["created_at"].isoformat()
                if isinstance(rec.get("updated_at"), datetime):
                    rec["updated_at"] = rec["updated_at"].isoformat()
                records.append(rec)

            return records

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to query leads: {str(e)}")
            raise DatabaseOperationError(f"Database query error: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    def get_lead_by_id(self, lead_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single lead by its integer ID.
        """
        select_sql = """
        SELECT id, name, phone, email, company, role,
               loan_type, loan_amount, tenure_months, notes,
               created_at, updated_at
        FROM leads
        WHERE id = %s;
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (lead_id,))
                row = cur.fetchone()

            if not row:
                return None

            rec = dict(row)
            if isinstance(rec.get("created_at"), datetime):
                rec["created_at"] = rec["created_at"].isoformat()
            if isinstance(rec.get("updated_at"), datetime):
                rec["updated_at"] = rec["updated_at"].isoformat()
            return rec

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to query lead #{lead_id}: {str(e)}")
            raise DatabaseOperationError(f"Database query error: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------------------
    # METHOD: update_lead
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Updates an existing lead record in PostgreSQL by its primary key ID
    #   and refreshes the 'updated_at' timestamp.
    # • INPUTS:
    #     - lead_id (int): Primary key ID of the existing record.
    #     - lead (Lead): Updated Pydantic Lead object with new/merged fields.
    # • OUTPUT: Updated row dictionary, or None if the record was not found.
    # • WHY IT IS USED: Prevents creating duplicate records when a user provides details
    #   in multiple conversational turns (e.g., provides name in turn 1, loan amount in turn 2).
    # • WHERE IT FITS IN THE FLOW:
    #     [Turn 2: Prospect adds info] -> [stream_lead_turn] -> [update_lead] -> [PostgreSQL]
    # ------------------------------------------------------------------------
    def update_lead(self, lead_id: int, lead: Lead) -> Optional[Dict[str, Any]]:
        """
        Updates an existing lead record in the 'leads' table by its integer ID.

        Updates 'updated_at' timestamp automatically.
        Returns the updated record dictionary, or None if the lead does not exist.
        """
        if not isinstance(lead, Lead):
            lead = Lead.model_validate(lead)

        update_sql = """
        UPDATE leads
        SET name = %s,
            phone = %s,
            email = %s,
            company = %s,
            role = %s,
            loan_type = %s,
            loan_amount = %s,
            tenure_months = %s,
            notes = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        RETURNING id, name, phone, email, company, role, loan_type, loan_amount, tenure_months, notes, created_at, updated_at;
        """

        conn = None
        is_pooled = False
        try:
            conn, is_pooled = self._acquire_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    update_sql,
                    (
                        lead.name,
                        lead.phone,
                        lead.email,
                        lead.company,
                        lead.role,
                        lead.loan_type,
                        lead.loan_amount,
                        lead.tenure_months,
                        lead.notes,
                        lead_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()

            if not row:
                return None

            record = dict(row)
            if isinstance(record.get("created_at"), datetime):
                record["created_at"] = record["created_at"].isoformat()
            if isinstance(record.get("updated_at"), datetime):
                record["updated_at"] = record["updated_at"].isoformat()

            return record

        except DatabaseError:
            if conn:
                conn.rollback()
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to update lead #{lead_id} in database: {str(e)}")
            raise DatabaseOperationError(f"Database update error: {str(e)}") from e
        finally:
            self._release_conn(conn, is_pooled)

