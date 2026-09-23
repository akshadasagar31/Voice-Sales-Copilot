import pytest
from unittest.mock import MagicMock, patch
import psycopg2
from pydantic import ValidationError
from fastapi.testclient import TestClient

from main import app
from services.lead_extractor import Lead
from services.lead_repository import (
    LeadRepository,
    DatabaseError,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseOperationError,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Database Repository Unit & Integration Tests
# ---------------------------------------------------------------------------

def test_successful_lead_insert():
    """Verify inserting a complete validated Lead into PostgreSQL returns saved record with id and timestamps."""
    repo = LeadRepository()
    repo.ensure_leads_table()

    lead = Lead(
        name="Vikram Sethi",
        phone="+91-9811223344",
        email="vikram.sethi@hdfcbank.com",
        company="HDFC Bank",
        role="Assistant Vice President",
        loan_type="Personal Loan",
        loan_amount=750000.0,
        tenure_months=48,
        notes="Pre-approved eligible candidate seeking low interest rate.",
    )

    saved = repo.create_lead(lead)
    assert saved["id"] is not None
    assert saved["id"] > 0
    assert saved["name"] == "Vikram Sethi"
    assert saved["phone"] == "+91-9811223344"
    assert saved["email"] == "vikram.sethi@hdfcbank.com"
    assert saved["company"] == "HDFC Bank"
    assert saved["role"] == "Assistant Vice President"
    assert saved["loan_type"] == "Personal Loan"
    assert saved["loan_amount"] == 750000.0
    assert saved["tenure_months"] == 48
    assert saved["notes"] == "Pre-approved eligible candidate seeking low interest rate."
    assert saved["created_at"] is not None
    assert saved["updated_at"] is not None


def test_insert_with_missing_optional_fields():
    """Verify inserting a Lead with only partial data leaves unmentioned fields strictly None (null)."""
    repo = LeadRepository()
    repo.ensure_leads_table()

    partial_lead = Lead(
        name="Sunita Rao",
        loan_type="Home Loan",
    )

    saved = repo.create_lead(partial_lead)
    assert saved["id"] is not None
    assert saved["name"] == "Sunita Rao"
    assert saved["loan_type"] == "Home Loan"
    assert saved["phone"] is None
    assert saved["email"] is None
    assert saved["company"] is None
    assert saved["role"] is None
    assert saved["loan_amount"] is None
    assert saved["tenure_months"] is None
    assert saved["notes"] is None


def test_get_leads_and_get_by_id():
    """Verify querying leads returns list and get_lead_by_id retrieves the target record."""
    repo = LeadRepository()
    repo.ensure_leads_table()

    lead = Lead(name="Query Test Prospect", loan_amount=100000.0)
    saved = repo.create_lead(lead)
    lead_id = saved["id"]

    # Test get_lead_by_id
    retrieved = repo.get_lead_by_id(lead_id)
    assert retrieved is not None
    assert retrieved["id"] == lead_id
    assert retrieved["name"] == "Query Test Prospect"
    assert retrieved["loan_amount"] == 100000.0

    # Test get_leads list
    all_leads = repo.get_leads(limit=10)
    assert len(all_leads) > 0
    assert any(l["id"] == lead_id for l in all_leads)


def test_database_connection_error_raises_exception():
    """Verify connection failure triggers DatabaseConnectionError."""
    bad_repo = LeadRepository(host="invalid.nonexistent.host.local", port=5999)
    with pytest.raises(DatabaseConnectionError):
        bad_repo.get_connection()


def test_database_query_error_handling():
    """Verify operational query errors raise DatabaseOperationError."""
    repo = LeadRepository()
    with patch.object(repo, "get_connection") as mock_conn_func:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.execute.side_effect = psycopg2.ProgrammingError("Simulated SQL syntax failure")
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn_func.return_value = mock_conn

        lead = Lead(name="Error Lead")
        with pytest.raises(DatabaseOperationError) as exc_info:
            repo.create_lead(lead)

        assert "Database insert error" in str(exc_info.value)
        mock_conn.rollback.assert_called()


# ---------------------------------------------------------------------------
# 2. FastAPI Endpoints Integration Tests (/api/leads)
# ---------------------------------------------------------------------------

def test_api_post_lead_endpoint_success():
    """Verify POST /api/leads returns HTTP 201 with saved lead JSON."""
    payload = {
        "name": "Kavita Nair",
        "phone": "9988776655",
        "email": "kavita.nair@wipro.com",
        "company": "Wipro",
        "role": "Lead Architect",
        "loan_type": "Education Loan",
        "loan_amount": 1500000.0,
        "tenure_months": 60,
        "notes": "Abroad MS degree funding",
    }

    resp = client.post("/api/leads", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "success"
    assert data["lead"]["name"] == "Kavita Nair"
    assert data["lead"]["loan_amount"] == 1500000.0
    assert data["lead"]["id"] is not None


def test_api_post_lead_endpoint_missing_optional_fields():
    """Verify POST /api/leads with partial payload saves successfully with nulls."""
    payload = {
        "name": "Anil Ambani",
        "loan_type": "Business Loan",
    }

    resp = client.post("/api/leads", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["lead"]["name"] == "Anil Ambani"
    assert data["lead"]["phone"] is None
    assert data["lead"]["loan_amount"] is None


def test_api_post_lead_endpoint_validation_error():
    """Verify POST /api/leads returns HTTP 422 for invalid types."""
    payload = {
        "name": "Invalid Lead",
        "loan_amount": "invalid_letters_not_float",
    }

    resp = client.post("/api/leads", json=payload)
    assert resp.status_code == 422


def test_api_post_lead_endpoint_db_connection_error():
    """Verify POST /api/leads returns HTTP 503 when database is unavailable."""
    with patch("main.LeadRepository") as MockRepo:
        mock_instance = MagicMock()
        mock_instance.create_lead.side_effect = DatabaseConnectionError("PostgreSQL host unreachable")
        MockRepo.return_value = mock_instance

        resp = client.post(
            "/api/leads",
            json={"name": "Fallback Prospect", "loan_type": "Personal"},
        )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()


def test_api_post_lead_endpoint_db_operation_error():
    """Verify POST /api/leads returns HTTP 500 when SQL operation fails."""
    with patch("main.LeadRepository") as MockRepo:
        mock_instance = MagicMock()
        mock_instance.create_lead.side_effect = DatabaseOperationError("Constraint violation")
        MockRepo.return_value = mock_instance

        resp = client.post(
            "/api/leads",
            json={"name": "Failed Prospect"},
        )
        assert resp.status_code == 500
        assert "operation error" in resp.json()["detail"].lower()


def test_api_get_leads_endpoint():
    """Verify GET /api/leads returns HTTP 200 and list of leads."""
    resp = client.get("/api/leads?limit=20")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert isinstance(data["leads"], list)
    assert data["count"] == len(data["leads"])


def test_update_lead_success():
    """Verify LeadRepository.update_lead updates an existing record without creating a duplicate."""
    repo = LeadRepository()
    repo.ensure_leads_table()

    # Create initial lead
    initial_lead = Lead(
        name="Sunita Rao",
        company="TechCorp",
        role="VP Engineering",
    )
    saved = repo.create_lead(initial_lead)
    lead_id = saved["id"]
    assert saved["loan_amount"] is None

    # Update the same lead with loan details
    updated_payload = Lead(
        name="Sunita Rao",
        company="TechCorp",
        role="VP Engineering",
        loan_type="Equipment Loan",
        loan_amount=2500000.0,
        tenure_months=36,
    )
    updated = repo.update_lead(lead_id, updated_payload)
    assert updated is not None
    assert updated["id"] == lead_id
    assert updated["name"] == "Sunita Rao"
    assert updated["company"] == "TechCorp"
    assert updated["loan_type"] == "Equipment Loan"
    assert updated["loan_amount"] == 2500000.0
    assert updated["tenure_months"] == 36


def test_api_put_lead_endpoint_success():
    """Verify PUT /api/leads/{id} updates existing lead and returns HTTP 200."""
    repo = LeadRepository()
    repo.ensure_leads_table()

    # Create lead
    created = repo.create_lead(Lead(name="Rohan Mehra", company="Infosys"))
    lead_id = created["id"]

    # Update via PUT API
    update_res = client.put(
        f"/api/leads/{lead_id}",
        json={
            "name": "Rohan Mehra",
            "company": "Infosys",
            "loan_type": "Commercial Loan",
            "loan_amount": 10000000.0,
            "tenure_months": 48,
        },
    )
    assert update_res.status_code == 200
    data = update_res.json()
    assert data["status"] == "success"
    assert data["lead"]["id"] == lead_id
    assert data["lead"]["loan_amount"] == 10000000.0
    assert data["lead"]["loan_type"] == "Commercial Loan"


def test_api_put_lead_endpoint_not_found():
    """Verify PUT /api/leads/{id} returns HTTP 404 when lead does not exist."""
    update_res = client.put(
        "/api/leads/99999999",
        json={"name": "Ghost Lead"},
    )
    assert update_res.status_code == 404
    assert "not found" in update_res.json()["detail"].lower()

