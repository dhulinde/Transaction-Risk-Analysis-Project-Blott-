"""
This file focuses on testing the API endpoints and request/response handling.
It complements test_llm.py (which tests LLM implementations) and test_suite.py (which tests full flows).
"""
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from base64 import b64encode

from app.config import settings
from app.models import RiskAnalysis
from app.main import app

# Create a test client
client = TestClient(app)

# Helper function to create authorization headers
def get_auth_header(username=settings.auth_username, password=settings.auth_password):
    credentials = b64encode(f"{username}:{password}".encode()).decode("ascii")
    return {"Authorization": f"Basic {credentials}"}

# Sample valid transaction data
VALID_TRANSACTION = {
    "transaction_id": "tx_12345abcde",
    "timestamp": "2025-05-07T14:30:45Z",
    "amount": 129.99,
    "currency": "USD",
    "customer": {
        "id": "cust_98765zyxwv",
        "country": "US",
        "ip_address": "192.168.1.1"
    },
    "payment_method": {
        "type": "credit_card",
        "last_four": "4242",
        "country_of_issue": "CA"
    },
    "merchant": {
        "id": "merch_abcde12345",
        "name": "Example Store",
        "category": "electronics"
    }
}

# Sample risk analysis response
SAMPLE_RISK_ANALYSIS = {
    "risk_score": 0.25,
    "risk_factors": ["Cross-border transaction"],
    "reasoning": "Transaction shows minor risk due to payment method country differing from customer country",
    "recommended_action": "allow"
}

class TestAPIEndpoints:
    def test_webhook_transaction_endpoint_exists(self):
        """Test that the webhook endpoint exists (even with auth)"""
        # We expect 401 because we didn't provide auth, but not 404
        response = client.post("/webhook/transaction")
        assert response.status_code == 401, "Webhook endpoint should exist but require authentication"
    
    def test_method_not_allowed(self):
        """Test that only POST is allowed on the webhook endpoint"""
        response = client.get(
            "/webhook/transaction", 
            headers=get_auth_header()
        )
        assert response.status_code == 405, "Only POST should be allowed on webhook endpoint"
    
    def test_content_type_validation(self):
        """Test that the API validates content-type"""
        response = client.post(
            "/webhook/transaction",
            headers={**get_auth_header(), "Content-Type": "text/plain"},
            content="This is not JSON"
        )
        assert response.status_code in [400, 415], "Should reject non-JSON content"

    def test_missing_required_fields(self):
        """Test validation of required fields in the transaction"""
        incomplete_transaction = {
            "transaction_id": "tx_12345abcde",
            # Missing all other required fields
        }
        
        response = client.post(
            "/webhook/transaction",
            headers=get_auth_header(),
            json=incomplete_transaction
        )
        
        assert response.status_code == 400
        # Check that validation errors are in the response
        assert "detail" in response.json()
    
    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        response = client.post(
            "/webhook/transaction",
            headers=get_auth_header(),
            content="{bad json"
        )
        
        assert response.status_code == 400
        assert "Invalid" in response.json().get("detail", ""), "Should indicate invalid format"
    
    def test_llm_error_handling(self):
        """Test handling of LLM analysis errors"""
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.side_effect = Exception("LLM service unavailable")
            
            response = client.post(
                "/webhook/transaction",
                headers=get_auth_header(),
                json=VALID_TRANSACTION
            )
            
            assert response.status_code == 500
            assert "LLM" in response.json().get("detail", ""), "Should indicate LLM error"
    
    def test_response_structure(self):
        """Test the structure of a successful response"""
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = RiskAnalysis(**SAMPLE_RISK_ANALYSIS)
            
            response = client.post(
                "/webhook/transaction",
                headers=get_auth_header(),
                json=VALID_TRANSACTION
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "transaction_id" in data
            assert "risk_score" in data
            assert "recommended_action" in data
            
            # Check values
            assert data["transaction_id"] == VALID_TRANSACTION["transaction_id"]
            assert data["risk_score"] == SAMPLE_RISK_ANALYSIS["risk_score"]
            assert data["recommended_action"] == SAMPLE_RISK_ANALYSIS["recommended_action"]

if __name__ == "__main__":
    pytest.main()
