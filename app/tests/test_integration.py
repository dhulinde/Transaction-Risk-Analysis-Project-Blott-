"""
Integration test for the webhook endpoint with different transaction types.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from base64 import b64encode

from app.models import RiskAnalysis
from app.config import settings
from app.main import app

# Create a test client
client = TestClient(app)

# Helper function to create authorization headers
def get_auth_header(username=settings.auth_username, password=settings.auth_password):
    credentials = b64encode(f"{username}:{password}".encode()).decode("ascii")
    return {"Authorization": f"Basic {credentials}"}

# Test cases
NORMAL_TRANSACTION = {
    "transaction_id": "tx_normal_01",
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
        "country_of_issue": "US"  # Same country as customer
    },
    "merchant": {
        "id": "merch_abcde12345",
        "name": "Example Store",
        "category": "electronics"
    }
}

CROSS_BORDER_TRANSACTION = {
    "transaction_id": "tx_cross_01",
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
        "country_of_issue": "CA"  # Different from customer (US)
    },
    "merchant": {
        "id": "merch_abcde12345",
        "name": "Example Store",
        "category": "electronics"
    }
}

HIGH_VALUE_TRANSACTION = {
    "transaction_id": "tx_highval_01",
    "timestamp": "2025-05-07T14:30:45Z",
    "amount": 9999.99,  # High value
    "currency": "USD",
    "customer": {
        "id": "cust_98765zyxwv",
        "country": "US",
        "ip_address": "192.168.1.1"
    },
    "payment_method": {
        "type": "credit_card",
        "last_four": "4242",
        "country_of_issue": "US"
    },
    "merchant": {
        "id": "merch_abcde12345",
        "name": "Example Store",
        "category": "electronics"
    }
}

HIGH_RISK_COUNTRY_TRANSACTION = {
    "transaction_id": "tx_riskcountry_01",
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
        "country_of_issue": "RU"  # High risk country
    },
    "merchant": {
        "id": "merch_abcde12345",
        "name": "Example Store",
        "category": "electronics"
    }
}

MISSING_FIELDS_TRANSACTION = {
    "transaction_id": "tx_missing_01",
    "timestamp": "2025-05-07T14:30:45Z",
    "amount": 129.99,
    "currency": "USD",
    # Missing customer data
    "payment_method": {
        "type": "credit_card",
        "last_four": "4242",
        "country_of_issue": "US"
    },
    "merchant": {
        "id": "merch_abcde12345",
        "name": "Example Store",
        "category": "electronics"
    }
}

class TestTransactionIntegration:
    def test_normal_transaction(self):
        """Test with a normal domestic transaction"""
        # Mock the risk analysis result
        risk_analysis = RiskAnalysis(
            risk_score=0.1,
            risk_factors=[],
            reasoning="Normal domestic transaction with matching customer and payment method countries",
            recommended_action="allow"
        )
        
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = risk_analysis
            
            response = client.post(
                "/webhook/transaction",
                headers=get_auth_header(),
                json=NORMAL_TRANSACTION
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["risk_score"] == 0.1
            assert data["recommended_action"] == "allow"
    
    def test_cross_border_transaction(self):
        """Test with a cross-border transaction"""
        # Mock the risk analysis result
        risk_analysis = RiskAnalysis(
            risk_score=0.4,
            risk_factors=["Cross-border transaction - customer country (US) differs from payment method country (CA)"],
            reasoning="Transaction shows medium risk due to geographic mismatch",
            recommended_action="review"
        )
        
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = risk_analysis
            
            response = client.post(
                "/webhook/transaction",
                headers=get_auth_header(),
                json=CROSS_BORDER_TRANSACTION
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["risk_score"] == 0.4
            assert data["recommended_action"] == "review"
    
    def test_high_value_transaction(self):
        """Test with a high value transaction"""
        # Mock the risk analysis result
        risk_analysis = RiskAnalysis(
            risk_score=0.6,
            risk_factors=["Unusually large transaction amount"],
            reasoning="Transaction shows elevated risk due to high value",
            recommended_action="review"
        )
        
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = risk_analysis
            
            response = client.post(
                "/webhook/transaction",
                headers=get_auth_header(),
                json=HIGH_VALUE_TRANSACTION
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["risk_score"] == 0.6
            assert data["recommended_action"] == "review"
    
    def test_high_risk_country_transaction(self):
        """Test with a transaction involving a high-risk country"""
        # Mock the risk analysis result
        risk_analysis = RiskAnalysis(
            risk_score=0.8,
            risk_factors=["Payment method from high-risk country (RU)"],
            reasoning="Transaction shows high risk due to payment originating from high-risk country",
            recommended_action="block"
        )
        
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze, \
             patch("app.business_logic.api_notifier.notify_api", new_callable=AsyncMock) as mock_notify:
            mock_analyze.return_value = risk_analysis
            
            response = client.post(
                "/webhook/transaction",
                headers=get_auth_header(),
                json=HIGH_RISK_COUNTRY_TRANSACTION
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["risk_score"] == 0.8
            assert data["recommended_action"] == "block"
            
            # Verify that the notify_api function was called (for high-risk transactions)
            assert mock_notify.called
    
    def test_missing_fields_transaction(self):
        """Test with a transaction missing required fields"""
        response = client.post(
            "/webhook/transaction",
            headers=get_auth_header(),
            json=MISSING_FIELDS_TRANSACTION
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

if __name__ == "__main__":
    pytest.main()