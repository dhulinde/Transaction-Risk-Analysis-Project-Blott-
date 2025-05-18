import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from base64 import b64encode
import httpx
import asyncio 

from app.models import Transaction, RiskAnalysis, Customer, PaymentMethod, Merchant
from app.config import settings
from app.llm.openai_llm import OpenAILLM
from app.llm.claude_llm import ClaudeLLM
from app.llm.groq_llm import GroqLLM
from app.business_logic.risk_analyzer import analyze_transaction
from app.business_logic.api_notifier import notify_api
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

# Sample high risk transaction data (cross-border with high amount)
HIGH_RISK_TRANSACTION = {
    "transaction_id": "tx_67890fghij",
    "timestamp": "2025-05-07T02:12:33Z",
    "amount": 4999.99,
    "currency": "USD",
    "customer": {
        "id": "cust_12345abcde",
        "country": "US",
        "ip_address": "203.0.113.195"  # IP from different country
    },
    "payment_method": {
        "type": "credit_card",
        "last_four": "9876",
        "country_of_issue": "RU"  # High-risk country
    },
    "merchant": {
        "id": "merch_67890fghij",
        "name": "Luxury Goods",
        "category": "jewelry"
    }
}

# Sample risk analysis response
SAMPLE_RISK_ANALYSIS = {
    "risk_score": 0.25,
    "risk_factors": ["Cross-border transaction"],
    "reasoning": "Transaction shows minor risk due to payment method country differing from customer country",
    "recommended_action": "allow"
}

HIGH_RISK_ANALYSIS = {
    "risk_score": 0.85,
    "risk_factors": [
        "Customer country (US) differs from card country (RU)",
        "High-risk country involved (RU)",
        "Unusually large transaction amount",
        "Transaction during unusual hours"
    ],
    "reasoning": "Multiple high-risk indicators present including payment from high-risk jurisdiction",
    "recommended_action": "block"
}


class TestAuthentication:
    def test_valid_credentials(self):
        """Test that valid credentials are accepted"""
        response = client.post(
            "/webhook/transaction", 
            headers=get_auth_header(), 
            json=VALID_TRANSACTION
        )
        assert response.status_code != 401
    
    def test_invalid_credentials(self):
        """Test that invalid credentials are rejected"""
        response = client.post(
            "/webhook/transaction", 
            headers=get_auth_header("wrong", "credentials"), 
            json=VALID_TRANSACTION
        )
        assert response.status_code == 401


class TestTransactionValidation:
    def test_valid_transaction(self):
        """Test that a valid transaction is accepted"""
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = RiskAnalysis(**SAMPLE_RISK_ANALYSIS)
            
            response = client.post(
                "/webhook/transaction", 
                headers=get_auth_header(), 
                json=VALID_TRANSACTION
            )
            
            assert response.status_code == 200
            assert "transaction_id" in response.json()
            assert "risk_score" in response.json()
            assert "recommended_action" in response.json()
    
    def test_invalid_transaction_format(self):
        """Test that invalid transaction format is rejected"""
        invalid_data = {
            "transaction_id": "tx_12345abcde",
            # Missing required fields
        }
        
        response = client.post(
            "/webhook/transaction", 
            headers=get_auth_header(), 
            json=invalid_data
        )
        
        assert response.status_code == 400


class TestRiskAnalyzer:
    @pytest.mark.asyncio
    async def test_openai_analysis(self):
        """Test OpenAI LLM risk analysis"""
        transaction = Transaction(**VALID_TRANSACTION)
        
        with patch.object(OpenAILLM, "analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = RiskAnalysis(**SAMPLE_RISK_ANALYSIS)
            
            result = await analyze_transaction(transaction, "openai")
            
            assert mock_analyze.called
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
    
    @pytest.mark.asyncio
    async def test_claude_analysis(self):
        """Test Claude LLM risk analysis"""
        transaction = Transaction(**VALID_TRANSACTION)
        
        with patch.object(ClaudeLLM, "analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = RiskAnalysis(**SAMPLE_RISK_ANALYSIS)
            
            result = await analyze_transaction(transaction, "claude")
            
            assert mock_analyze.called
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
    
    @pytest.mark.asyncio
    async def test_groq_analysis(self):
        """Test Groq LLM risk analysis"""
        transaction = Transaction(**VALID_TRANSACTION)
        
        with patch.object(GroqLLM, "analyze_transaction", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = RiskAnalysis(**SAMPLE_RISK_ANALYSIS)
            
            result = await analyze_transaction(transaction, "groq")
            
            assert mock_analyze.called
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
    
    @pytest.mark.asyncio
    async def test_invalid_llm_provider(self):
        """Test handling of invalid LLM provider"""
        transaction = Transaction(**VALID_TRANSACTION)
        
        with pytest.raises(ValueError) as excinfo:
            await analyze_transaction(transaction, "nonexistent_llm")
        
        assert "not supported" in str(excinfo.value)


class TestLLMImplementations:
    @pytest.mark.asyncio
    async def test_openai_llm_integration(self):
        """Test OpenAI LLM integration with mocked API response"""
        llm = OpenAILLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(SAMPLE_RISK_ANALYSIS)
                    }
                }
            ],
            "usage": {"total_tokens": 150}
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await llm.analyze_transaction(transaction)
            
            assert mock_post.called
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
    
    @pytest.mark.asyncio
    async def test_claude_llm_integration(self):
        """Test Claude LLM integration with mocked API response"""
        llm = ClaudeLLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": json.dumps(SAMPLE_RISK_ANALYSIS)}]
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await llm.analyze_transaction(transaction)
            
            assert mock_post.called
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
    
    @pytest.mark.asyncio
    async def test_groq_llm_integration(self):
        """Test Groq LLM integration with mocked API response"""
        llm = GroqLLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(SAMPLE_RISK_ANALYSIS)
                    }
                }
            ],
            "usage": {"total_tokens": 150}
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await llm.analyze_transaction(transaction)
            
            assert mock_post.called
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
    
    @pytest.mark.asyncio
    async def test_openai_error_handling(self):
        """Test OpenAI error handling for rate limits"""
        llm = OpenAILLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        # First response - rate limit, second - success
        responses = [
            MagicMock(
                status_code=429,
                json=lambda: {"error": {"message": "Rate limit exceeded"}}
            ),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"content": json.dumps(SAMPLE_RISK_ANALYSIS)}}],
                    "usage": {"total_tokens": 150}
                }
            )
        ]
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            
            mock_post.side_effect = responses
            
            result = await llm.analyze_transaction(transaction)
            
            assert mock_post.call_count == 2
            assert mock_sleep.called
            assert result.risk_score == 0.25


class TestAPINotifier:
    @pytest.mark.asyncio
    async def test_notify_api_high_risk(self):
        """Test admin notification for high-risk transactions"""
        transaction = Transaction(**HIGH_RISK_TRANSACTION)
        risk_analysis = RiskAnalysis(**HIGH_RISK_ANALYSIS)
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            await notify_api(transaction, risk_analysis)
            
            # Check that notification was sent
            assert mock_post.called
            
            # Verify payload structure
            payload = mock_post.call_args[1]["json"]
            assert payload["alert_type"] == "high_risk_transaction"
            assert payload["transaction_id"] == transaction.transaction_id
            assert payload["risk_score"] == risk_analysis.risk_score
            assert payload["risk_factors"] == risk_analysis.risk_factors
    
    @pytest.mark.asyncio
    async def test_notify_api_error_handling(self):
        """Test error handling in admin notification"""
        transaction = Transaction(**HIGH_RISK_TRANSACTION)
        risk_analysis = RiskAnalysis(**HIGH_RISK_ANALYSIS)
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
             patch("builtins.print") as mock_print:
            
            # Simulate HTTP error
            mock_post.side_effect = httpx.HTTPStatusError(
                "Error", 
                request=MagicMock(), 
                response=MagicMock(status_code=500, text="Server error")
            )
            
            # Should not raise exception
            await notify_api(transaction, risk_analysis)
            
            # Verify error was logged
            assert mock_print.called
            assert "HTTP error occurred" in mock_print.call_args[0][0]


class TestEndToEndFlow:
    def test_normal_transaction_flow(self):
        """Test end-to-end flow with normal risk transaction"""
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze, \
             patch("app.business_logic.api_notifier.notify_api", new_callable=AsyncMock) as mock_notify:
            
            # Set up mock returns
            mock_analyze.return_value = RiskAnalysis(**SAMPLE_RISK_ANALYSIS)
            
            # Send request
            response = client.post(
                "/webhook/transaction", 
                headers=get_auth_header(), 
                json=VALID_TRANSACTION
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["transaction_id"] == VALID_TRANSACTION["transaction_id"]
            assert result["risk_score"] == SAMPLE_RISK_ANALYSIS["risk_score"]
            assert result["recommended_action"] == SAMPLE_RISK_ANALYSIS["recommended_action"]
            
            # Verify notification was NOT sent (risk score < 0.7)
            assert not mock_notify.called
    
    def test_high_risk_transaction_flow(self):
        """Test end-to-end flow with high risk transaction"""
        with patch("app.business_logic.risk_analyzer.analyze_transaction", new_callable=AsyncMock) as mock_analyze, \
             patch("app.business_logic.api_notifier.notify_api", new_callable=AsyncMock) as mock_notify:
            
            # Set up mock returns
            mock_analyze.return_value = RiskAnalysis(**HIGH_RISK_ANALYSIS)
            
            # Send request
            response = client.post(
                "/webhook/transaction", 
                headers=get_auth_header(), 
                json=HIGH_RISK_TRANSACTION
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["transaction_id"] == HIGH_RISK_TRANSACTION["transaction_id"]
            assert result["risk_score"] == HIGH_RISK_ANALYSIS["risk_score"]
            assert result["recommended_action"] == HIGH_RISK_ANALYSIS["recommended_action"]
            
            # Verify notification was sent (risk score >= 0.7)
            assert mock_notify.called


if __name__ == "__main__":
    pytest.main()