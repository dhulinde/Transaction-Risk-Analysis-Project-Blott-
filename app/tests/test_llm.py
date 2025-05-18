"""This file contains unit tests specifically focused on the LLM implementations"""

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
import re

from app.models import Transaction, RiskAnalysis
from app.llm.openai_llm import OpenAILLM
from app.llm.claude_llm import ClaudeLLM
from app.llm.groq_llm import GroqLLM

# Sample valid transaction data for testing
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

# Sample LLM responses
CLEAN_RESPONSE = {
    "risk_score": 0.25,
    "risk_factors": ["Cross-border transaction"],
    "reasoning": "Transaction shows minor risk due to payment method country differing from customer country",
    "recommended_action": "allow"
}

MARKDOWN_RESPONSE = """```json
{
    "risk_score": 0.25,
    "risk_factors": ["Cross-border transaction"],
    "reasoning": "Transaction shows minor risk due to payment method country differing from customer country",
    "recommended_action": "allow"
}
```"""

THINK_TAG_RESPONSE = """<think>
This transaction has a customer in the US with a payment method from Canada, which is a minor geographic mismatch.
The amount is relatively small and the merchant category is common.
I'll assign a low risk score but note the cross-border aspect.
</think>

{
    "risk_score": 0.25,
    "risk_factors": ["Cross-border transaction"],
    "reasoning": "Transaction shows minor risk due to payment method country differing from customer country",
    "recommended_action": "allow"
}"""

MALFORMED_RESPONSE = """{
    "risk_score": "medium", 
    "risk_factors": ["Cross-border transaction"],
    "reasoning": "Transaction shows minor risk due to payment method country differing from customer country"
    "recommended_action": "allow",
}"""


class TestOpenAILLM:
    @pytest.mark.asyncio
    async def test_analyze_transaction_success(self):
        """Test OpenAI LLM with a successful API response"""
        llm = OpenAILLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(CLEAN_RESPONSE)
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
            assert len(result.risk_factors) == 1
            assert result.risk_factors[0] == "Cross-border transaction"
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test OpenAI LLM retries on rate limit errors"""
        llm = OpenAILLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        # First response is rate limit error, second is success
        responses = [
            MagicMock(
                status_code=429,
                json=lambda: {"error": {"message": "Rate limit exceeded"}}
            ),
            MagicMock(
                status_code=200, 
                json=lambda: {
                    "choices": [{"message": {"content": json.dumps(CLEAN_RESPONSE)}}],
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
    
    @pytest.mark.asyncio 
    async def test_quota_exceeded_handling(self):
        """Test OpenAI LLM handles quota exceeded errors appropriately"""
        llm = OpenAILLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock(
            status_code=429,
            json=lambda: {"error": {"message": "You exceeded your current quota"}}
        )
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
             pytest.raises(Exception) as excinfo:
            
            mock_post.return_value = mock_response
            await llm.analyze_transaction(transaction)
        
        assert "Insufficient quota" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Test OpenAI LLM handles malformed JSON responses"""
        llm = OpenAILLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": MALFORMED_RESPONSE
                    }
                }
            ]
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
             pytest.raises(Exception) as excinfo:
            
            mock_post.return_value = mock_response
            await llm.analyze_transaction(transaction)
        
        assert "Error decoding JSON" in str(excinfo.value) or "JSONDecodeError" in str(excinfo.value)


class TestClaudeLLM:
    @pytest.mark.asyncio
    async def test_analyze_transaction_success(self):
        """Test Claude LLM with a successful API response"""
        llm = ClaudeLLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": json.dumps(CLEAN_RESPONSE)}]
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await llm.analyze_transaction(transaction)
            
            assert mock_post.called
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
            assert len(result.risk_factors) == 1
    
    @pytest.mark.asyncio
    async def test_claude_api_error_handling(self):
        """Test Claude LLM handles API errors correctly"""
        llm = ClaudeLLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        # Create a response that will raise HTTPStatusError when raise_for_status is called
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Client Error", 
            request=MagicMock(), 
            response=MagicMock(
                status_code=400, 
                json=lambda: {"error": {"type": "invalid_request_error"}}
            )
        )
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
             patch("builtins.print") as mock_print, \
             pytest.raises(httpx.HTTPStatusError):
            
            mock_post.return_value = mock_response
            await llm.analyze_transaction(transaction)
            
            assert mock_print.called
            assert "API Error Details" in mock_print.call_args[0][0]


class TestGroqLLM:
    @pytest.mark.asyncio
    async def test_analyze_transaction_success(self):
        """Test Groq LLM with a successful API response"""
        llm = GroqLLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(CLEAN_RESPONSE)
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
            assert len(result.risk_factors) == 1
    
    @pytest.mark.asyncio
    async def test_markdown_cleanup(self):
        """Test Groq LLM cleans up markdown formatting in responses"""
        llm = GroqLLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": MARKDOWN_RESPONSE
                    }
                }
            ],
            "usage": {"total_tokens": 180}
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await llm.analyze_transaction(transaction)
            
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
    
    @pytest.mark.asyncio
    async def test_think_tag_removal(self):
        """Test Groq LLM removes <think> tags in responses"""
        llm = GroqLLM()
        transaction = Transaction(**VALID_TRANSACTION)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": THINK_TAG_RESPONSE
                    }
                }
            ],
            "usage": {"total_tokens": 250}
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await llm.analyze_transaction(transaction)
            
            assert result.risk_score == 0.25
            assert result.recommended_action == "allow"
    
    @pytest.mark.asyncio
    async def test_extract_json_method(self):
        """Test the _extract_json method directly"""
        llm = GroqLLM()
        
        # Test with clean JSON
        clean_json = json.dumps(CLEAN_RESPONSE)
        result = llm._extract_json(clean_json)
        assert result["risk_score"] == 0.25
        
        # Test with markdown formatting
        result = llm._extract_json(MARKDOWN_RESPONSE)
        assert result["risk_score"] == 0.25
        
        # Test with think tags
        result = llm._extract_json(THINK_TAG_RESPONSE)
        assert result["risk_score"] == 0.25
        
        # Test with JSON embedded in text
        text_with_json = "Here's my analysis:\n\n" + json.dumps(CLEAN_RESPONSE) + "\n\nHope this helps!"
        result = llm._extract_json(text_with_json)
        assert result["risk_score"] == 0.25
        
        # Test with malformed JSON
        with pytest.raises(json.JSONDecodeError):
            llm._extract_json(MALFORMED_RESPONSE)


if __name__ == "__main__":
    pytest.main()