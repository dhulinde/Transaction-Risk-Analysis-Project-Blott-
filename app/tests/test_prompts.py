"""
This module contains unit tests for the prompt templates used in the LLM implementations.
It ensures that prompts are correctly formatted and optimized.
"""
import pytest
from app.models import Transaction, Customer, PaymentMethod, Merchant
from app.llm.openai_llm import OpenAILLM
from app.llm.claude_llm import ClaudeLLM
from app.llm.groq_llm import GroqLLM

# Create a sample transaction for testing
SAMPLE_TRANSACTION = Transaction(
    transaction_id="tx_12345abcde",
    timestamp="2025-05-07T14:30:45Z",
    amount=129.99,
    currency="USD",
    customer=Customer(
        id="cust_98765zyxwv",
        country="US",
        ip_address="192.168.1.1"
    ),
    payment_method=PaymentMethod(
        type="credit_card",
        last_four="4242",
        country_of_issue="CA"
    ),
    merchant=Merchant(
        id="merch_abcde12345",
        name="Example Store",
        category="electronics"
    )
)

class TestPromptTemplates:
    def test_openai_prompt_format(self):
        """Test that the OpenAI prompt is correctly formatted"""
        llm = OpenAILLM()
        prompt = llm._build_prompt(SAMPLE_TRANSACTION)
        
        # Check that the prompt contains all necessary instructions
        assert "risk_score" in prompt
        assert "risk_factors" in prompt
        assert "reasoning" in prompt
        assert "recommended_action" in prompt
        
        # Check that guidelines are included
        assert "0.0-0.3 → allow" in prompt
        assert "0.3-0.7 → review" in prompt
        assert "0.7-1.0 → block" in prompt
        
        # Check that transaction data is included
        assert "tx_12345abcde" in prompt
        assert "cust_98765zyxwv" in prompt
        assert "merch_abcde12345" in prompt
        
        # Check prompt optimization
        assert len(prompt) < 2000, "Prompt should be optimized for token efficiency"
    
    def test_claude_prompt_format(self):
        """Test that the Claude prompt is correctly formatted"""
        llm = ClaudeLLM()
        prompt = llm._build_prompt(SAMPLE_TRANSACTION)
        
        # Check that the prompt contains necessary instructions
        assert "valid JSON" in prompt, "Should instruct to return valid JSON"
        
        # Check that transaction data is included
        assert "tx_12345abcde" in prompt
        assert "cust_98765zyxwv" in prompt
        assert "merch_abcde12345" in prompt
    
    def test_groq_prompt_format(self):
        """Test that the Groq prompt is correctly formatted"""
        llm = GroqLLM()
        prompt = llm._build_prompt(SAMPLE_TRANSACTION)
        
        # Check specific Groq instructions
        assert "Return **only** this JSON" in prompt, "Should instruct to return only JSON"
        assert "no markdown" in prompt, "Should instruct to avoid markdown"
        
        # Check that risk factors to consider are included
        assert "Geographic" in prompt
        assert "Pattern" in prompt
        assert "Payment-method" in prompt
        assert "Merchant" in prompt
        
        # Check that transaction data is included
        assert "tx_12345abcde" in prompt
        assert "cust_98765zyxwv" in prompt
        assert "merch_abcde12345" in prompt
        
        # Check for specific Groq-related instructions
        assert "dont include <think>" in prompt or "no <think>" in prompt
    
    def test_prompt_content_completeness(self):
        """Test that prompts include all necessary instructions for risk analysis"""
        # Test all LLM implementations
        for llm_class in [OpenAILLM, ClaudeLLM, GroqLLM]:
            llm = llm_class()
            prompt = llm._build_prompt(SAMPLE_TRANSACTION)
            
            # Check for essential risk factors
            risk_factors = [
                "country", "geographic", "mismatch",  # Geographic checks
                "amount", "pattern", "transaction",   # Transaction patterns
                "payment", "method",                 # Payment method
                "merchant"                           # Merchant factors
            ]
            
            # At least some of these terms should be present
            found_factors = [factor for factor in risk_factors if factor.lower() in prompt.lower()]
            assert len(found_factors) >= 4, f"{llm_class.__name__} prompt missing key risk factors"
            
            # Check that action guidance is included
            assert "allow" in prompt.lower()
            assert "review" in prompt.lower() or "flag" in prompt.lower()
            assert "block" in prompt.lower()

if __name__ == "__main__":
    pytest.main()