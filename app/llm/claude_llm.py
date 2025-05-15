import httpx
import json #to read json file
import time
from app.config import settings
from app.llm.base import LLM
from app.models import Transaction, RiskAnalysis

class ClaudeLLM(LLM):
    model = "claude-3-opus-20240229"

    async def analyze_transaction(self, transaction: Transaction) -> RiskAnalysis:
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json" 
        }
        transaction_json = transaction.model_dump_json(indent=2)
        prompt = self._build_prompt(transaction)
        body = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "user", 
                 "content": prompt
                 }
            ]
        }


        start_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
        
        duration = time.time() - start_time

        response.raise_for_status()
        content = response.json()["content"][0]["text"]
        print(f"Claude Response Time: {duration:.2f}s")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Claude response: {content}") from e

        return RiskAnalysis(**result)

    def _build_prompt(self, transaction: Transaction) -> str:
        return #Add prompt & test it against tuned prompts
        f"""
## System Instructions 
You are a specialised financial risk analyst. Your task is to evaluate 
transaction data and determine a risk score from 0.0 (no risk) to 1.0 
(extremely high risk) based on patterns and indicators of potential fraud. 
You must also provide clear reasoning for your risk assessment. 
## Response Format 
Respond in JSON format with the following structure: 
\`\`\`json  
"risk_score": 0.0-1.0, 
"risk_factors": ["factor1", "factor2"...], 
"reasoning": "A brief explanation of your analysis", 
"recommended_action": "allow|review|block" 
\`\`\` 
## Risk Factors to Consider 
1. **Geographic Anomalies**: - Transactions where the customer country differs from the payment 
method country 
- Transactions from high-risk countries (consider jurisdiction with 
weak AML controls) - IP address location inconsistent with the customer's country 
2. **Transaction Patterns**: - Unusual transaction amount for the merchant category - Transactions outside normal business hours for the merchant's 
location - Multiple transactions in short succession 
3. **Payment Method Indicators**: - Payment method type and associated risks - New payment methods have recently been added to accounts 
4. **Merchant Factors**: - Merchant category and typical fraud rates - Merchant's history and reputation 
## Additional Guidelines - Assign higher risk scores to combinations of multiple risk factors - Consider the transaction amount - higher amounts generally warrant more 
scrutiny - Account for normal cross-border shopping patterns while flagging unusual 
combinations - Provide actionable reasoning that explains why the transaction received 
its risk score - Recommend "allow" for scores 0.0-0.3, "review" for scores 0.3-0.7, and 
"block" for scores 0.7-1.0 
## Transaction Data 
{{TRANSACTION_JSON}} 
`;  
        """