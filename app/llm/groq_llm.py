import httpx
import json
import time
from app.config import settings
from app.llm.base import LLM
from app.models import Transaction, RiskAnalysis

class GroqLLM(LLM):
    def __init__(self, model_name: str = "gemma2-9b-it"): #gemma-7b-it, llama3-70b-8192, deepseek-coder
        self.model_name = model_name

    async def analyze_transaction(self, transaction: Transaction) -> RiskAnalysis:
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json"
        }

        prompt = self._build_prompt(transaction)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 800
        }

        start_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

        duration = time.time() - start_time
        response.raise_for_status()
        response_data = response.json()

        content = response_data['choices'][0]['message']['content']
        print(f"Groq [{self.model_name}] Response Time: {duration:.2f}s | Tokens: {response_data.get('usage', {}).get('total_tokens')}")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}\nResponse: {content}")
            raise

        return RiskAnalysis(**result)

    def _build_prompt(self, transaction: Transaction) -> str:
        transaction_json = transaction.model_dump_json(indent=2)
        return f"""
# Transaction Risk Analysis Prompt 
## System Instructions 
You are a specialised financial risk analyst. Your task is to evaluate 
transaction data and determine a risk score from 0.0 (no risk) to 1.0 
(extremely high risk) based on patterns and indicators of potential fraud. 
You must also provide clear reasoning for your risk assessment. 

!!! IMPORTANT: Respond in valid JSON format only and only the json format given below.
!!! Do not include any other text or explanations outside of the JSON response. 
!!! Please give the reasoning and follow the format strictly.
!!! Do NOT include Markdown formatting like ```json
## Response Format  
Respond in JSON format with the following structure: 
{{  
        "risk_score": 0.0-1.0, 
        "risk_factors": ["factor1", "factor2"...], 
        "reasoning": "A brief explanation of your analysis", 
        "recommended_action": "allow|review|block"
}}
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
{transaction_json}
"""
