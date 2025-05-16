import httpx
import json
import time
from app.config import settings
from app.llm.base import LLM
from app.models import Transaction, RiskAnalysis

class ClaudeLLM(LLM):
    model = "claude-3-opus-20240229"

    async def analyze_transaction(self, transaction: Transaction) -> RiskAnalysis:
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",  # Consider updating this to "2023-06-01" or latest
            "Content-Type": "application/json" 
        }
        
        transaction_json = transaction.model_dump_json(indent=2)
        prompt = self._build_prompt(transaction)
        
        body = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "system": "You are a specialized financial risk analyst responding with valid JSON only."
        }

        start_time = time.time()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages", 
                    headers=headers, 
                    json=body,
                    timeout=30.0  # Adding explicit timeout
                )
                response.raise_for_status()
                
                duration = time.time() - start_time
                print(f"Claude Response Time: {duration:.2f}s")
                
                content = response.json()["content"][0]["text"]
                
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"Raw response: {content}")
                    raise ValueError(f"Failed to parse Claude response: {content}") from e
                
                return RiskAnalysis(**result)
                
            except httpx.HTTPStatusError as e:
                error_detail = None
                try:
                    error_detail = e.response.json()
                    print(f"API Error Details: {json.dumps(error_detail, indent=2)}")
                except:
                    print(f"Status code: {e.response.status_code}, Response text: {e.response.text}")
                raise e

    def _build_prompt(self, transaction: Transaction) -> str:
        transaction_json = transaction.model_dump_json(indent=2)
        return f"""
Analyze this financial transaction and respond ONLY with a valid JSON object containing:
- risk_score: number between 0.0 and 1.0
- risk_factors: array of strings
- reasoning: brief string explanation
- recommended_action: string ("allow", "review", or "block")

Transaction data:
{transaction_json}

Your response must be valid JSON without any additional text, explanation, or markdown.
"""