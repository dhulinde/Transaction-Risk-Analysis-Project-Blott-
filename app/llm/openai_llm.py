import httpx
import json
import time
import asyncio
from app.config import settings
from app.llm.base import LLM
from app.models import Transaction, RiskAnalysis

class OpenAILLM(LLM):
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 1000  # Default max tokens (For testing purposes)
    

    async def analyze_transaction(self, transaction: Transaction) -> RiskAnalysis:
        prompt = self._build_prompt(transaction)
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": self.max_tokens
        }

        start_time = time.time()

        for attempt in range(3):
            async with httpx.AsyncClient() as client:
                response = await client.post(settings.openai_api_url, headers=headers, json=data)

            if response.status_code == 429:
                try:
                    error_data = response.json()
                    message = error_data.get("error", {}).get("message", "")
                    if "quota" in message.lower() or "insufficient" in message.lower():
                        raise Exception(f"OpenAI Error: Insufficient quota or credits. Message: {message}")
                    else:
                        wait = 2 ** attempt
                        print(f"[Retry {attempt+1}] Rate limit hit. Waiting {wait}s... Message: {message}")
                        await asyncio.sleep(wait)
                        continue
                except Exception as e:
                    raise Exception(f"OpenAI 429 Error: {e}")
            else:
                response.raise_for_status()
                break
        else:
            raise Exception("OpenAI: Too many requests after retries.")

        duration = time.time() - start_time
        response_data = response.json()

        content = response_data['choices'][0]['message']['content']
        usage = response_data.get('usage', {})
        print(f"Response time: {duration:.2f}s | Tokens: {usage.get('total_tokens')}")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"LLM Output: {content}")
            raise

        return RiskAnalysis(**result)

    def _build_prompt(self, transaction: Transaction) -> str:
        transaction_json = transaction.model_dump_json(indent=2)
        return f"""
# Transaction Risk Analysis Prompt

## System Instructions
You are a specialised financial risk analyst. Your task is to evaluate transaction data and determine a risk score from 0.0 (no risk) to 1.0 (extremely high risk) based on patterns and indicators of potential fraud. You must also provide clear reasoning for your risk assessment.

## Response Format
Respond ONLY with a valid JSON object in the following structure. Do NOT include markdown, backticks, or explanations — just the pure JSON. You must include all fields, even if they are empty or null.
You MUST include ALL of the following keys: 
- "risk_score"
- "risk_factors"
- "reasoning" (This should be a brief explanation of your analysis) AND IT MUST BE PRESENT
- "recommended_action"

## Risk Factors to Consider

1. **Geographic Anomalies**
   - Customer country differs from payment method country
   - Transactions from high-risk countries (weak AML controls)
   - IP address location inconsistent with customer country

2. **Transaction Patterns**
   - Unusual amount for the merchant category
   - Transactions outside business hours
   - Rapid repeat transactions

3. **Payment Method Indicators**
   - Risky payment method types
   - Recently added payment methods

4. **Merchant Factors**
   - Merchant category with high fraud rates
   - Poor merchant history or reputation

## Additional Guidelines
- Combine multiple factors to increase risk
- Higher amounts = higher scrutiny
- Account for common cross-border activity
- Provide actionable reasoning behind the score
- Use thresholds: allow (0.0–0.3), review (0.3–0.7), block (0.7–1.0)

## Transaction Data
{transaction_json}

"""
