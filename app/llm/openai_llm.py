import httpx
import json
import time
import asyncio
from app.config import settings
from app.llm.base import LLM
from app.models import Transaction, RiskAnalysis

class OpenAILLM(LLM):
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 2000  # Default max tokens (For testing purposes)
    

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

        #Error handling for 429 errors (This displays the error message and retries, allows to identify the direct issue)
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
            raise json.JSONDecodeError(f"Failed to decode JSON from LLM response: {content}") from e

        return RiskAnalysis(**result)

    def _build_prompt(self, transaction: Transaction) -> str:
        transaction_json = transaction.model_dump_json(indent=2)
        return f"""
You are a financial-fraud analyst.

Return **only** this JSON, no markdown, no extra keys:

{{
  "risk_score": 0.0-1.0,          // float
  "risk_factors": ["…"],          // list of short strings
  "reasoning": "…",               // ≤ 40 words
  "recommended_action": "allow" | "review" | "block"
}}

Assess risk using:
• Geographic mismatch (customer ↔ card ↔ IP, high-risk country list)
• Pattern anomalies (amount, time-of-day, velocity)
• Payment-method risk
• Merchant reputation / category

Guidelines  
HIGH_RISK_COUNTRIES = ['RU', 'IR', 'KP', 'VE', 'MM']
Assign higher risk scores to combinations of multiple risk factors 
Consider the transaction amount, higher amounts generally warrant more 
scrutiny 
Account for normal cross-border shopping patterns while flagging unusual 
combinations 
Provide actionable reasoning that explains why the transaction received 
its risk score
0.0-0.3 → allow 0.3-0.7 → review 0.7-1.0 → block

Transaction:
{transaction_json}

"""
