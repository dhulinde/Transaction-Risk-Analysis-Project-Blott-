import httpx
import json
import time
from app.config import settings
from app.llm.base import LLM
from app.models import Transaction, RiskAnalysis
import re

#testing 
import logging
logger = logging.getLogger(__name__)

class GroqLLM(LLM):
    def __init__(self, model_name: str = "deepseek-r1-distill-llama-70b"): #gemma-7b-it, gemma2-9b-it, llama3-70b-8192, deepseek-coder
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

        if content.strip().startswith("```"):
            print("Detected Markdown formatting in LLM response. Stripping...")
            content = re.sub(r"^```[a-z]*\n?", "", content.strip())
            content = re.sub(r"\n?```$", "", content.strip())

        try:
            #result = json.loads(content)
            result = self._extract_json(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}\nResponse: {content}")
            raise

        return RiskAnalysis(**result)

    #ai generated to remove the <think> and <reasoning> tags
    def _extract_json(self, raw: str) -> dict:
        """
        Clean and extract the first valid JSON object from the raw LLM response.
        Handles markdown fences, <think> blocks, and extra text.
        """
        txt = raw.strip()

        # Remove <think>...</think>
        txt = re.sub(r'<think>.*?</think>', '', txt, flags=re.DOTALL | re.IGNORECASE)

        # Remove markdown code fences
        if txt.startswith("```"):
            txt = re.sub(r'^```[a-zA-Z]*\s*', '', txt, count=1)
            txt = re.sub(r'\s*```$', '', txt, count=1)

        # Extract the first valid JSON object
        match = re.search(r'\{.*\}', txt, flags=re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found in response", txt, 0)

        return json.loads(match.group(0))

    def _build_prompt(self, transaction: Transaction) -> str:
        transaction_json = transaction.model_dump_json(indent=2)
        return f"""
You are a financial-fraud analyst.

Return **only** this JSON, no markdown, no extra keys:
Remove all markdown formatting. 
Avoid using the word "JSON" in your response.
Also dont include <think> or <reasoning> in your response.

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

!!Guidelines  
HIGH_RISK_COUNTRIES = ['RU', 'IR', 'KP', 'VE', 'MM']
0.1 to 0.3 → allow 
0.3 to 0.7 → review 
0.7- to 1.0 → block

Transaction:
{transaction_json}

"""
