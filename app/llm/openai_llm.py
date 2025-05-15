import httpx
import json #to read json file
import time
from app.config import settings
from app.llm.base import LLM
from app.models import Transaction, RiskAnalysis

class OpenAILLM(LLM):
    model_name: str = "gpt-3.5-turbo"  #default model name / will test with other models as well
    max_tokens: int = 1000  # Default max tokens (For testing purposes)

    async def analyze_transaction(self, transaction: Transaction) -> RiskAnalysis:
        prompt = self._build_prompt(transaction)
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.7
        }

        start_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(settings.OPENAI_API_URL, headers=headers, json=data)
            response.raise_for_status()
        
        duration = time.time() - start_time
        response.raise_for_status()
        response_data = response.json()

        content = response_data['choices'][0]['message']['content']
        usage = response_data.get('usage', {})
        print(f"Response time: {duration:.2f} seconds | Status code: {response.status_code} | TOkens: {response_data['usage']['total_tokens']}")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Response content: {content}")
            raise

        risk_analysis = RiskAnalysis(**result)
        return risk_analysis
    
    def _build_prompt(self, transaction: Transaction) -> str:
        return #Add prompt & test it against tuned prompts
