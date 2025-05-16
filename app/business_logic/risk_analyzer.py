from app.llm.base import LLM
from app.models import Transaction, RiskAnalysis
from app.llm.openai_llm import OpenAILLM
from app.llm.claude_llm import ClaudeLLM
from app.llm.groq_llm import GroqLLM 
from app.config import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create instances of LLM providers
llm_provider = {
    "openai": OpenAILLM(),
    "claude": ClaudeLLM(),
    "groq": GroqLLM(),  # Uncomment when GroqLLM is implemented
    # Add other LLM providers here once implemented
}

async def analyze_transaction(transaction: Transaction, llm_name: str) -> RiskAnalysis:
    llm_name = llm_name.lower()
    
    logger.info(f"Analyzing transaction {transaction.transaction_id} using {llm_name}")
    
    if llm_name not in llm_provider:
        logger.error(f"LLM provider '{llm_name}' is not supported.")
        raise ValueError(f"LLM provider '{llm_name}' is not supported.")
    
    try:
        llm = llm_provider[llm_name]
        logger.info(f"Starting analysis with {llm_name}")
        risk_analysis = await llm.analyze_transaction(transaction)
        logger.info(f"Analysis complete with risk score: {risk_analysis.risk_score}")
        return risk_analysis
    except Exception as e:
        logger.error(f"Error during transaction analysis: {str(e)}")
        raise