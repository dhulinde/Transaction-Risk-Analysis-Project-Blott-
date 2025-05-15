from abc import ABC, abstractmethod
from app.models import Transaction, RiskAnalysis

#implementation of the LLM base class so that all LLMs can be used interchangeably
class LLM(ABC):
    """
    Abstract base class for all LLMs.
    """
    @abstractmethod
    async def analyze_transaction(self, transaction: Transaction) -> RiskAnalysis:
        """
        Analyze a transaction and return a risk analysis.
        """
        pass