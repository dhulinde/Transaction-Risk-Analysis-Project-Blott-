#add imports for FastAPI
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, ValidationError
from app.config import settings
from app.models import Transaction, RiskAnalysis
from app.business_logic.risk_analyzer import analyze_transaction

#Addded logging for console outputs and testing 
#import logging #(havent implemented the logging yet)

app = FastAPI()
security = HTTPBasic()

@app.post("/webhook/transaction")
async def transaction_webhook(
    request: Request
):
    
    try:
        data = await request.json()
        transaction = Transaction(**data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        print(f"Error parsing request: {e}")
        raise HTTPException(status_code=400, detail="Invalid request format")

    # Analyze risk using selected LLM
    try:
        analysis: RiskAnalysis = await analyze_transaction(transaction, settings.llm_provider)
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        raise HTTPException(status_code=500, detail="LLM analysis failed")

    return {
        "transaction_id": transaction.transaction_id,
        "risk_score": analysis.risk_score,
        "recommended_action": analysis.recommended_action
    }
