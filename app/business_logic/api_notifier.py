import httpx
from app.models import Transaction, RiskAnalysis
from app.config import settings

async def notify_api(transaction: Transaction, risk_analysis: RiskAnalysis):
    message = {
        "alert_type": "high_risk_transaction",
        "transaction_id": transaction.transaction_id,
        "risk_score": risk_analysis.risk_score,
        "risk_factors": risk_analysis.risk_factors,
        "reasoning": risk_analysis.reasoning,
        "transaction_details": {
            "amount": transaction.amount,
            "currency": transaction.currency,
            "customer": {
                "id": transaction.customer.id,
                "country": transaction.customer.country,
                "ip_address": transaction.customer.ip_address
            },
            "payment_method": {
                "type": transaction.payment_method.type,
                "last_four": transaction.payment_method.last_four,
                "country_of_issue": transaction.payment_method.country_of_issue
            },
            "merchant": {
                "id": transaction.merchant.id,
                "name": transaction.merchant.name,
                "category": transaction.merchant.category
            }    
        }
    }   
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(settings.notifyadmin_api_url, json=message)
            response.raise_for_status()
            print(f"Notification sent successfully: {response.status_code}")
    except httpx.RequestError as e:
        print(f"Error sending notification: {e}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")

