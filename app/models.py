from pydantic import BaseModel
from typing import List


#Json 
"""  { 
"transaction_id": "tx_12345abcde", 
"timestamp": "2025-05-07T14:30:45Z", 
"amount": 129.99, 
"currency": "USD", 
"customer": { 
"id": "cust_98765zyxwv", 
"country": "US", 
"ip_address": "192.168.1.1" 
}, 
"payment_method": { 
"type": "credit_card", 
"last_four": "4242", 
"country_of_issue": "CA" 
}, 
"merchant": { 
"id": "merch_abcde12345", 
"name": "Example Store", 
"category": "electronics" 
} 
}
"""

class Transaction(BaseModel):
    transaction_id: str 
    timestamp: str 
    amount: float 
    currency: str
    customer : 'Customer'
    payment_method : 'PaymentMethod'   
    merchant : 'Merchant'

#json for customer
""" "customer": { 
"id": "cust_98765zyxwv", 
"country": "US", 
"ip_address": "192.168.1.1" 
}, """

class Customer(BaseModel):
    id: str
    country: str 
    ip_address: str

#json for payment method
""" 
"payment_method": { 
"type": "credit_card", 
"last_four": "4242", 
"country_of_issue": "CA" 
}, 
"""

class PaymentMethod(BaseModel):
    type: str 
    last_four: str 
    country_of_issue: str 

#json for merchant
""" "merchant": { 
"id": "merch_abcde12345", 
"name": "Example Store", 
"category": "electronics" 
} """

class Merchant(BaseModel):
    id: str 
    name: str 
    category: str

#json for risk analysis
"""
{ 
"risk_score": 0.0-1.0, 
"risk_factors": ["factor1", "factor2"...], 
"reasoning": "A brief explanation of your analysis", 
"recommended_action": "allow|review|block"
}
"""

class RiskAnalysis(BaseModel):
    risk_score: float
    risk_factors: List[str]
    reasoning: str
    recommended_action: str

#json for admin notification
"""
 { 
"alert_type": "high_risk_transaction", 
"transaction_id": "tx_12345abcde", 
"risk_score": 0.85, 
"risk_factors": [ 
"Customer country (US) differs from card country (CA)", 
"Transaction amount significantly higher than customer 
average", 
"Multiple transactions within short timeframe" 
], 
"transaction_details": { 
// The original transaction JSON 
}, 
"llm_analysis": "This transaction shows multiple risk 
indicators including cross-border payment method, unusual amount 
for this customer, and velocity pattern concerns." 
} 
"""

class AdminNotification(BaseModel):
    alert_type: str
    transaction_id: str
    risk_score: float
    risk_factors: List[str]
    transaction_details: dict
    llm_analysis: str
