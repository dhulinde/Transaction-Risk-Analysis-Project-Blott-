# Transaction Risk Analysis Project

This project implements a system that receives transaction data via webhook, uses Large Language Models (LLMs) to analyze risk patterns, and notifies administrators about suspicious transactions.

## Features

Webhook Service: Accepts POST requests with JSON transaction data, with basic authentication \
LLM Integration: Uses LLMs (OpenAI GPT, Claude, or Groq) to analyze transaction risk \
Admin Notification: Notifies administrators of high-risk transactions \
Support for Multiple LLM Providers: Can switch between OpenAI, Claude, and Groq

# Setup

Clone the repository \
Create a virtual environment: \
python -m venv venv \
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

## .env
Create a .env file with the following variables: \
openai_api_key=your_openai_api_key \
anthropic_api_key=your_anthropic_api_key \
groq_api_key=your_groq_api_key \

!Please dont keep any spaces between the = and your key

## config.py
### Optional: customize these 
auth_username: str = your_preferred_username \
auth_password: str = your_preferred_password

### Default LLM provider (openai, claude, or groq)
llm_provider=openai (To use openai) \
llm_provider=claude (To use claude/Anthropic) \
llm_provider=qroq (To use grop (Deepseek, Llama, Gemma))


# Running the Application
Start the FastAPI server: \
uvicorn main:app --reload 

The API will be available at http://127.0.0.1:8000

## API Endpoints
Transaction Webhook \
POST /webhook/transaction (http://127.0.0.1:8000/webhook/transaction) \
Authentication: Basic Authentication (set in .env) \
Request Body: JSON conforming to the Transaction model (Refer postman file for request bodies used in manual testing) \
Response: Transaction ID, risk score, and recommended action \

## Testing 
Run all tests: \
pytest 

### Run specific test files: 
pytest tests/test_api.py \ 
pytest tests/test_llm.py \
pytest tests/test_integration.py \
pytest tests/test_prompts.py \
pytest tests/test_suite.py \

### Run with coverage:
pytest --cov=app tests 

## Example Transactions 
### Normal Transaction 
json{ \
  "transaction_id": "tx_normal_01",\
  "timestamp": "2025-05-07T14:30:45Z",\
  "amount": 129.99,\
  "currency": "USD",\
  "customer": {\
    "id": "cust_98765zyxwv",\
    "country": "US",\
    "ip_address": "192.168.1.1"\
  },\
  "payment_method": {\
    "type": "credit_card",\
    "last_four": "4242",\
    "country_of_issue": "US"\
  },\
  "merchant": {\
    "id": "merch_abcde12345",\
    "name": "Example Store",\
    "category": "electronics"\
  }\
}

### High Risk Transaction
json{ \
  "transaction_id": "tx_high_risk_01",\
  "timestamp": "2025-05-07T02:12:33Z",\
  "amount": 4999.99,\
  "currency": "USD",\
  "customer": {\
    "id": "cust_12345abcde",\
    "country": "US",\
    "ip_address": "203.0.113.195"\
  },\
  "payment_method": {\
    "type": "credit_card",\
    "last_four": "9876",\
    "country_of_issue": "RU"\
  },\
  "merchant": {\
    "id": "merch_67890fghij",\
    "name": "Luxury Goods",\
    "category": "jewelry"\
  }\
}
