# #config.py is used for storing the configuration of the application from .env and the LLM selection 

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI API settings
    OPENAI_API_KEY: str 
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    
    #Anthropic API settings
    anthropic_api_key: str 
    ANTHROPIC_API_URL: str = "https://api.anthropic.com/v1/messages"
    
    # Default LLM provider
    llm_provider: str = "claude" #or "openai" if the default is OpenAI 
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    auth_username = "Testuser"
    auth_password = "Random321"

    notifyadmin_api_url: str

settings = Settings()