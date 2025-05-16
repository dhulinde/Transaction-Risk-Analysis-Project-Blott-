# #config.py is used for storing the configuration of the application from .env and the LLM selection 

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI API settings
    openai_api_key: str 
    openai_api_url: str = "https://api.openai.com/v1/chat/completions"
    
    #Anthropic API settings
    anthropic_api_key: str 
    anthropic_api_url: str = "https://api.anthropic.com/v1/messages"
    
    # Default LLM provider
    #llm_provider: str = "openai" #or "claude" if the default is Claude
    llm_provider: str = "claude" #or "claude" if the default is Claude 
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    auth_username: str = "Testuser"
    auth_password: str = "Random321"

    notifyadmin_api_url: str

settings = Settings()