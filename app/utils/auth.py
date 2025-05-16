from fastapi.security import HTTPBasicCredentials
from app.config import settings

def verify_credentials(credentials: HTTPBasicCredentials) -> bool:
    return (
        credentials.username == settings.auth_username and
        credentials.password == settings.auth_password
    )
