from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

token_auth_scheme = HTTPBearer()

# Dummy token for authentication
VALID_TOKEN = "mysecuretoken"

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(token_auth_scheme)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authentication token")
    return "authorized_user"
