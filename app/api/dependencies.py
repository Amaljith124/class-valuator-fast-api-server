import hmac
from fastapi import Header, HTTPException
from app.core.config import settings

def require_internal_token(x_internal_token: str | None = Header(default=None)):
    if x_internal_token and hmac.compare_digest(x_internal_token, settings.internal_api_token):
        return None
    raise HTTPException(status_code=401, detail="Unauthorised")
