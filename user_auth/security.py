import logging
import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from config import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory token blacklist (lost on restart, fine for dev/testing)
_blacklisted_tokens: dict = {}


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, email: str, roles: list[str]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "roles": roles,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "iat": datetime.now(timezone.utc),
        "exp": expire,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "iat": datetime.now(timezone.utc),
        "exp": expire,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


async def decode_token(token: str, expected_type: str = "access") -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise

    if payload.get("type") != expected_type:
        raise JWTError(f"Invalid token type: expected {expected_type}")

    jti = payload.get("jti")
    if jti and is_token_blacklisted(jti):
        raise JWTError("Token has been revoked")

    return payload


async def blacklist_token(jti: str, exp: datetime) -> None:
    _blacklisted_tokens[jti] = exp
    logger.info(f"Token {jti} blacklisted")


def is_token_blacklisted(jti: str) -> bool:
    if jti not in _blacklisted_tokens:
        return False
    if datetime.now(timezone.utc) > _blacklisted_tokens[jti]:
        del _blacklisted_tokens[jti]
        return False
    return True


async def check_rate_limit(key: str, limit: int, window_seconds: int = 60) -> bool:
    # No Redis — always allow in dev mode
    return True