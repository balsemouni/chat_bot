import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db
from dependencies import get_current_user, apply_rate_limit
from security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    blacklist_token,
)
from user import User
from user_service import (
    authenticate_user,
    change_password,
    create_user,
    delete_user_data,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Schemas ─────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MessageResponse(BaseModel):
    message: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: bool
    roles: list

    model_config = {"from_attributes": True}

    @classmethod
    def model_validate(cls, user):
        return cls(
            id=str(user.id),
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            roles=user.roles,
        )


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)):
    await apply_rate_limit(request, settings.RATE_LIMIT_REGISTER, "register")

    if await get_user_by_email(db, body.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    if await get_user_by_username(db, body.username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")

    user = await create_user(db, email=body.email, username=body.username,
                              password=body.password, full_name=body.full_name)
    logger.info(f"New user registered: {user.id}")
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    await apply_rate_limit(request, settings.RATE_LIMIT_LOGIN, "login")

    user = await authenticate_user(db, body.email, body.password)

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if user.is_locked:
        raise HTTPException(status_code=status.HTTP_423_LOCKED,
                            detail="Account temporarily locked due to too many failed attempts")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated")

    access_token = create_access_token(user_id=str(user.id), email=user.email, roles=user.roles)
    refresh_token = create_refresh_token(user_id=str(user.id))

    logger.info(f"User {user.id} logged in from {request.client.host}")
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh_token(body: RefreshRequest, request: Request, db: AsyncSession = Depends(get_db)):
    await apply_rate_limit(request, settings.RATE_LIMIT_REFRESH, "refresh")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
    )
    try:
        payload = await decode_token(body.refresh_token, expected_type="refresh")
        user_id: str = payload.get("sub")
    except JWTError:
        raise credentials_exception

    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise credentials_exception

    new_access_token = create_access_token(user_id=str(user.id), email=user.email, roles=user.roles)
    return AccessTokenResponse(access_token=new_access_token)


@router.post("/logout", response_model=MessageResponse)
async def logout(body: LogoutRequest, request: Request, current_user: User = Depends(get_current_user)):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        raw_access = auth_header[7:]
        try:
            payload = jwt.decode(raw_access, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            jti = payload.get("jti")
            exp = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
            if jti:
                await blacklist_token(jti, exp)
        except JWTError:
            pass

    if body.refresh_token:
        try:
            payload = jwt.decode(body.refresh_token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            jti = payload.get("jti")
            exp = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
            if jti:
                await blacklist_token(jti, exp)
        except JWTError:
            pass

    logger.info(f"User {current_user.id} logged out")
    return MessageResponse(message="Successfully logged out")


@router.post("/change-password", response_model=MessageResponse)
async def change_password_endpoint(body: ChangePasswordRequest,
                                    current_user: User = Depends(get_current_user),
                                    db: AsyncSession = Depends(get_db)):
    success = await change_password(db, current_user, body.current_password, body.new_password)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")
    return MessageResponse(message="Password updated successfully")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


@router.delete("/me", response_model=MessageResponse)
async def erase_my_data(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await delete_user_data(db, current_user)
    logger.info(f"GDPR erasure completed for user {current_user.id}")
    return MessageResponse(message="Your data has been erased")


@router.post("/verify-token", response_model=UserResponse)
async def verify_token(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)