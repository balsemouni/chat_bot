import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from security import hash_password, verify_password
from user import User

logger = logging.getLogger(__name__)

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email.lower()))
    return result.scalar_one_or_none()


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.username == username.lower()))
    return result.scalar_one_or_none()


async def get_all_users(db: AsyncSession) -> list[User]:
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return result.scalars().all()


async def create_user(
    db: AsyncSession,
    email: str,
    username: str,
    password: str,
    full_name: Optional[str] = None,
    roles: list[str] = None,
) -> User:
    user = User(
        email=email.lower(),
        username=username.lower(),
        hashed_password=hash_password(password),
        full_name=full_name,
        roles=roles or ["user"],
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    logger.info(f"Created user {user.id} ({user.email})")
    return user


async def authenticate_user(
    db: AsyncSession, email: str, password: str
) -> Optional[User]:
    user = await get_user_by_email(db, email)
    if not user:
        verify_password(password, "$2b$12$fakehashfakehashfakehashfakehashfakehash")
        return None

    if user.is_locked:
        logger.warning(f"Login attempt on locked account {user.email}")
        return None

    if not verify_password(password, user.hashed_password):
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=LOCKOUT_DURATION_MINUTES
            )
            logger.warning(
                f"Account {user.email} locked after {MAX_FAILED_ATTEMPTS} failed attempts"
            )
        await db.flush()
        return None

    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = datetime.now(timezone.utc)
    await db.flush()
    return user


async def update_user(
    db: AsyncSession,
    user: User,
    full_name: Optional[str] = None,
    username: Optional[str] = None,
) -> User:
    if full_name is not None:
        user.full_name = full_name
    if username is not None:
        user.username = username.lower()
    user.updated_at = datetime.now(timezone.utc)
    await db.flush()
    await db.refresh(user)
    return user


async def change_password(
    db: AsyncSession, user: User, current_password: str, new_password: str
) -> bool:
    if not verify_password(current_password, user.hashed_password):
        return False
    user.hashed_password = hash_password(new_password)
    user.updated_at = datetime.now(timezone.utc)
    await db.flush()
    logger.info(f"Password changed for user {user.id}")
    return True


async def deactivate_user(db: AsyncSession, user: User) -> User:
    user.is_active = False
    user.updated_at = datetime.now(timezone.utc)
    await db.flush()
    logger.info(f"User {user.id} deactivated (GDPR)")
    return user


async def delete_user_data(db: AsyncSession, user: User) -> None:
    user.email = f"deleted_{user.id}@deleted.invalid"
    user.username = f"deleted_{user.id}"
    user.full_name = None
    user.is_active = False
    user.hashed_password = "DELETED"
    user.updated_at = datetime.now(timezone.utc)
    await db.flush()
    logger.info(f"User {user.id} data erased (GDPR erasure)")