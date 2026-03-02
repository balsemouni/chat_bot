import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from database import engine, Base, get_db
from user_service import get_all_users

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Auth Microservice...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Auth Microservice", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from auth import router as auth_router
    app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
except ImportError:
    pass


@app.get("/health")
def health():
    return {"status": "ok", "service": "auth-service", "port": 8001}


@app.get("/users", tags=["Users"])
async def list_users(db: AsyncSession = Depends(get_db)):
    users = await get_all_users(db)
    return [
        {
            "id":         str(u.id),
            "email":      u.email,
            "username":   u.username,
            "full_name":  u.full_name,
            "is_active":  u.is_active,
            "roles":      u.roles,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]

@app.get("/users/{user_id}/exists", tags=["Users"])
async def user_exists(user_id: str, db: AsyncSession = Depends(get_db)):
    from user_service import get_user_by_id
    user = await get_user_by_id(db, user_id)
    return {"exists": user is not None}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8006, reload=True)