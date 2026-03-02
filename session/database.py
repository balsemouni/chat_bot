# database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# ✅ Use psycopg2 (sync) instead of asyncpg (async)
DATABASE_URL = "postgresql+psycopg2://postgres:balsem@localhost:5432/session_db"

engine       = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    print("[SessionService] Tables ready.")