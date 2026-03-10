"""
main.py - Session Microservice  →  port 8002
"""
import os
import httpx
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from database import get_db, init_db
from crud import SessionCRUD

USER_SERVICE_URL = os.getenv("USER_SERVICE_URL", "http://localhost:8006")

app = FastAPI(title="Session Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    init_db()


# ── Helpers ──────────────────────────────────────────────────────────────────

async def verify_user(user_id: str):
    """Call user-service to confirm the user exists before creating a session."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{USER_SERVICE_URL}/users/{user_id}/exists", timeout=5)
            if resp.status_code != 200 or not resp.json().get("exists"):
                raise HTTPException(404, f"User {user_id} not found in user-service.")
        except httpx.RequestError:
            raise HTTPException(503, "User service unreachable.")


# ── Schemas ──────────────────────────────────────────────────────────────────

class CreateSessionBody(BaseModel):
    user_id: str
    title:   Optional[str] = None

class UpdateTitleBody(BaseModel):
    title: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "session-service", "port": 8005}


@app.post("/sessions", status_code=201)
async def create_session(body: CreateSessionBody, db: Session = Depends(get_db)):
    await verify_user(body.user_id)
    session = SessionCRUD(db).create(body.user_id, body.title)
    return session.to_dict()


@app.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    # Try Redis hot cache first
    cached = SessionCRUD(db).get_cached(session_id)
    if cached:
        return {**cached, "source": "cache"}
    session = SessionCRUD(db).get_by_id(session_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    return {**session.to_dict(), "source": "db"}


@app.get("/users/{user_id}/sessions")
def list_sessions(user_id: str, db: Session = Depends(get_db)):
    sessions = SessionCRUD(db).list_by_user(user_id)
    return [s.to_dict() for s in sessions]


@app.patch("/sessions/{session_id}/title")
def update_title(session_id: str, body: UpdateTitleBody, db: Session = Depends(get_db)):
    session = SessionCRUD(db).update_title(session_id, body.title)
    if not session:
        raise HTTPException(404, "Session not found.")
    return session.to_dict()


@app.post("/sessions/{session_id}/touch", status_code=204)
def touch_session(session_id: str, db: Session = Depends(get_db)):
    """Called by history-service after saving a message to bump updated_at."""
    SessionCRUD(db).touch(session_id)


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    if not SessionCRUD(db).delete(session_id):
        raise HTTPException(404, "Session not found.")


# Internal — called by history-service before saving messages
@app.get("/sessions/{session_id}/exists")
def session_exists(session_id: str, db: Session = Depends(get_db)):
    return {"exists": SessionCRUD(db).exists(session_id)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)