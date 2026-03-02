"""
Message Microservice  →  port 8003

POST   /sessions/{session_id}/messages      add message  (validates session via HTTP)
GET    /sessions/{session_id}/messages      list messages
GET    /messages/{message_id}               get one message
DELETE /messages/{message_id}               delete one message
DELETE /sessions/{session_id}/messages      clear all messages in session
"""
import os
import httpx
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, init_db
from crud import MessageCRUD

SESSION_SERVICE_URL = os.getenv("SESSION_SERVICE_URL", "http://localhost:8005")

app = FastAPI(title="Message Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def verify_session(session_id: str):
    """Confirm session exists in session-service before saving a message."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{SESSION_SERVICE_URL}/sessions/{session_id}/exists", timeout=5
            )
            if resp.status_code != 200 or not resp.json().get("exists"):
                raise HTTPException(404, f"Session {session_id} not found.")
        except httpx.RequestError:
            raise HTTPException(503, "Session service unreachable.")

async def touch_session(session_id: str):
    """Tell session-service to bump updated_at after a new message."""
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{SESSION_SERVICE_URL}/sessions/{session_id}/touch", timeout=5
            )
        except httpx.RequestError:
            pass  # non-critical — don't fail the message write over this


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateMessageBody(BaseModel):
    role:    str   # "user" | "agent"
    content: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "message-service", "port": 8003}


@app.post("/sessions/{session_id}/messages", status_code=201)
async def create_message(
    session_id: str, body: CreateMessageBody, db: Session = Depends(get_db)
):
    await verify_session(session_id)
    try:
        msg = MessageCRUD(db).create(session_id, body.role, body.content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    await touch_session(session_id)
    return msg.to_dict()


@app.get("/sessions/{session_id}/messages")
def list_messages(
    session_id: str,
    limit:  int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return [m.to_dict() for m in MessageCRUD(db).list_by_session(session_id, limit, offset)]


@app.get("/messages/{message_id}")
def get_message(message_id: str, db: Session = Depends(get_db)):
    msg = MessageCRUD(db).get_by_id(message_id)
    if not msg:
        raise HTTPException(404, "Message not found.")
    return msg.to_dict()


@app.delete("/messages/{message_id}", status_code=204)
def delete_message(message_id: str, db: Session = Depends(get_db)):
    if not MessageCRUD(db).delete(message_id):
        raise HTTPException(404, "Message not found.")


@app.delete("/sessions/{session_id}/messages")
def clear_messages(session_id: str, db: Session = Depends(get_db)):
    deleted = MessageCRUD(db).delete_by_session(session_id)
    return {"deleted": deleted}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8007, reload=True)