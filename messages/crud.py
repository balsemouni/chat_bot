import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from models import ChatMessage, VALID_ROLES


class MessageCRUD:
    def __init__(self, db: Session):
        self.db = db

    def create(self, session_id: str, role: str, content: str) -> ChatMessage:
        if role not in VALID_ROLES:
            raise ValueError(f"role must be 'user' or 'agent', got '{role}'")
        msg = ChatMessage(
            session_id=uuid.UUID(session_id),
            role=role,
            content=content,
        )
        self.db.add(msg)
        self.db.commit()
        self.db.refresh(msg)
        return msg

    def get_by_id(self, message_id: str) -> Optional[ChatMessage]:
        return self.db.query(ChatMessage).filter(
            ChatMessage.id == uuid.UUID(message_id)
        ).first()

    def list_by_session(self, session_id: str, limit: int = 100, offset: int = 0) -> List[ChatMessage]:
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == uuid.UUID(session_id))
            .order_by(ChatMessage.created_at.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def delete(self, message_id: str) -> bool:
        msg = self.get_by_id(message_id)
        if not msg:
            return False
        self.db.delete(msg)
        self.db.commit()
        return True

    def delete_by_session(self, session_id: str) -> int:
        count = self.db.query(ChatMessage).filter(
            ChatMessage.session_id == uuid.UUID(session_id)
        ).delete(synchronize_session=False)
        self.db.commit()
        return count