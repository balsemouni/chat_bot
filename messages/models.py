import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()

VALID_ROLES = {"user", "agent"}


class ChatMessage(Base):
    __tablename__ = "messages"

    id         = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)  # no DB FK — validated via HTTP
    role       = Column(String(10), nullable=False)   # "user" | "agent"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":         str(self.id),
            "session_id": str(self.session_id),
            "role":       self.role,
            "content":    self.content,
            "created_at": self.created_at.isoformat(),
        }