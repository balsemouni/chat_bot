"""
Session Manager — per-connection state
Mirrors the monolith's ConversationMemory + UIHandler pattern.

History is stored as a deque (maxlen=20, same as memory.py) and
is passed to the LLM on every request so CAG has full conversation context.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
from collections import deque


@dataclass
class Utterance:
    speaker: str          # "user" | "ai"
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    interrupted: bool = False


class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self.state = "idle"           # idle | listening | thinking | ai_speaking
        self.interrupted = False
        self.history: deque = deque(maxlen=20)   # same limit as memory.py
        self.created_at = datetime.now()

        # VAD state — mirrors vad_processor_thread in the monolith
        self.user_is_talking = False
        self.ai_is_speaking = False
        self.ai_is_thinking = False

    # ── Transcript helpers ────────────────────────────────────────────────────

    def add_user_utterance(self, text: str):
        self.history.append(Utterance(speaker="user", text=text))

    def add_ai_utterance(self, text: str, interrupted: bool = False):
        self.history.append(
            Utterance(speaker="ai", text=text, interrupted=interrupted)
        )

    def get_full_transcript(self) -> str:
        """Human-readable transcript for HubSpot."""
        lines = []
        for u in self.history:
            ts = u.timestamp[:19].replace("T", " ")
            suffix = " [interrupted]" if u.interrupted else ""
            lines.append(f"[{ts}] {u.speaker.upper()}: {u.text}{suffix}")
        return "\n".join(lines)

    def get_messages_for_llm(self) -> List[Dict]:
        """
        OpenAI-compatible message list — identical to memory.py get_messages().
        Passed as `history` to the LLM service so CAG has full context.
        """
        messages = []
        for u in self.history:
            role = "user" if u.speaker == "user" else "assistant"
            messages.append({"role": role, "content": u.text})
        return messages

    def mark_interrupted(self):
        """Mark the most recent AI utterance as interrupted (barge-in)."""
        for u in reversed(self.history):
            if u.speaker == "ai":
                u.interrupted = True
                break


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self, session_id: str) -> Session:
        s = Session(session_id)
        self._sessions[session_id] = s
        return s

    def get(self, session_id: str) -> Session:
        return self._sessions.get(session_id)

    def remove(self, session_id: str):
        self._sessions.pop(session_id, None)

    def list_all(self) -> List[Dict]:
        return [
            {
                "id":    s.id,
                "state": s.state,
                "turns": len(s.history),
                "created_at": s.created_at.isoformat(),
            }
            for s in self._sessions.values()
        ]