"""
CAG Architecture - Conversation Memory Module
Manages conversation history, user profiles, and context awareness

FIXES:
- should_ask_name: now triggers on ANY first message, not just greetings
- Added ConversationStage state machine to track where we are in the flow
- extract_name_from_response: improved patterns and edge-case handling
- get_stage_instruction: injects stage-aware rule into every prompt
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


# ============================================================
# CONVERSATION STAGE — enforced state machine
# ============================================================

class ConversationStage:
    """
    Tracks the current phase of the conversation so the system prompt
    can tell the model exactly what to do at each turn.

    Flow:
        AWAITING_NAME
            ↓  (name received)
        AWAITING_PROBLEM
            ↓  (problem stated)
        AWAITING_CLARIFICATION
            ↓  (clarification answered)
        PROVIDING_SOLUTION
            ↓  (solution given)
        FOLLOWUP          ← loops here for follow-up questions
    """
    AWAITING_NAME          = "awaiting_name"
    AWAITING_PROBLEM       = "awaiting_problem"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PROVIDING_SOLUTION     = "providing_solution"
    FOLLOWUP               = "followup"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Message:
    """Single message in conversation"""
    role: str                          # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserProfile:
    """User profile information"""
    name: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    first_interaction: str = field(default_factory=lambda: datetime.now().isoformat())
    last_interaction: str = field(default_factory=lambda: datetime.now().isoformat())
    total_interactions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        return cls(**data)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Conversation Memory Manager

    Features:
    1. Stores conversation history
    2. Manages user profiles (name, preferences)
    3. Enforces a strict conversation stage machine
    4. Provides stage-aware system prompt instructions
    5. Handles multi-turn conversations
    6. Optional persistent storage to disk
    """

    def __init__(self, config, max_history: int = 10):
        """
        Initialize conversation memory.

        Args:
            config: CAG configuration object
            max_history: Maximum number of message pairs to keep in memory
        """
        self.config = config
        self.max_history = max_history

        # Memory storage
        self.messages: List[Message] = []
        self.user_profile: UserProfile = UserProfile()

        # ── Stage machine ──────────────────────────────────────
        # Start waiting for the user's name on every fresh session.
        self.stage: str = ConversationStage.AWAITING_NAME
        # Legacy flag kept for backward compatibility
        self.is_first_interaction: bool = True
        self.name_asked: bool = False

        # File paths
        self.memory_dir = os.path.join(
            os.path.dirname(config.cache_file_path), "memory"
        )
        os.makedirs(self.memory_dir, exist_ok=True)

        self.conversation_file = os.path.join(
            self.memory_dir, "conversation_history.json"
        )
        self.profile_file = os.path.join(
            self.memory_dir, "user_profile.json"
        )

        # Load existing memory (if persistence is enabled)
        self.load_memory()

    # ----------------------------------------------------------
    # Stage helpers
    # ----------------------------------------------------------

    def advance_stage(self, new_stage: str):
        """Move the conversation to a new stage."""
        self.stage = new_stage

    def get_stage_instruction(self) -> str:
        """
        Return a terse instruction that matches the current stage.
        This is injected into the system prompt so the model always
        knows what to do next.
        """
        instructions = {
            ConversationStage.AWAITING_NAME: (
                "The user has NOT given their name yet. "
                "Your ONLY job right now is to ask for their name politely. "
                "Do NOT help with any other request until you have their name."
            ),
            ConversationStage.AWAITING_PROBLEM: (
                f"You know the user's name is {self.user_profile.name}. "
                "Greet them by name and ask how you can help. "
                "Do NOT solve anything yet — just invite them to share their issue."
            ),
            ConversationStage.AWAITING_CLARIFICATION: (
                f"The user ({self.user_profile.name}) has described a problem. "
                f"If you haven't greeted them by name yet in this turn, greet them as '{self.user_profile.name}' first. "
                "Then ask exactly ONE focused clarifying question to gather the missing detail "
                "you need before you can give a solution. "
                "Do NOT provide any solution in this turn."
            ),
            ConversationStage.PROVIDING_SOLUTION: (
                f"You now have enough detail about {self.user_profile.name}'s problem. "
                "Provide a clear, direct solution. "
                "You may end with a single optional follow-up offer, "
                "but do NOT ask for more clarification."
            ),
            ConversationStage.FOLLOWUP: (
                f"The user ({self.user_profile.name}) is asking a follow-up question "
                "about the solution you already gave. "
                "Answer ONLY the specific question asked — concisely. "
                "Do NOT re-explain the full previous solution. "
                "If they mention a completely new problem, treat it as a new issue: "
                "ask one clarifying question before solving."
            ),
        }
        return instructions.get(self.stage, "")

    # ----------------------------------------------------------
    # Message management
    # ----------------------------------------------------------

    def add_message(self, role: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to conversation history and update the stage.

        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
        """
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)

        # Trim to max history
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]

        # Update profile counters
        if role == 'user':
            self.user_profile.total_interactions += 1
            self.user_profile.last_interaction = datetime.now().isoformat()
            self.is_first_interaction = False

        # Auto-save
        if self.config.enable_cache_persistence:
            self.save_memory()

    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Message]:
        """
        Get conversation history.

        Args:
            last_n: Number of recent message pairs (None = all)
        """
        if last_n is None:
            return self.messages
        return self.messages[-(last_n * 2):]

    def format_conversation_for_prompt(self) -> str:
        """
        Format recent conversation history in Llama 3 chat tokens
        for direct inclusion in the prompt.
        """
        if not self.messages:
            return ""

        formatted = []
        for msg in self.get_conversation_history(last_n=5):
            if msg.role == "user":
                formatted.append(
                    f"<|start_header_id|>user<|end_header_id|>\n"
                    f"{msg.content}<|eot_id|>"
                )
            else:
                formatted.append(
                    f"<|start_header_id|>assistant<|end_header_id|>\n"
                    f"{msg.content}<|eot_id|>"
                )

        return "\n".join(formatted)

    def get_context_summary(self) -> str:
        """Get a short text summary of conversation context."""
        if not self.messages:
            return ""

        recent = self.get_conversation_history(last_n=3)
        parts = []
        if self.user_profile.name:
            parts.append(f"User's name: {self.user_profile.name}")

        if recent:
            parts.append("Recent conversation:")
            for msg in recent[-6:]:
                prefix = "User" if msg.role == "user" else "Assistant"
                content = (msg.content[:100] + "...") if len(msg.content) > 100 else msg.content
                parts.append(f"{prefix}: {content}")

        return "\n".join(parts)

    # ----------------------------------------------------------
    # Name handling  (FIXED)
    # ----------------------------------------------------------

    def should_ask_name(self, query: str) -> bool:
        """
        FIX: Previously only returned True on greetings/short messages.
        Now returns True on ANY first interaction when the name is unknown.

        This ensures that users who jump straight to their problem
        (e.g. "My website is down!") are still asked for their name first.
        """
        # Don't ask if name is already known
        if self.user_profile.name:
            return False

        # Don't ask again if already asked this session
        if self.name_asked:
            return False

        # FIX: Ask on ANY first interaction, regardless of message content
        if self.stage == ConversationStage.AWAITING_NAME:
            self.name_asked = True
            return True

        return False

    def extract_name_from_response(self, user_response: str) -> Optional[str]:
        """
        Extract the user's name from their message.

        Handles:
          - "My name is Sarah"
          - "I'm Alex"  /  "I am Alex"
          - "Call me Mike"
          - "Just call me Mike"
          - "People usually call me Tom, and I need help with my CRM"
          - Bare name: "Sarah" / "Ahmed"
        """
        response_stripped = user_response.strip()
        response_lower = response_stripped.lower()

        # Ordered patterns — more specific first
        patterns = [
            "people usually call me ",
            "everyone calls me ",
            "just call me ",
            "my name is ",
            "name's ",
            "call me ",
            "i'm ",
            "i am ",
        ]

        for pattern in patterns:
            if pattern in response_lower:
                idx = response_lower.index(pattern)
                after = response_stripped[idx + len(pattern):].strip()
                # Take the first word, strip punctuation and conjunctions
                name_candidate = after.split()[0].strip('.,!?;:"\'-') if after.split() else ""
                # Reject conjunctions / filler words
                if name_candidate.lower() not in {
                    "and", "but", "or", "so", "the", "a", "an",
                    "yes", "no", "ok", "sure", "yeah", "nope", ""
                }:
                    return name_candidate.capitalize()

        # Last resort: very short response (1-2 words) that looks like a name
        words = response_stripped.split()
        if len(words) <= 2:
            candidate = words[0].strip('.,!?;:"\'-').capitalize()
            reject = {
                "yes", "no", "ok", "sure", "yeah", "nope", "hi",
                "hello", "hey", "fine", "good", "great", "help",
                "please", "thanks", "thank"
            }
            if candidate.lower() not in reject and len(candidate) >= 2:
                return candidate

        return None

    def set_user_name(self, name: str):
        """Save the user's name and advance the stage."""
        self.user_profile.name = name
        # After we have the name, next expectation is for them to share a problem
        self.advance_stage(ConversationStage.AWAITING_PROBLEM)
        if self.config.enable_cache_persistence:
            self.save_memory()

    # ----------------------------------------------------------
    # Personalization
    # ----------------------------------------------------------

    def get_personalized_greeting(self) -> str:
        if self.user_profile.name:
            return f"Hello {self.user_profile.name}! How can I help you today?"
        return "Hello! How can I help you today?"

    # ----------------------------------------------------------
    # Reset / clear
    # ----------------------------------------------------------

    def clear_conversation(self):
        """Clear conversation history but keep user profile."""
        self.messages = []
        self.stage = ConversationStage.AWAITING_PROBLEM  # name still known
        if self.config.enable_cache_persistence:
            self.save_memory()

    def reset_all(self):
        """Reset everything including user profile."""
        self.messages = []
        self.user_profile = UserProfile()
        self.stage = ConversationStage.AWAITING_NAME
        self.is_first_interaction = True
        self.name_asked = False
        if self.config.enable_cache_persistence:
            self.save_memory()

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def save_memory(self):
        """Save conversation history and user profile to disk."""
        try:
            conversation_data = {
                'messages': [msg.to_dict() for msg in self.messages],
                'is_first_interaction': self.is_first_interaction,
                'name_asked': self.name_asked,
                'stage': self.stage,
            }
            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            with open(self.profile_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile.to_dict(), f, indent=2, ensure_ascii=False)

        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Failed to save memory: {e}")

    def load_memory(self):
        """Load conversation history and user profile from disk."""
        try:
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.messages = [Message(**m) for m in data.get('messages', [])]
                self.is_first_interaction = data.get('is_first_interaction', True)
                self.name_asked = data.get('name_asked', False)
                self.stage = data.get('stage', ConversationStage.AWAITING_NAME)

                if self.config.verbose:
                    print(f"📝 Loaded {len(self.messages)} messages from memory")

            if os.path.exists(self.profile_file):
                with open(self.profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                self.user_profile = UserProfile.from_dict(profile_data)

                if self.config.verbose and self.user_profile.name:
                    print(f"👤 Loaded user profile: {self.user_profile.name}")

                # If we already know the name, don't stay in AWAITING_NAME
                if self.user_profile.name and self.stage == ConversationStage.AWAITING_NAME:
                    self.stage = ConversationStage.AWAITING_PROBLEM

        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Failed to load memory: {e}")

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_messages': len(self.messages),
            'user_name': self.user_profile.name,
            'total_interactions': self.user_profile.total_interactions,
            'first_interaction': self.user_profile.first_interaction,
            'last_interaction': self.user_profile.last_interaction,
            'is_first_interaction': self.is_first_interaction,
            'stage': self.stage,
        }