"""
CAG Architecture - Inference Engine Module
Stateless inference engine that uses pre-computed cache

FIXES:
- _build_query_prompt: added a full rule-based SYSTEM_PROMPT so the model
  always follows the correct conversation flow regardless of user input.
- _build_query_prompt now accepts an optional `memory` argument and injects:
    1. The stage-specific instruction (what to do THIS turn)
    2. The conversation history (so the model has context)
- generate / stream_query callers must pass `memory=` to activate stage logic.
"""

import torch
import gc
from typing import Optional, Dict, Any


# ============================================================
# SYSTEM PROMPT  — the backbone of correct behaviour
# ============================================================

SYSTEM_PROMPT = """\
You are a professional, friendly technical support agent for Ask Novation.
You MUST follow these rules in STRICT ORDER every conversation:

══════════════════════════════════════════════════════
RULE 1 — NAME FIRST (non-negotiable)
══════════════════════════════════════════════════════
• If you do NOT yet know the user's name, your ONLY response is to ask
  for their name — nothing else.
• Even if the user jumps straight to a problem, ask for their name first.
• Once you have their name, use it naturally throughout the conversation.

══════════════════════════════════════════════════════
RULE 2 — ONE CLARIFYING QUESTION BEFORE SOLVING
══════════════════════════════════════════════════════
• When a user describes a problem for the first time, ask exactly ONE
  focused clarifying question.
• Do NOT provide any solution in the same turn as your clarifying question.
• Good clarifying questions target: error codes, platform/stack, file size,
  frequency of occurrence, or the specific endpoint/feature affected.

══════════════════════════════════════════════════════
RULE 3 — SOLVE AFTER CLARIFICATION
══════════════════════════════════════════════════════
• Once you have sufficient detail, give a direct, actionable solution.
• You may end with ONE optional offer to walk them through steps.
• Do NOT ask further clarifying questions at this stage.

══════════════════════════════════════════════════════
RULE 4 — FOCUSED FOLLOW-UPS
══════════════════════════════════════════════════════
• If the user asks a specific follow-up question about the solution,
  answer ONLY that question, concisely.
• Do NOT re-explain the full solution.

══════════════════════════════════════════════════════
RULE 5 — NEW PROBLEMS RESTART THE FLOW
══════════════════════════════════════════════════════
• If the user raises a completely new problem, go back to RULE 2:
  ask one clarifying question first, then solve.
• Do NOT repeat or reference the previous solution.

══════════════════════════════════════════════════════
TONE
══════════════════════════════════════════════════════
• Stay calm and professional even if the user is impatient or rude.
• Be concise — do not pad responses with unnecessary explanation.
"""


# ============================================================
# INFERENCE ENGINE
# ============================================================

class CAGInferenceEngine:
    """
    CAG Inference Engine — Stateless Query Handler

    Key principle: This engine is STATELESS relative to conversations.
    It uses the pre-computed knowledge cache for every query.

    The separation between CacheManager (stateful) and InferenceEngine
    (stateless) is what makes this an "architecture" instead of a script.
    """

    def __init__(self, model, tokenizer, cache_manager, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_manager = cache_manager
        self.device = device
        self.config = config
        self.query_count = 0

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def generate(self, query: str, memory=None) -> Dict[str, Any]:
        """
        Generate an answer for a query using cached knowledge.

        Args:
            query:  User's question
            memory: ConversationMemory instance (optional but recommended).
                    When provided, the stage-aware system prompt and
                    conversation history are injected automatically.

        Returns:
            Dictionary with answer and metadata
        """
        if not self.cache_manager.is_initialized:
            raise ValueError("Cache not initialized. Run pre-loading phase first.")

        self.query_count += 1

        cache_state = self.cache_manager.cache_state
        full_prompt = self._build_query_prompt(cache_state, query, memory=memory)

        input_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_tokens + 200,
        ).input_ids.to(self.device)

        query_tokens = input_ids.shape[-1]
        self.cache_manager.handle_overflow(query_tokens)

        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        temperature=None,
                        top_p=None,
                    )

            answer = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )

            del input_ids, output_ids
            self._cleanup_memory()
            self.cache_manager.truncate_to_knowledge()

            return {
                'answer': answer.strip(),
                'query_number': self.query_count,
                'input_tokens': query_tokens,
                'success': True,
            }

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("❌ OOM during generation")
                self._cleanup_memory()
                self.cache_manager.truncate_to_knowledge()
                return {
                    'answer': "Error: Out of memory. Try a shorter query.",
                    'query_number': self.query_count,
                    'success': False,
                    'error': str(e),
                }
            raise

    def generate_streaming(self, query: str, memory=None):
        """
        Placeholder for streaming generation.
        The actual streaming path lives in CAGSystemFreshSession.stream_query().
        Kept here for interface completeness.
        """
        raise NotImplementedError(
            "Use CAGSystemFreshSession.stream_query() for streaming."
        )

    def batch_generate(self, queries: list, memory=None) -> list:
        """
        Generate answers for multiple queries in batch.

        All queries share the same knowledge cache, making this efficient.

        Args:
            queries: List of query strings
            memory:  Shared ConversationMemory instance (optional)

        Returns:
            List of result dictionaries
        """
        results = []
        for query in queries:
            result = self.generate(query, memory=memory)
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        cache_info = self.cache_manager.get_cache_info()
        return {
            'total_queries': self.query_count,
            'cache_initialized': cache_info['initialized'],
            'knowledge_tokens': cache_info.get('knowledge_tokens', 0),
            'max_new_tokens': self.config.max_new_tokens,
        }

    def reset_stats(self):
        """Reset query counter"""
        self.query_count = 0

    # ----------------------------------------------------------
    # Prompt building  (FIXED)
    # ----------------------------------------------------------

    def _build_query_prompt(self, cache_state, query: str,
                            memory=None) -> str:
        """
        Build the complete prompt for inference.

        Structure:
            <system>
              SYSTEM_PROMPT + stage instruction + knowledge base
            </system>
            [conversation history]
            <user>  current query  </user>
            <assistant>            ← model generates from here

        Args:
            cache_state: Pre-computed KV cache state
            query:       Current user message
            memory:      ConversationMemory instance (optional)
        """
        # Decode the knowledge base from the cached input_ids
        knowledge_text = self.tokenizer.decode(
            cache_state.input_ids[0],
            skip_special_tokens=True,   # clean text for the system block
        )

        # ── Stage-specific instruction ─────────────────────────
        stage_instruction = ""
        if memory is not None:
            stage_instruction = (
                "\n\n══ CURRENT TURN INSTRUCTION ══\n"
                + memory.get_stage_instruction()
            )

        # ── System block ───────────────────────────────────────
        system_block = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            + SYSTEM_PROMPT
            + stage_instruction
            + "\n\n══ KNOWLEDGE BASE ══\n"
            + knowledge_text
            + "<|eot_id|>\n"
        )

        # ── Conversation history ───────────────────────────────
        history_block = ""
        if memory is not None:
            history_block = memory.format_conversation_for_prompt()
            if history_block:
                history_block += "\n"

        # ── Current user turn ──────────────────────────────────
        # Personalise the query with the user's name if we have it
        if memory is not None and memory.user_profile.name:
            display_query = f"[{memory.user_profile.name}] {query}"
        else:
            display_query = query

        user_block = (
            "<|start_header_id|>user<|end_header_id|>\n"
            + display_query
            + "<|eot_id|>"
        )

        # ── Assistant header (model generates from here) ───────
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n"

        return system_block + history_block + user_block + assistant_header

    # ----------------------------------------------------------
    # Memory cleanup
    # ----------------------------------------------------------

    def _cleanup_memory(self):
        """Memory cleanup after generation"""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# ============================================================
# SESSION MANAGER (unchanged logic, updated generate calls)
# ============================================================

class CAGSessionManager:
    """
    Session Manager for multi-user CAG deployment.

    Production pattern:
      - One shared KnowledgeStore
      - One shared CacheManager (pre-computed cache)
      - Multiple InferenceEngines (one per user session)

    This lets thousands of users query the same knowledge base
    without re-encoding it for each user.
    """

    def __init__(self, model, tokenizer, cache_manager, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_manager = cache_manager
        self.device = device
        self.config = config
        self.sessions: Dict[str, CAGInferenceEngine] = {}

    def create_session(self, session_id: str) -> CAGInferenceEngine:
        """Create a new inference session"""
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        engine = CAGInferenceEngine(
            self.model,
            self.tokenizer,
            self.cache_manager,
            self.device,
            self.config,
        )
        self.sessions[session_id] = engine
        return engine

    def get_session(self, session_id: str) -> Optional[CAGInferenceEngine]:
        """Get existing session"""
        return self.sessions.get(session_id)

    def close_session(self, session_id: str):
        """Close and cleanup a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_active_sessions(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)