"""
CAG Architecture - System (Fresh Session + With Memory variants)

FIXES (v3):
- Added `system_prompt` attribute + `set_system_prompt()` method so the test
  harness's AGENT_SYSTEM_PROMPT is actually used instead of silently ignored.
- Added `reset_conversation()` — the test harness calls this between suites.
  Previously it fell through to `cag.memory.clear()` (doesn't exist), leaving
  Sarah's name leaking into every subsequent test.
- `_clarification_sent` flag: AWAITING_CLARIFICATION only advances to
  PROVIDING_SOLUTION after the agent has actually sent its clarifying question.
  Without this, the stage jumped too early, skipping clarification in Test 2.
- `_build_prompt_with_memory()` now uses `self.system_prompt` (injected) instead
  of the hard-coded constant, so the test's carefully crafted prompt is used.
"""

import os
import sys
import torch
import gc
from typing import Optional, Dict, Any, Generator

from cag_config import CAGConfig
from gpu import free_gpu_smart, force_gpu, get_gpu_memory_info, cleanup_gpu_memory
from model_loader import ModelLoader
from knowledge_store import KnowledgeStore
from cache_manager import CacheManager
from conversation_memory import ConversationMemory, ConversationStage


# ============================================================
# Default system prompt (used when none is injected externally)
# ============================================================

DEFAULT_SYSTEM_PROMPT = """\
You are a professional, friendly technical support agent for Ask Novation.
You MUST follow these rules in STRICT ORDER every conversation:

RULE 1 — NAME FIRST (non-negotiable):
If you do NOT yet know the user's name, your ONLY response is to ask for
their name. Do NOT help with any other request until you have their name.
Even if the user jumps straight to a problem, ask for their name first.
Once you have their name, use it naturally throughout the conversation.

RULE 2 — ONE CLARIFYING QUESTION BEFORE SOLVING:
When a user describes a problem, ask exactly ONE focused clarifying question.
Do NOT provide any solution in the same turn as your clarifying question.

RULE 3 — SOLVE AFTER CLARIFICATION:
Once the user gives specific details (error codes, file sizes, platform names,
numbers, endpoint names), give a direct actionable solution.
Do NOT ask further clarifying questions at this stage.

RULE 4 — FOCUSED FOLLOW-UPS:
If the user asks a specific follow-up question about the solution, answer ONLY
that question concisely. Do NOT re-explain the full solution.

RULE 5 — NEW PROBLEMS RESTART THE FLOW:
If the user raises a new problem, go back to RULE 2: ask one clarifying
question first, then solve. Do NOT repeat the previous solution.

TONE: Stay calm and professional even if the user is impatient or rude.
"""


# ============================================================
# Stage-transition heuristics
# ============================================================

def _looks_like_problem(text: str) -> bool:
    keywords = [
        "crash", "error", "fail", "down", "broken", "slow", "issue",
        "bug", "not work", "doesn't work", "can't", "cannot", "won't",
        "help", "problem", "trouble", "fix", "wrong", "timeout", "sync",
        "upload", "load", "connect", "access", "miss", "lost", "drop",
        "crashing", "failing", "timing out", "not loading",
    ]
    return any(kw in text.lower() for kw in keywords)


def _looks_like_new_problem(text: str) -> bool:
    phrases = [
        "second issue", "another issue", "new issue",
        "second problem", "another problem", "new problem",
        "also have", "also need", "something else",
        "different question", "one more thing", "now i also",
    ]
    return any(p in text.lower() for p in phrases)


# ============================================================
# CAGSystemFreshSession
# ============================================================

class CAGSystemFreshSession:
    """
    CAG System — Fresh Session Mode.

    Asks for name at the start of every session. Conversation memory lives
    only in RAM; nothing persists to disk (KV cache still saved).
    """

    def __init__(self, config: Optional[CAGConfig] = None):
        self.config = config or CAGConfig()
        self.config.enable_cache_persistence = True

        # ── Injected system prompt ──────────────────────────────────────
        # test.py's init_agent() sets this via attribute assignment or
        # set_system_prompt() BEFORE initialize() is called.
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT

        # Core components (populated during initialize())
        self.model_loader    = None
        self.knowledge_store = None
        self.cache_manager   = None

        # In-memory conversation — no disk persistence for fresh sessions
        self.memory = ConversationMemory(
            config=self.config,
            max_history=self.config.max_conversation_history,
        )
        self._disable_memory_persistence()

        # Stage helpers
        # Tracks whether the agent has actually sent the clarifying question
        # in the current problem cycle. Without this, the stage advances from
        # AWAITING_CLARIFICATION to PROVIDING_SOLUTION before the question
        # is even asked (Test 2 Step 3 bug).
        self._clarification_sent: bool = False

        # Runtime
        self.device           = None
        self.model            = None
        self.tokenizer        = None
        self.is_initialized   = False
        self.total_queries    = 0
        self.session_start_time = None

    # ----------------------------------------------------------
    # System prompt injection
    # ----------------------------------------------------------

    def set_system_prompt(self, prompt: str):
        """
        Inject a custom system prompt before initialize() is called.
        The test harness uses this so its AGENT_SYSTEM_PROMPT is active.
        """
        self.system_prompt = prompt

    # ----------------------------------------------------------
    # Memory persistence control
    # ----------------------------------------------------------

    def _disable_memory_persistence(self):
        """Override save/load so conversation never touches disk."""
        self.memory.save_memory = lambda: None
        self.memory.load_memory = lambda: None
        for path in (self.memory.conversation_file, self.memory.profile_file):
            if os.path.exists(path):
                os.remove(path)

    # ----------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------

    def initialize(self, force_cache_rebuild: bool = False):
        from datetime import datetime
        self.session_start_time = datetime.now()

        print("\n" + "=" * 70)
        print("🚀 CAG SYSTEM — FRESH SESSION MODE")
        print("=" * 70)

        self._initialize_gpu()
        self._load_model()
        self._load_knowledge()
        self._precompute_cache(force_rebuild=force_cache_rebuild)

        self.is_initialized = True
        print("\n✅ CAG SYSTEM READY")
        self._print_system_status()

    def _initialize_gpu(self):
        print("\n📊 PHASE 1: GPU INITIALIZATION")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.config.get_pytorch_alloc_config()
        freed = free_gpu_smart(min_mem_mb=self.config.min_free_memory_mb)
        print(f"✅ GPU cleanup: {freed} processes freed")
        gc.collect()
        torch.cuda.empty_cache()
        self.device = force_gpu()

    def _load_model(self):
        print("\n📦 PHASE 2: MODEL LOADING")
        self.model_loader = ModelLoader(self.config)
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(
            self.device
        )
        print("✅ Model and tokenizer loaded")

    def _load_knowledge(self):
        print("\n📚 PHASE 3: KNOWLEDGE BASE LOADING")
        self.knowledge_store = KnowledgeStore(self.tokenizer, self.config)
        entry_count = self.knowledge_store.load_from_sources()
        print(f"✅ Loaded {entry_count:,} knowledge entries")
        if self.config.verbose:
            self.knowledge_store.preview_entries(n=3)

    def _precompute_cache(self, force_rebuild: bool = False):
        print("\n🎯 PHASE 4: CACHE PRECOMPUTATION")
        knowledge_text = self.knowledge_store.build_knowledge_text(use_compact=True)
        self.cache_manager = CacheManager(
            self.model, self.tokenizer, self.device, self.config
        )
        cache_loaded = False
        if not force_rebuild and self.config.enable_cache_persistence:
            cache_loaded = self.cache_manager.load_cache()
        if not cache_loaded:
            self.cache_manager.precompute_cache(knowledge_text)
            if self.config.enable_cache_persistence:
                self.cache_manager.save_cache()
                self.knowledge_store.save_metadata()
        print("✅ Cache ready")

    # ----------------------------------------------------------
    # Core query path
    # ----------------------------------------------------------

    def query(self, query: str) -> Dict[str, Any]:
        """Process a query with stage-aware in-session memory."""
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        self.total_queries += 1

        # ── STAGE: AWAITING_NAME ──────────────────────────────
        if self.memory.stage == ConversationStage.AWAITING_NAME:
            name = self.memory.extract_name_from_response(query)
            self.memory.add_message('user', query)

            if name:
                self.memory.set_user_name(name)   # also advances stage → AWAITING_PROBLEM

                # FIX: if the user gave BOTH their name AND a problem in the
                # same message (e.g. "I'm Lina and my uploads crash with large files"),
                # don't waste a turn asking "What can I help you with?" — they already
                # told us.  Advance the stage to AWAITING_CLARIFICATION and let the
                # LLM generate the clarifying question immediately.
                if _looks_like_problem(query):
                    self.memory.advance_stage(ConversationStage.AWAITING_CLARIFICATION)
                    self._clarification_sent = False
                    # Fall through to the LLM block below
                else:
                    # Pure name-only message — just greet and wait for the problem
                    response = f"Nice to meet you, {name}! What can I help you with today?"
                    self.memory.add_message('assistant', response)
                    return self._ok(response)
            else:
                response = self._ask_for_name_message()
                self.memory.add_message('assistant', response)
                return self._ok(response, waiting_for_name=True)

        # ── All other stages: LLM generation ──────────────────
        try:
            self._advance_stage_on_user_input(query)
            self.memory.add_message('user', query)

            full_prompt    = self._build_prompt_with_memory(query)
            inputs         = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_context_tokens,
            )
            input_ids      = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            query_tokens   = input_ids.shape[-1]

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                    )

            answer = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            self._advance_stage_after_response()
            self.memory.add_message('assistant', answer)

            del input_ids, output_ids, attention_mask, inputs
            self._aggressive_cleanup()
            self.cache_manager.truncate_to_knowledge()

            return {
                'answer': answer,
                'query_number': self.total_queries,
                'input_tokens': query_tokens,
                'success': True,
                'user_name': self.memory.user_profile.name,
            }

        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'query_number': self.total_queries,
                'success': False,
                'error': str(e),
            }

    def stream_query(self, query: str) -> Generator[str, None, None]:
        """Stream response token-by-token with stage-aware memory."""
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        self.total_queries += 1

        # ── STAGE: AWAITING_NAME ──────────────────────────────
        if self.memory.stage == ConversationStage.AWAITING_NAME:
            name = self.memory.extract_name_from_response(query)
            self.memory.add_message('user', query)

            if name:
                self.memory.set_user_name(name)

                # FIX: name + problem in one message → skip the "What can I help
                # you with?" turn and go straight to clarification via the LLM.
                if _looks_like_problem(query):
                    self.memory.advance_stage(ConversationStage.AWAITING_CLARIFICATION)
                    self._clarification_sent = False
                    # Fall through to the LLM streaming block below
                else:
                    response = f"Nice to meet you, {name}! What can I help you with today?"
                    self.memory.add_message('assistant', response)
                    for word in response.split():
                        yield word + " "
                    return
            else:
                response = self._ask_for_name_message()
                self.memory.add_message('assistant', response)
                for word in response.split():
                    yield word + " "
                return

        # ── All other stages: stream from LLM ─────────────────
        try:
            self._advance_stage_on_user_input(query)
            self.memory.add_message('user', query)

            full_prompt    = self._build_prompt_with_memory(query)
            inputs         = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_context_tokens,
            )
            input_ids      = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=30.0,
            )

            gen_kwargs = {
                "input_ids":          input_ids,
                "attention_mask":     attention_mask,
                "max_new_tokens":     self.config.max_new_tokens,
                "streamer":           streamer,
                "do_sample":          False,
                "pad_token_id":       self.tokenizer.eos_token_id,
                "eos_token_id":       self.tokenizer.eos_token_id,
                "use_cache":          True,
                "num_beams":          1,
                "repetition_penalty": 1.0,
                "length_penalty":     1.0,
            }

            thread = Thread(target=self._generate_thread, kwargs=gen_kwargs)
            thread.start()

            response_text = ""
            try:
                for new_text in streamer:
                    if new_text:
                        response_text += new_text
                        yield new_text
            except Exception as e:
                print(f"\n❌ Streaming error: {e}")
            finally:
                thread.join(timeout=5.0)
                if response_text:
                    self._advance_stage_after_response()
                    self.memory.add_message('assistant', response_text.strip())
                del input_ids, attention_mask, inputs
                self._aggressive_cleanup()
                self.cache_manager.truncate_to_knowledge()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._aggressive_cleanup()
                self.cache_manager.truncate_to_knowledge()
                yield "\n[Error: GPU out of memory. Please try again.]"
            else:
                yield f"\n[Error: {str(e)}]"
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    # ----------------------------------------------------------
    # Stage transition logic  (FIXED)
    # ----------------------------------------------------------

    def _advance_stage_on_user_input(self, query: str):
        """
        Advance the conversation stage based on the user's message,
        BEFORE the LLM generates a response.

        Critical fix — the _clarification_sent gate:
        Previously the stage went AWAITING_CLARIFICATION → PROVIDING_SOLUTION
        the instant the user said ANYTHING while in AWAITING_CLARIFICATION.
        This meant if the user said "My website has been down for 2 hours"
        the stage jumped to PROVIDING_SOLUTION before the agent even asked its
        clarifying question, so the agent gave a solution instead of clarifying.

        The fix: only advance to PROVIDING_SOLUTION when _clarification_sent
        is True, meaning the agent has already sent the clarifying question
        in a previous turn.
        """
        stage = self.memory.stage

        if stage == ConversationStage.AWAITING_PROBLEM:
            if _looks_like_problem(query):
                self.memory.advance_stage(ConversationStage.AWAITING_CLARIFICATION)
                self._clarification_sent = False  # reset for new problem cycle

        elif stage == ConversationStage.AWAITING_CLARIFICATION:
            if self._clarification_sent:
                # Agent already asked; user is now answering → solve next
                self.memory.advance_stage(ConversationStage.PROVIDING_SOLUTION)
            # If clarification NOT yet sent, stay in AWAITING_CLARIFICATION
            # so the stage instruction tells the model to ask, not solve.

        elif stage in (ConversationStage.PROVIDING_SOLUTION,
                       ConversationStage.FOLLOWUP):
            if _looks_like_new_problem(query):
                self.memory.advance_stage(ConversationStage.AWAITING_CLARIFICATION)
                self._clarification_sent = False
            else:
                self.memory.advance_stage(ConversationStage.FOLLOWUP)

    def _advance_stage_after_response(self):
        """
        Advance stage AFTER the assistant has responded.

        - In AWAITING_CLARIFICATION: agent just sent its question → mark sent.
        - In PROVIDING_SOLUTION: agent just gave the solution → move to FOLLOWUP.
        """
        if self.memory.stage == ConversationStage.AWAITING_CLARIFICATION:
            self._clarification_sent = True

        elif self.memory.stage == ConversationStage.PROVIDING_SOLUTION:
            self.memory.advance_stage(ConversationStage.FOLLOWUP)

    # ----------------------------------------------------------
    # Prompt building
    # ----------------------------------------------------------

    def _build_prompt_with_memory(self, query: str) -> str:
        """
        Build complete prompt:
          1. self.system_prompt  (injected by test harness or DEFAULT_SYSTEM_PROMPT)
          2. Stage-specific instruction from memory
          3. Knowledge base (decoded from KV cache)
          4. Conversation history
          5. Current user query (personalised with name)
          6. Assistant header
        """
        cache_state    = self.cache_manager.cache_state
        knowledge_text = self.tokenizer.decode(
            cache_state.input_ids[0],
            skip_special_tokens=True,
        )

        stage_instruction = (
            "\n\n══ CURRENT TURN INSTRUCTION ══\n"
            + self.memory.get_stage_instruction()
        )

        system_block = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            + self.system_prompt          # ← uses injected prompt (not hard-coded)
            + stage_instruction
            + "\n\n══ KNOWLEDGE BASE ══\n"
            + knowledge_text
            + "<|eot_id|>\n"
        )

        history_block = self.memory.format_conversation_for_prompt()
        if history_block:
            history_block += "\n"

        user_name     = self.memory.user_profile.name
        display_query = f"[{user_name}] {query}" if user_name else query

        user_block = (
            "<|start_header_id|>user<|end_header_id|>\n"
            + display_query
            + "<|eot_id|>"
        )

        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n"

        return system_block + history_block + user_block + assistant_header

    # ----------------------------------------------------------
    # Session control  (FIXED — test harness compatibility)
    # ----------------------------------------------------------

    def reset_conversation(self):
        """
        Reset conversation and stage machine between test suites.

        The test harness calls:
            if hasattr(cag, 'reset_conversation'): cag.reset_conversation()

        Previously this method didn't exist, so the harness fell through to
        `cag.memory.clear()` which also doesn't exist, so nothing was reset
        and 'Sarah' from Test 1 leaked into every subsequent test.
        """
        if self.cache_manager:
            self.cache_manager.truncate_to_knowledge()
        self.memory.reset_all()
        self._clarification_sent = False
        self.total_queries = 0
        self._aggressive_cleanup()

    def reset_session(self):
        """Alias for reset_conversation() — backward compatibility."""
        self.reset_conversation()

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _ask_for_name_message(self) -> str:
        messages = [
            "Hello! Before we start, could I get your name?",
            "I'd love to help — could you first tell me your name?",
            "I really want to assist you well. What's your name?",
            "Please share your name so I can address you properly.",
        ]
        return messages[min(self.total_queries - 1, len(messages) - 1)]

    def _ok(self, answer: str, **extra) -> Dict[str, Any]:
        return {'answer': answer, 'query_number': self.total_queries,
                'success': True, **extra}

    # ----------------------------------------------------------
    # Stats / debug
    # ----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        if not self.is_initialized:
            return {'initialized': False}
        return {
            'initialized':  self.is_initialized,
            'total_queries': self.total_queries,
            'knowledge': {
                'entries': self.knowledge_store.get_entry_count(),
                'tokens':  self.knowledge_store.get_token_count(),
            },
            'cache':  self.cache_manager.get_cache_info(),
            'config': {
                'max_context_tokens': self.config.max_context_tokens,
                'max_new_tokens':     self.config.max_new_tokens,
            },
            'gpu_memory':    get_gpu_memory_info(),
            'session_mode':  'fresh_session_no_persistence',
            'memory':        self.memory.get_stats(),
            'session_start': (
                self.session_start_time.isoformat()
                if self.session_start_time else None
            ),
        }

    def generate_session_summary(self) -> Dict[str, Any]:
        """
        After the conversation ends, ask the LLM to extract:
          1. The user's name (as it understood it from the dialogue)
          2. A concise summary of the whole session

        Returns a dict:
            {
                'user_name':  str | None,   # name from memory (fast path)
                'llm_name':   str | None,   # name as LLM recalls it
                'summary':    str,          # LLM-generated session summary
            }
        """
        if not self.is_initialized or not self.memory.messages:
            return {'user_name': None, 'llm_name': None, 'summary': 'No conversation to summarise.'}

        # Build a plain-text transcript for the summary prompt
        transcript_lines = []
        for msg in self.memory.messages:
            role = "User" if msg.role == "user" else "Assistant"
            transcript_lines.append(f"{role}: {msg.content}")
        transcript = "\n".join(transcript_lines)

        summary_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a precise conversation analyst. "
            "Read the support transcript below and respond with EXACTLY two lines — no more.\n"
            "Line 1 must be:  Name: <the user's first name, or 'Unknown' if never given>\n"
            "Line 2 must be:  Summary: <one concise sentence describing the user's issue and outcome>\n"
            "Do NOT add any other text, greetings, or explanation."
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"TRANSCRIPT:\n{transcript}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        try:
            inputs = self.tokenizer(
                summary_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_tokens,
            )
            input_ids      = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=120,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                    )

            raw = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            del input_ids, output_ids, attention_mask, inputs
            self._aggressive_cleanup()

            # Parse the two-line response
            llm_name = None
            summary  = raw
            for line in raw.splitlines():
                line = line.strip()
                if line.lower().startswith("name:"):
                    candidate = line[5:].strip()
                    if candidate.lower() not in {"unknown", "n/a", "none", ""}:
                        llm_name = candidate
                elif line.lower().startswith("summary:"):
                    summary = line[8:].strip()

            return {
                'user_name': self.memory.user_profile.name,   # from memory (reliable)
                'llm_name':  llm_name,                        # from LLM recall
                'summary':   summary,
            }

        except Exception as e:
            return {
                'user_name': self.memory.user_profile.name,
                'llm_name':  None,
                'summary':   f'[Summary generation failed: {e}]',
            }

    def cleanup(self):
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self._aggressive_cleanup()
        print("\n🧹 Cleaned up CAG system")

    def print_cache_content(self):
        if not self.cache_manager or not self.cache_manager.cache_state:
            print("❌ Cache is empty or not initialized")
            return
        ids  = self.cache_manager.cache_state.input_ids[0]
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        print(text)
        print(f"📏 Total Tokens: {len(ids)}")

    def _generate_thread(self, **kw):
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    self.model.generate(**kw)
        except Exception as e:
            print(f"\n❌ Generation thread error: {e}")

    def _aggressive_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _print_system_status(self):
        stats = self.get_stats()
        print(f"\n📊 Knowledge entries: {stats['knowledge']['entries']:,}")
        print(f"   Cache initialized: {stats['cache']['initialized']}")
        print(f"   Stage:             {stats['memory']['stage']}")


# ============================================================
# CAGSystemWithMemory — persistent profile variant
# ============================================================

class CAGSystemWithMemory(CAGSystemFreshSession):
    """
    CAG System with persistent conversation memory.

    Same as CAGSystemFreshSession except the user profile and conversation
    history are saved to disk so the name is remembered across sessions.
    """

    def __init__(self, config: Optional[CAGConfig] = None):
        super().__init__(config)

        # Re-create memory WITH disk persistence (undo the fresh-session disable)
        self.memory = ConversationMemory(
            config=self.config,
            max_history=self.config.max_conversation_history,
        )
        # Note: do NOT call _disable_memory_persistence() here

    def reset_all(self):
        """Wipe everything including the saved user profile."""
        self.memory.reset_all()
        self._clarification_sent = False
        if self.cache_manager:
            self.cache_manager.truncate_to_knowledge()
        self._aggressive_cleanup()
        print("🗑️ All memory cleared (including user profile)")