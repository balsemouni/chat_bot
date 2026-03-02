"""
CAG Architecture - Cache Manager Module for Solution Recommendations
Manages KV cache lifecycle with truncation and persistence
"""

import torch
import gc
import os
import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict


@dataclass
class CacheState:
    """Represents the state of KV cache"""
    input_ids: torch.Tensor
    token_count: int
    knowledge_token_count: int  # The N_N tokens we preserve
    past_key_values: Optional[Any] = None  # Actual KV cache tensors stored here
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding tensors)"""
        return {
            'token_count': self.token_count,
            'knowledge_token_count': self.knowledge_token_count,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }


class CacheManager:
    """
    Cache Manager - Core of CAG Architecture for Solution Recommendations
    
    Responsibilities:
    1. Pre-compute KV cache from solution knowledge base (happens ONCE)
    2. Truncate cache after each query (reset to N_N tokens)
    3. Persist and load cache from disk
    4. Handle cache overflow according to policy
    """
    
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        self.cache_state: Optional[CacheState] = None
        self.is_initialized = False
        
    def precompute_cache(self, knowledge_text: str) -> CacheState:
        """
        Pre-compute KV cache from solution knowledge base
        
        Args:
            knowledge_text: Formatted solution knowledge (PROBLEM:|SOLUTION: format)
        """
        print("\n" + "="*60)
        print("🎯 PRECOMPUTING KV CACHE (Solution Recommendation System)")
        print("="*60)
        
        self._cleanup_memory()
        free_before = torch.cuda.mem_get_info()[0] // 1024**2
        print(f"📊 Free memory before cache: {free_before}MB")
        
        # Build system prompt with solution knowledge
        prompt = self._build_cache_prompt(knowledge_text)
        
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_tokens
        ).input_ids.to(self.device)
        
        token_count = input_ids.shape[-1]
        print(f"📝 Solution knowledge base tokens: {token_count}")
        
        if token_count > self.config.max_context_tokens:
            print(f"⚠️  Truncating to {self.config.max_context_tokens} tokens")
            input_ids = input_ids[:, :self.config.max_context_tokens]
            token_count = input_ids.shape[-1]
        
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids,
                        use_cache=True,
                        return_dict=True,
                    )
            
            # Handle newer Transformers DynamicCache objects
            past_key_values = outputs.past_key_values
            if hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = past_key_values.to_legacy_cache()

            cache_state = CacheState(
                input_ids=input_ids,
                token_count=token_count,
                knowledge_token_count=token_count,
                past_key_values=past_key_values,
                timestamp=None,
                metadata={'source': 'solution_knowledge_base', 'type': 'recommendations'}
            )
            
            del outputs
            self._cleanup_memory()
            
            free_after = torch.cuda.mem_get_info()[0] // 1024**2
            memory_used = free_before - free_after
            
            print(f"✅ KV Cache pre-computed successfully")
            print(f"   Cache length: {token_count} tokens")
            print(f"   Memory used: ~{memory_used}MB")
            
            self.cache_state = cache_state
            self.is_initialized = True
            
            return cache_state
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ OUT OF MEMORY with {token_count} tokens!")
                raise
            raise
    
    def truncate_to_knowledge(self):
        """
        Truncate KV cache back to knowledge-base length (N_N tokens).
        This resets the cache to only contain solution knowledge, removing query/response.
        """
        if not self.is_initialized or self.cache_state is None:
            raise ValueError("Cache not initialized.")

        N = self.cache_state.knowledge_token_count

        # Truncate input_ids
        if self.cache_state.input_ids.shape[-1] > N:
            self.cache_state.input_ids = self.cache_state.input_ids[:, :N]

        # Truncate past_key_values (handles any tuple structure)
        if self.cache_state.past_key_values is not None:
            new_pkv = []
            for layer_idx, layer_pair in enumerate(self.cache_state.past_key_values):
                # Process each tensor in the layer tuple
                truncated_layer = []
                for item in layer_pair:
                    if isinstance(item, torch.Tensor) and item.dim() >= 3:
                        # Standard KV tensors are [batch, heads, seq_len, head_dim]
                        # Slice the sequence dimension (usually dim 2)
                        truncated_layer.append(item[:, :, :N, :])
                    else:
                        # Pass through non-tensor items unmodified
                        truncated_layer.append(item)
                new_pkv.append(tuple(truncated_layer))
            
            self.cache_state.past_key_values = tuple(new_pkv)

        self.cache_state.token_count = N
        # NOTE: No empty_cache() here — truncation runs after every query and
        # GPU memory is NOT actually freed by slicing tensors. Calling
        # empty_cache() would force a costly driver round-trip for zero gain.
    
    def handle_overflow(self, query_tokens: int) -> bool:
        """Handle cache overflow according to policy"""
        if self.cache_state is None:
            return False
        
        total_tokens = self.cache_state.knowledge_token_count + query_tokens
        available_tokens = self.config.max_context_tokens + self.config.max_new_tokens
        
        if total_tokens <= available_tokens:
            return True
        
        if self.config.cache_overflow_policy == "error":
            raise ValueError(f"Query overflow: {total_tokens} > {available_tokens}")
        elif self.config.cache_overflow_policy == "truncate":
            return True
        return False
    
    def save_cache(self, path: Optional[str] = None):
        """
        Save cache to disk for persistence
        """
        if self.cache_state is None:
            raise ValueError("No cache to save")
        
        if path is None:
            path = self.config.cache_file_path
        
        # Convert past_key_values to CPU (handles any structure)
        pkv_cpu = None
        if self.cache_state.past_key_values is not None:
            pkv_cpu = tuple(
                tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in layer_pair)
                for layer_pair in self.cache_state.past_key_values
            )
        
        cache_data = {
            'input_ids': self.cache_state.input_ids.cpu(),
            'past_key_values': pkv_cpu,
            'metadata': self.cache_state.to_dict()
        }
        
        torch.save(cache_data, path)
        
        if self.config.verbose:
            print(f"💾 Cache saved to {path}")
    
    def load_cache(self, path: Optional[str] = None) -> bool:
        """
        Load cache from disk
        """
        if path is None:
            path = self.config.cache_file_path
        
        if not os.path.exists(path):
            return False
        
        try:
            cache_data = torch.load(path, map_location=self.device, weights_only=False)
            metadata = cache_data['metadata']

            pkv_raw = cache_data.get('past_key_values')
            if pkv_raw is None:
                print("⚠️  Cache is stale (no past_key_values). Rebuilding...")
                return False

            # Restore past_key_values to GPU (handles any structure)
            past_key_values = tuple(
                tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in layer_pair)
                for layer_pair in pkv_raw
            )
            
            self.cache_state = CacheState(
                input_ids=cache_data['input_ids'].to(self.device),
                token_count=metadata['token_count'],
                knowledge_token_count=metadata['knowledge_token_count'],
                past_key_values=past_key_values,
                timestamp=metadata.get('timestamp'),
                metadata=metadata.get('metadata')
            )
            
            self.is_initialized = True
            
            if self.config.verbose:
                print(f"✅ Cache loaded from {path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state"""
        if self.cache_state is None:
            return {'initialized': False}
        return {
            'initialized': self.is_initialized,
            'token_count': self.cache_state.token_count,
            'knowledge_tokens': self.cache_state.knowledge_token_count,
            'metadata': self.cache_state.metadata
        }
    
    def _build_cache_prompt(self, knowledge_text: str) -> str:
        """
        Build the solution recommendation system prompt.

        FIX: The prompt now ends after the system turn <|eot_id|>.
        It does NOT open the user turn here — that is handled by
        _build_fresh_prompt() in cag_system.py so the model sees a
        clean, properly closed turn boundary before generating.
            """
        prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are the AI receptionist for 'Ask Novation', a business solutions company.\n"
        "You are warm, professional, and conversational. You NEVER sound robotic.\n"
        "You NEVER invent or assume names.\n"
        "You NEVER say \"Hi John\" unless the user told you their name is John.\n"
        "\n"
        "=== YOUR CONVERSATION HAS 4 PHASES. NEVER SKIP A PHASE. ===\n"
        "\n"
        "PHASE 1 — GREETING\n"
        "(Triggers: Hi, Hello, Hey, Hey there)\n"
        "→ Welcome the user to Ask Novation.\n"
        "→ Ask ONLY for their name.\n"
        "→ Do NOT ask about their problem yet.\n"
        "→ Do NOT greet them with any name. You do not know their name yet.\n"
        "→ CORRECT: \"Welcome to Ask Novation! Could I get your name?\"\n"
        "→ WRONG: \"Hi John! How can I help?\" ← NEVER do this. You don't know their name.\n"
        "\n"
        "PHASE 2 — NAME RECEIVED\n"
        "→ Greet them by name (e.g. \"Hi Sarah!\").\n"
        "→ Ask: \"What can I help you with today?\"\n"
        "→ If the user gives their name AND a problem in one message: greet by name, then continue to PHASE 3.\n"
        "\n"
        "PHASE 3 — CLARIFICATION\n"
        "(Triggers: user describes any problem)\n"
        "→ DO NOT give a solution yet.\n"
        "→ Ask exactly ONE specific clarifying question.\n"
        "→ Good examples:\n"
        "   • \"Does this happen with all file sizes, or only large ones?\"\n"
        "   • \"Is there an error message or error code you can see?\"\n"
        "   • \"Which browser or operating system are you using?\"\n"
        "   • \"How long has this been happening?\"\n"
        "   • \"Which specific endpoint or feature is affected?\"\n"
        "\n"
        "PHASE 4 — SOLUTION\n"
        "(Triggers: user answers your clarifying question with specific details)\n"
        "→ NOW give a focused, relevant solution based on everything they told you.\n"
        "→ Do NOT ask another clarifying question.\n"
        "→ Do NOT repeat a solution already given in this conversation.\n"
        "→ Only mention pricing if the user asks.\n"
        "\n"
        "=== EXAMPLE CONVERSATIONS ===\n"
        "\n"
        "--- EXAMPLE A ---\n"
        "User: Hi!\n"
        "You: Welcome to Ask Novation! I'm happy to help. Could I start by getting your name?\n"
        "\n"
        "User: I'm Sarah.\n"
        "You: Hi Sarah! What can I help you with today?\n"
        "\n"
        "User: My software crashes when I upload files.\n"
        "You: I'm sorry to hear that, Sarah. To help narrow this down — does the crash happen with all files, or only when the files are large?\n"
        "\n"
        "User: Only with large video files, around 1GB.\n"
        "You: That makes sense. For large video uploads like that, the issue is usually a server timeout or memory limit. I'd recommend chunked upload handling — this breaks the file into smaller pieces so the server never processes the full 1GB at once. Would you like me to walk you through the setup?\n"
        "\n"
        "--- EXAMPLE B ---\n"
        "User: My website is down!\n"
        "You: Oh no, let's get that sorted. First, could I get your name so I can assist you properly?\n"
        "\n"
        "User: Ahmed.\n"
        "You: Hi Ahmed! Website downtime is serious. Is there an error code showing, like a 503 or 404?\n"
        "\n"
        "User: It shows a 503 error and I'm on shared hosting.\n"
        "You: Got it. A 503 on shared hosting usually means your site hit the server's resource limit. The quickest fix is to contact your host to check server logs and restart your PHP process. Long-term, a VPS plan would give you control over resource limits. Want me to walk you through what to tell your hosting provider?\n"
        "\n"
        "--- EXAMPLE C (name + problem in one message) ---\n"
        "User: Hi, I'm Lina and my dashboard loads really slowly.\n"
        "You: Hi Lina! How slow are we talking — and does it happen on all browsers or just one?\n"
        "\n"
        "User: About 15 seconds, on all browsers.\n"
        "You: 15 seconds across all browsers tells me the slowdown is server-side, not a browser issue. The most common causes are unoptimized database queries or too many uncompressed assets. I'd start by enabling server-side caching and running a query profiler. Would you like a step-by-step plan?\n"
        "\n"
        "--- EXAMPLE D (name hidden in sentence) ---\n"
        "User: Hey there!\n"
        "You: Welcome to Ask Novation! Could I get your name before we get started?\n"
        "\n"
        "User: People usually call me Tom, and I need help with my CRM system.\n"
        "You: Hi Tom! What's specifically going on with your CRM — are you seeing errors, sync issues, or something else?\n"
        "\n"
        "User: It's not syncing contacts with our email platform.\n"
        "You: To help narrow it down — which CRM and email platform are you using?\n"
        "\n"
        "User: We use Salesforce and Mailchimp.\n"
        "You: Got it. Salesforce-Mailchimp sync failures are almost always a broken API key or contact field mapping mismatch. Check your Mailchimp integration settings in Salesforce and verify the API key is still active. If that's fine, re-map the contact fields manually and trigger a manual sync. Want me to walk through each step?\n"
        "\n"
        "User: How do I trigger the manual sync?\n"
        "You: In Salesforce, go to the Mailchimp integration settings, find the \"Sync Now\" or \"Manual Sync\" button — usually under the \"Audience\" or \"Contacts\" tab. If your integration uses a middleware like Zapier, trigger the Zap manually from the Zapier dashboard.\n"
        "\n"
        "--- EXAMPLE E: follow-up question after solution ---\n"
        "User: Thanks. How long does chunked upload setup take?\n"
        "You: Typically 1–3 days for a developer depending on your stack. With AWS S3 multipart upload or Cloudflare's built-in tools, it can be a few hours.\n"
        "\n"
        "=== CRITICAL RULE — WHEN TO STOP CLARIFYING AND SOLVE ===\n"
        "When the user gives SPECIFIC details (product names, error codes, file sizes, platform names, numbers), that IS their answer to your clarifying question.\n"
        "Do NOT ask another question.\n"
        "Give the solution immediately.\n"
        "\n"
        "Examples of SPECIFIC details that mean: SOLVE NOW:\n"
        "• \"We use Salesforce and Mailchimp\" → Solve for Salesforce + Mailchimp\n"
        "• \"503 error on shared hosting\" → Solve for 503 / shared hosting\n"
        "• \"1GB video files, happens every time\" → Solve for large video uploads\n"
        "• \"30 seconds on /orders endpoint\" → Solve for API timeout on that endpoint\n"
        "\n"
        "=== ABSOLUTE RULES ===\n"
        "✗ NEVER greet with a name you were not told.\n"
        "✗ NEVER invent or assume a name.\n"
        "✗ NEVER give a solution before asking at least one clarifying question.\n"
        "✗ NEVER ask another question after the user gives specific details — give the solution.\n"
        "✗ NEVER repeat the same solution twice in one conversation.\n"
        "✗ NEVER ask two questions at once.\n"
        "✗ NEVER re-explain the full solution when the user only asks a specific follow-up question.\n"
        "\n"
        "<|eot_id|>"
        )

        return prompt




    
    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()