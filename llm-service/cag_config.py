"""
CAG Architecture Configuration Module
OPTIMIZED FOR ZERO LATENCY + COMPLETE RESPONSES
Fixed version - prevents mid-sentence cutoffs
"""

import os 
base_dir = os.path.dirname(__file__)
from dataclasses import dataclass
from typing import Optional


@dataclass
class CAGConfig:
    """
    Configuration for CAG Architecture
    
    OPTIMIZED FOR:
    - Zero-latency streaming responses
    - Complete responses (no mid-sentence cutoffs)
    - Loading some entries 
    - Using Llama 3.2's context window efficiently
    - 4-bit quantization on RTX 4050 
    
    KEY FIXES:
    - max_new_tokens increased to 512 (prevents cutoffs and truncated recommendations)
    - max_context_tokens optimized for stability
    - Better memory management settings
    """
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    model_id: str = "unsloth/Llama-3.2-3B-Instruct"
    model_max_tokens: int = 128000  # Llama 3.2 full context window
    
    # ========================================================================
    # MEMORY CONFIGURATION - OPTIMIZED FOR STABILITY + ZERO LATENCY
    # ========================================================================
    
    # CRITICAL FIX: Use a sensible default that matches the "safe" preset.
    # The previous value of 500 is far too small for any real use and is
    # inconsistent with every preset (all start at 4000+).
    max_context_tokens: int = 4096    #
    
    # CRITICAL FIX: Increased from 100 to 256 tokens
    # This prevents mid-sentence cutoffs like "Can you please tell me what ❌ Error"
    # 256 tokens = ~190 words = complete responses
    max_new_tokens: int = 512  
    
    # GPU Memory Management
    gpu_memory_fraction: float = 0.85  # Use 95% of GPU memory
    min_free_memory_mb: int = 100       # Keep 100MB free for safety
    
    # ========================================================================
    # CACHE CONFIGURATION
    # ========================================================================
    cache_file_path: str = "commercial_kv_cache_7500.pt"
    cache_metadata_path: str = "cache_metadata_7500.json"
    enable_cache_persistence: bool = True  # Save/load cache from disk
    
    # ========================================================================
    # KNOWLEDGE BASE CONFIGURATION
    # ========================================================================
    
    # JSONL file path
    knowledge_jsonl_path: str = os.path.join(base_dir, ".\\data\\cache_metadata.json")
    
    # Allow all entries to be considered (will be limited by tokens)
    max_knowledge_entries: int = 50000
    
    # ========================================================================
    # QUANTIZATION CONFIGURATION (4-bit - Essential for 6GB GPU!)
    # ========================================================================
    use_4bit: bool = True                # Enable 4-bit quantization
    quant_type: str = "nf4"              # NF4 quantization (best quality)
    use_double_quant: bool = True        # Double quantization (saves more memory)
    compute_dtype: str = "float16"       # FP16 for computations
    
    # ========================================================================
    # OPTIMIZATION CONFIGURATION - FOR ZERO LATENCY
    # ========================================================================
    enable_tf32: bool = True                    # TF32 for faster matmul on Ampere+
    enable_gradient_checkpointing: bool = True  # Save memory during training
    use_flash_attention: bool = False           # Disabled for compatibility
    
    # Streaming optimization
    streaming_buffer_size: int = 1  # Minimum buffer for lowest latency
    
    # ========================================================================
    # CACHE POLICY CONFIGURATION
    # ========================================================================
    cache_overflow_policy: str = "truncate"  # Options: "truncate", "error", "compress"
    
    # Buffer tokens reserved for queries and generation
    cache_truncation_buffer: int = 50  # Keep 50 tokens free
    
    # ========================================================================
    # GENERATION PARAMETERS - FOR PERFECT RESPONSES
    # ========================================================================
    
    # Temperature and sampling (set to None for greedy decoding)
    temperature: Optional[float] = None  # None = greedy (deterministic)
    top_p: Optional[float] = None        # None = no nucleus sampling
    top_k: Optional[int] = None          # None = no top-k filtering
    
    # Penalties for quality responses
    repetition_penalty: float = 1.0      # 1.0 = no penalty
    length_penalty: float = 1.0          # 1.0 = no penalty
    no_repeat_ngram_size: int = 0        # 0 = allow natural repetition
    
    # Beam search (1 = greedy, faster)
    num_beams: int = 1  # Greedy search for speed
    
    # Early stopping
    early_stopping: bool = False  # Don't stop until max_new_tokens or EOS
    
    # ========================================================================
    # SYSTEM CONFIGURATION
    # ========================================================================
    cuda_device: int = 0       # GPU device ID
    verbose: bool = True       # Print detailed logs
    debug_mode: bool = False   # Extra debug output
    
    # ========================================================================
    # CONVERSATION MEMORY CONFIGURATION
    # ========================================================================
    max_conversation_history: int = 10  # Keep last 10 message pairs
    enable_conversation_memory: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        
        # Check total tokens don't exceed model capacity
        if self.max_context_tokens + self.max_new_tokens > self.model_max_tokens:
            raise ValueError(
                f"max_context_tokens ({self.max_context_tokens}) + "
                f"max_new_tokens ({self.max_new_tokens}) = "
                f"{self.max_context_tokens + self.max_new_tokens} exceeds "
                f"model_max_tokens ({self.model_max_tokens})"
            )
        
        # Validate cache overflow policy
        valid_policies = ["truncate", "error", "compress"]
        if self.cache_overflow_policy not in valid_policies:
            raise ValueError(
                f"Invalid cache_overflow_policy: {self.cache_overflow_policy}. "
                f"Must be one of {valid_policies}"
            )
        
        # Warn if max_new_tokens is too small
        if self.max_new_tokens < 150:
            print(f"⚠️  WARNING: max_new_tokens ({self.max_new_tokens}) is low!")
            print(f"   This may cause incomplete responses.")
            print(f"   Recommended: 256+ for complete answers")
        
        # Warn if using too much context
        if self.max_context_tokens > 8000:
            print(f"⚠️  WARNING: max_context_tokens ({self.max_context_tokens}) is very high!")
            print(f"   This may cause OOM errors on 6GB GPU")
            print(f"   Recommended: 7500 or lower for 6GB GPU with Llama 3.2-3B")
    
    @classmethod
    def from_env(cls) -> 'CAGConfig':
        """Load configuration from environment variables"""
        return cls(
            model_id=os.getenv("CAG_MODEL_ID", cls.model_id),
            max_context_tokens=int(os.getenv("CAG_MAX_CONTEXT_TOKENS", cls.max_context_tokens)),
            max_new_tokens=int(os.getenv("CAG_MAX_NEW_TOKENS", cls.max_new_tokens)),
            cache_file_path=os.getenv("CAG_CACHE_FILE", cls.cache_file_path),
            verbose=os.getenv("CAG_VERBOSE", "true").lower() == "true",
            debug_mode=os.getenv("CAG_DEBUG", "false").lower() == "true",
        )
    
    def get_pytorch_alloc_config(self) -> str:
        """Get PyTorch CUDA allocation configuration"""
        return 'expandable_segments:True'
    
    def get_bnb_config_dict(self):
        """Get BitsAndBytes configuration as dictionary"""
        return {
            "load_in_4bit": self.use_4bit,
            "bnb_4bit_quant_type": self.quant_type,
            "bnb_4bit_use_double_quant": self.use_double_quant,
            "bnb_4bit_compute_dtype": self.compute_dtype,
        }
    
    def get_generation_config_dict(self):
        """Get generation configuration as dictionary"""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "early_stopping": self.early_stopping,
            "do_sample": self.temperature is not None,  # Auto-detect
        }
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("📋 CAG CONFIGURATION SUMMARY")
        print("="*70)
        
        print(f"\n🤖 MODEL:")
        print(f"   ID: {self.model_id}")
        print(f"   Max tokens: {self.model_max_tokens:,}")
        
        print(f"\n🧠 MEMORY:")
        print(f"   Context tokens: {self.max_context_tokens:,}")
        print(f"   Generation tokens: {self.max_new_tokens}")
        print(f"   Total capacity: {self.max_context_tokens + self.max_new_tokens:,} / {self.model_max_tokens:,}")
        print(f"   Usage: {((self.max_context_tokens + self.max_new_tokens) / self.model_max_tokens * 100):.1f}%")
        
        print(f"\n⚡ OPTIMIZATION:")
        print(f"   4-bit quantization: {self.use_4bit}")
        print(f"   TF32: {self.enable_tf32}")
        print(f"   Flash attention: {self.use_flash_attention}")
        
        print(f"\n🎯 GENERATION:")
        print(f"   Mode: {'Greedy (deterministic)' if self.temperature is None else f'Sampling (temp={self.temperature})'}")
        print(f"   Beams: {self.num_beams}")
        print(f"   Repetition penalty: {self.repetition_penalty}")
        
        print(f"\n💾 CACHE:")
        print(f"   File: {self.cache_file_path}")
        print(f"   Persistence: {self.enable_cache_persistence}")
        print(f"   Overflow policy: {self.cache_overflow_policy}")
        
        print(f"\n📚 KNOWLEDGE:")
        print(f"   JSONL path: {self.knowledge_jsonl_path}")
        print(f"   Max entries: {self.max_knowledge_entries:,}")
        
        print("="*70)
    
    def print_memory_estimate(self):
        """Print estimated memory usage"""
        print("\n" + "="*70)
        print("📊 MEMORY ESTIMATION")
        print("="*70)
        
        # Model size (4-bit quantized Llama 3.2-3B)
        # 3B params * 0.5 bytes (4-bit) = 1.5GB
        # + some overhead for buffers
        model_size_mb = 1200
        
        # KV cache for context tokens
        # Llama 3.2-3B: 28 layers
        # Per token: (hidden_size * 2 * num_layers * 2) bytes for key + value
        # 3072 * 2 * 28 * 2 / 1024 / 1024 = ~0.32 MB per 1k tokens
        bytes_per_token = 16
        kv_cache_mb = (self.max_context_tokens * bytes_per_token * 28) / (1024 * 1024)
        
        # Activation memory (rough estimate)
        activation_mb = 800
        
        # Total
        total_mb = model_size_mb + kv_cache_mb + activation_mb
        
        print(f"\n💾 Component Memory Usage:")
        print(f"   Model (4-bit):               ~{model_size_mb}MB")
        print(f"   KV Cache ({self.max_context_tokens:,} tokens):    ~{kv_cache_mb:.0f}MB")
        print(f"   Activations:                 ~{activation_mb}MB")
        print(f"   {'─'*70}")
        print(f"   Total Estimated:             ~{total_mb:.0f}MB")
        print(f"   Available GPU VRAM:          6140MB (RTX 4050)")
        print(f"   Remaining:                   ~{6140-total_mb:.0f}MB")
        print(f"   Usage:                       {(total_mb/6140)*100:.1f}%")
        
        # Status indicator
        if total_mb > 5800:
            print(f"\n⚠️  WARNING: Very tight on memory!")
            print(f"   Risk: HIGH - May cause OOM errors")
            print(f"   Solutions:")
            print(f"   1. Reduce max_context_tokens to 100000")
            print(f"   2. Reduce max_new_tokens to 200")
            print(f"   3. Close all other GPU programs")
            print(f"   4. Run: python gpu.py")
        elif total_mb > 5000:
            print(f"\n⚠️  Memory usage is high but should work")
            print(f"   Risk: MEDIUM - Monitor for OOM")
            print(f"   Make sure no other programs are using GPU")
        else:
            print(f"\n✅ Memory usage is comfortable")
            print(f"   Risk: LOW - Should run smoothly")
        
        # Calculate expected entries
        avg_tokens_per_entry = 5  # With compact format
        expected_entries = (self.max_context_tokens - self.cache_truncation_buffer) / avg_tokens_per_entry
        coverage_pct = (expected_entries / 44827) * 100
        
        print(f"\n📚 EXPECTED KNOWLEDGE COVERAGE:")
        print(f"   Avg tokens/entry:        ~{avg_tokens_per_entry}")
        print(f"   Expected entries:        ~{expected_entries:,.0f}")
        print(f"   Total available:         44,827")
        print(f"   Coverage:                ~{coverage_pct:.1f}%")
        
        print("="*70)
    
    def validate_for_gpu(self, gpu_memory_mb: int = 6140):
        """Validate configuration for given GPU memory"""
        model_size = 1200
        kv_cache = (self.max_context_tokens * 16 * 28) / (1024 * 1024)
        activation = 800
        total = model_size + kv_cache + activation
        
        if total > gpu_memory_mb * 0.95:
            raise ValueError(
                f"Configuration requires ~{total:.0f}MB but GPU only has {gpu_memory_mb}MB!\n"
                f"Reduce max_context_tokens or max_new_tokens."
            )
        
        return True


# ========================================================================
# PRESET CONFIGURATIONS
# ========================================================================

def get_config_preset(preset_name: str) -> CAGConfig:
    """
    Get a preset configuration
    
    Available presets:
    - "default": Balanced performance (120k context, 256 tokens generation)
    - "max_coverage": Maximum knowledge coverage (125k context, 200 tokens)
    - "fast": Fast responses (100k context, 150 tokens)
    - "safe": Most stable (80k context, 256 tokens)
    """
    
    # All presets sized for 6GB GPU (RTX 4050) with 4-bit Llama 3.2-3B.
    # Model uses ~2.5GB, leaving ~2.5GB for KV cache = ~7500 tokens max.
    presets = {
        "default": CAGConfig(
            max_context_tokens=7500,
            max_new_tokens=256,
            cache_file_path="commercial_kv_cache_7500.pt",
            cache_metadata_path="cache_metadata_7500.json",
        ),
        "large": CAGConfig(
            max_context_tokens=6000,
            max_new_tokens=512,
            cache_file_path="commercial_kv_cache_6k.pt",
            cache_metadata_path="cache_metadata_6k.json",
        ),
        "fast": CAGConfig(
            max_context_tokens=5000,
            max_new_tokens=256,
            cache_file_path="commercial_kv_cache_5k.pt",
            cache_metadata_path="cache_metadata_5k.json",
        ),
        "safe": CAGConfig(
            max_context_tokens=4000,
            max_new_tokens=256,
            cache_file_path="commercial_kv_cache_4k.pt",
            cache_metadata_path="cache_metadata_4k.json",
        ),
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    return presets[preset_name]


# ========================================================================
# GLOBAL CONFIGURATION INSTANCE
# ========================================================================

# Default configuration instance
config = CAGConfig()


# ========================================================================
# CLI INTERFACE FOR TESTING
# ========================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CAG Configuration Tool")
    parser.add_argument(
        '--preset',
        choices=['default', 'max_coverage', 'fast', 'safe'],
        help='Use a preset configuration'
    )
    parser.add_argument(
        '--estimate',
        action='store_true',
        help='Print memory estimation'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate configuration'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.preset:
        config = get_config_preset(args.preset)
        print(f"📦 Loaded preset: {args.preset}")
    else:
        config = CAGConfig()
        print(f"📦 Using default configuration")
    
    # Print summary
    config.print_config_summary()
    
    # Print memory estimate
    if args.estimate:
        config.print_memory_estimate()
    
    # Validate
    if args.validate:
        try:
            config.validate_for_gpu()
            print("\n✅ Configuration is valid for RTX 4050 (6GB)")
        except ValueError as e:
            print(f"\n❌ Configuration validation failed:")
            print(f"   {e}")