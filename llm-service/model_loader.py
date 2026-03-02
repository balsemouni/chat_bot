"""
CAG Architecture - Enhanced Model Loader Module
Centralized model and tokenizer loading with STREAMING support
"""

import torch
import gc
import asyncio
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


class ModelLoader:
    """
    Model Loader - Handles model and tokenizer initialization
    
    Separating this into its own module makes the architecture more modular
    and easier to swap out different models.
    
    NEW: Includes streaming response capabilities for real-time generation
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def load_model_and_tokenizer(self, device):
        """
        Load model and tokenizer with optimizations
        
        Args:
            device: PyTorch device to load model on
        
        Returns:
            Tuple of (model, tokenizer)
        """
        self.device = device
        
        print("\n" + "="*60)
        print("🚀 LOADING MODEL AND TOKENIZER")
        print("="*60)
        print(f"📥 Model: {self.config.model_id}")
        
        # Cleanup before loading
        torch.cuda.empty_cache()
        gc.collect()
        
        # Memory before loading
        free_before = torch.cuda.mem_get_info()[0] // 1024**2
        total_mem = torch.cuda.mem_get_info()[1] // 1024**2
        print(f"\n📊 GPU Memory: {free_before}MB / {total_mem}MB available")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Load model
        self.model = self._load_model()
        
        # Apply optimizations
        self._apply_model_optimizations()
        
        # Memory after loading
        torch.cuda.empty_cache()
        gc.collect()
        
        free_after = torch.cuda.mem_get_info()[0] // 1024**2
        memory_used = free_before - free_after
        
        print(f"\n✅ Model loaded successfully")
        print(f"📊 Memory used: ~{memory_used}MB")
        print(f"📊 Free memory: {free_after}MB")
        
        return self.model, self.tokenizer
    
    def _load_tokenizer(self):
        """Load and configure tokenizer"""
        print("\n🔤 Loading tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        tokenizer.padding_side = "left"
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        return tokenizer
    
    def _load_model(self):
        """Load model with quantization"""
        print("\n🔧 Loading model with quantization...")
        
        # Build quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type=self.config.quant_type,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
            bnb_4bit_compute_dtype=self._get_compute_dtype(),
            llm_int8_enable_fp32_cpu_offload=False
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map={"": 0},
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation="eager"  # Can change to "flash_attention_2" if available
        )
        
        print("✅ Model loaded with 4-bit quantization")
        
        return model
    
    def _apply_model_optimizations(self):
        """Apply model-level optimizations"""
        print("\n⚡️ Applying optimizations...")
        
        # Enable gradient checkpointing if configured
        if self.config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing enabled")
        
        # Set model to eval mode (no gradients)
        self.model.eval()
        print("   ✅ Model set to eval mode")
        
        # Enable TF32 if configured
        if self.config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   ✅ TF32 enabled")
        
        # Configure cuDNN
        torch.backends.cudnn.benchmark = False
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(
            self.config.gpu_memory_fraction,
            device=0
        )
        print(f"   ✅ GPU memory fraction: {self.config.gpu_memory_fraction}")
    
    def _get_compute_dtype(self):
        """Get compute dtype for quantization"""
        if self.config.compute_dtype == "float16":
            return torch.float16
        elif self.config.compute_dtype == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float16
    
    # ========================================================================
    # STREAMING METHODS - FIXED VERSION
    # ========================================================================
    
    def stream_response(self, input_ids, attention_mask=None, max_new_tokens=None):
        """
        Stream model response using TextIteratorStreamer.

        LOGIC FIX: The previous implementation called model() once per token
        in a Python loop, concatenating input_ids every step. This means for
        a 256-token response the model processed 1 token, then 2 tokens, then
        3 tokens … — O(n²) compute cost. It also got no benefit from the KV
        cache because each forward pass started from scratch.

        TextIteratorStreamer runs model.generate() (which uses the KV cache
        correctly) in a background thread and yields decoded text chunks as
        they arrive, giving true O(n) streaming with no wasted compute.

        Args:
            input_ids:       Input token IDs (torch.Tensor)
            attention_mask:  Attention mask (torch.Tensor, optional)
            max_new_tokens:  Maximum tokens to generate

        Yields:
            Text chunks as they are generated
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")

        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30.0,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            repetition_penalty=1.0,
        )
        # Remove None kwargs (attention_mask may be None)
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        def _generate():
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    self.model.generate(**gen_kwargs)

        thread = Thread(target=_generate)
        thread.start()

        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join(timeout=10.0)
    
    def stream_text_response(self, text_input, max_new_tokens=None):
        """
        Stream response from text input (convenience method)
        
        Args:
            text_input: Input text string
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            Individual words as they're generated
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_context_tokens
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Stream response
        yield from self.stream_response(input_ids, attention_mask, max_new_tokens)
    
    async def stream_response_async(self, input_ids, attention_mask=None, max_new_tokens=None):
        """
        Async wrapper over stream_response — uses TextIteratorStreamer via a thread-pool
        executor so the event loop is never blocked.

        LOGIC FIX: Same O(n²) problem as the sync version was. Now delegates to
        the corrected sync method and bridges it to async callers through a Queue.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")

        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                for chunk in self.stream_response(input_ids, attention_mask, max_new_tokens):
                    loop.call_soon_threadsafe(q.put_nowait, ("chunk", chunk))
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        loop.run_in_executor(None, _producer)

        while True:
            kind, value = await q.get()
            if kind == "chunk":
                yield value
            elif kind == "error":
                raise RuntimeError(value)
            else:
                break

    async def stream_text_response_async(self, text_input, max_new_tokens=None):
        """
        Async stream response from text input
        
        Args:
            text_input: Input text string
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            Individual words as they're generated
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_context_tokens
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Stream response
        async for word in self.stream_response_async(input_ids, attention_mask, max_new_tokens):
            yield word
    
    # ========================================================================
    # ORIGINAL METHODS
    # ========================================================================
    
    def get_model(self):
        """Get loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        return self.model
    
    def get_tokenizer(self):
        """Get loaded tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        return self.tokenizer
    
    def unload_model(self):
        """Unload model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Model unloaded")