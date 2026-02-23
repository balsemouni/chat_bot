"""
CAG Architecture - Enhanced Model Loader Module
Centralized model and tokenizer loading with STREAMING support
"""

import torch
import gc
import asyncio
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
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
        Stream model response WORD BY WORD - FIXED VERSION
        
        Args:
            input_ids: Input token IDs (torch.Tensor)
            attention_mask: Attention mask (torch.Tensor, optional)
            max_new_tokens: Maximum tokens to generate (uses config default if None)
            
        Yields:
            Individual words IMMEDIATELY as they're generated
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Keep track of generated text
        char_buffer = []
        generated_tokens = []
        full_generated_text = ""  # Track full generated text for stop string detection
        
        # Stop strings that indicate the model is simulating conversation
        stop_strings = [
            "User:", "user:", "USER:",
            "<|start_header_id|>user<|end_header_id|>",
            "\nUser:", "\nuser:",
        ]
        
        # Keep original input length
        original_length = input_ids.shape[-1]
        
        with torch.no_grad():
            # Generate tokens one at a time
            for token_idx in range(max_new_tokens):
                # Get next token
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Store generated token
                generated_tokens.append(next_token.item())
                
                # Check for EOS - THIS IS CRITICAL
                if next_token.item() == self.tokenizer.eos_token_id:
                    # Flush any remaining characters in buffer
                    if char_buffer:
                        word = ''.join(char_buffer)
                        if word.strip():  # Only yield if not empty
                            yield word + ' '
                    break
                
                # Decode ONLY the new token
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                # Add to full generated text for stop string detection
                full_generated_text += token_text
                
                # Check if we hit a stop string (model trying to simulate conversation)
                should_stop = False
                for stop_str in stop_strings:
                    if stop_str in full_generated_text:
                        should_stop = True
                        break
                
                if should_stop:
                    # Flush buffer and stop
                    if char_buffer:
                        word = ''.join(char_buffer)
                        if word.strip():
                            yield word + ' '
                    break
                
                # Check if we're generating repeated output (indicates loop)
                if len(generated_tokens) > 10:
                    # Check last 10 tokens for repetition
                    last_10 = generated_tokens[-10:]
                    if len(set(last_10)) <= 2:  # Only 1-2 unique tokens in last 10
                        if char_buffer:
                            word = ''.join(char_buffer)
                            if word.strip():
                                yield word + ' '
                        break
                
                # Process character by character for instant word detection
                for char in token_text:
                    # Word boundary detected (space, tab, newline)
                    if char in ' \t\n':
                        if char_buffer:
                            word = ''.join(char_buffer)
                            char_buffer = []
                            # ⚡ YIELD WORD IMMEDIATELY (with trailing space)
                            if word.strip():  # Only yield non-empty words
                                yield word + ' '
                    
                    # Punctuation that ends a word
                    elif char in '.,!?;:':
                        # Yield current word if exists
                        if char_buffer:
                            word = ''.join(char_buffer)
                            char_buffer = []
                            if word.strip():
                                yield word
                        # ⚡ YIELD PUNCTUATION IMMEDIATELY (with space)
                        yield char + ' '
                    
                    else:
                        # Regular character - build current word
                        char_buffer.append(char)
                
                # CRITICAL FIX: Update input_ids for next iteration
                # This is necessary for the model to continue generation
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                    ], dim=-1)
        
        # Yield any remaining word at end of generation
        if char_buffer:
            final_word = ''.join(char_buffer)
            if final_word.strip():
                yield final_word
    
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
        Async version - yields INDIVIDUAL WORDS IMMEDIATELY
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            Individual words as they're generated
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        char_buffer = []
        generated_tokens = []
        full_generated_text = ""  # Track for stop strings
        
        # Stop strings
        stop_strings = [
            "User:", "user:", "USER:",
            "<|start_header_id|>user<|end_header_id|>",
            "\nUser:", "\nuser:",
        ]
        
        with torch.no_grad():
            for token_idx in range(max_new_tokens):
                # Get next token
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_tokens.append(next_token.item())
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    if char_buffer:
                        word = ''.join(char_buffer)
                        if word.strip():
                            yield word + ' '
                    break
                
                # Decode the token
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                # Add to full text for stop detection
                full_generated_text += token_text
                
                # Check for stop strings
                should_stop = False
                for stop_str in stop_strings:
                    if stop_str in full_generated_text:
                        should_stop = True
                        break
                
                if should_stop:
                    if char_buffer:
                        word = ''.join(char_buffer)
                        if word.strip():
                            yield word + ' '
                    break
                
                # Check for repetition
                if len(generated_tokens) > 10:
                    last_10 = generated_tokens[-10:]
                    if len(set(last_10)) <= 2:
                        if char_buffer:
                            word = ''.join(char_buffer)
                            if word.strip():
                                yield word + ' '
                        break
                
                # Process EACH CHARACTER for instant word detection
                for char in token_text:
                    # Word boundary: space, tab, newline
                    if char in ' \t\n':
                        if char_buffer:
                            word = ''.join(char_buffer)
                            char_buffer = []
                            if word.strip():
                                yield word + ' '
                    
                    # Punctuation that ends a word
                    elif char in '.,!?;:':
                        if char_buffer:
                            word = ''.join(char_buffer)
                            char_buffer = []
                            if word.strip():
                                yield word
                        yield char + ' '
                    
                    else:
                        # Build current word
                        char_buffer.append(char)
                
                # Update for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                    ], dim=-1)
                
                # Yield control to event loop
                await asyncio.sleep(0)
        
        # Yield remaining word
        if char_buffer:
            final_word = ''.join(char_buffer)
            if final_word.strip():
                yield final_word
    
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