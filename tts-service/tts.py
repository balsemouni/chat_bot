"""
🔥 ULTRA-LOW LATENCY TTS WITH POCKET-TTS 🔥
============================================
Multiple TTS backends with streaming support:
- PocketTTS: Sub-50ms latency streaming
- Piper: Fast ONNX-based TTS
"""

import threading
import queue
import time
import numpy as np
import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# ============================================================================
# POCKET-TTS STREAMING TTS (FASTEST - Sub-50ms latency)
# ============================================================================

class PocketTTSStreaming:
    """
    POCKET-TTS INSTANT STREAMING TTS
    
    ⚡ FEATURES:
    - Sub-50ms time to first audio
    - True streaming (generates while speaking)
    - Multiple voice options
    - Zero buffering latency
    """
    
    def __init__(self, voice: str = "alba", rate: float = 1.0):
        """
        Args:
            voice: Voice name (alba, etc.)
            rate: Speech rate multiplier (1.0 = normal)
        """
        print("⚡ Initializing POCKET-TTS STREAMING...")
        
        try:
            from pocket_tts import TTSModel
            self.TTSModel = TTSModel
        except ImportError:
            raise ImportError(
                "pocket_tts not installed!\n"
                "Install with: pip install pocket-tts"
            )
        
        # Load model
        self.model = self.TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt(voice)
        self.rate = rate
        self.sample_rate = self.model.sample_rate
        
        # State
        self._is_speaking = False
        self._should_stop = False
        self._lock = threading.Lock()
        self._token_buffer = []
        
        # Stream for continuous playback
        self._stream = None
        
        print(f"✅ POCKET-TTS ready (voice: {voice}, rate: {rate}x)")
    
    def process_token(self, token: str):
        """
        Process token with intelligent chunking
        
        Strategy:
        - Accumulate tokens
        - Speak on punctuation or after 4-5 words
        - Stream audio as it generates
        """
        if not token or self._should_stop:
            return
        
        with self._lock:
            self._token_buffer.append(token)
            combined = ''.join(self._token_buffer)
        
        # Check if we should speak
        word_count = len(combined.split())
        has_punct = any(c in combined for c in '.!?,;:\n')
        
        # Speak on natural breaks or after 4 words
        if has_punct or word_count >= 4:
            text = combined.strip()
            if text:
                self._speak_chunk_streaming(text)
                with self._lock:
                    self._token_buffer = []
    
    def _speak_chunk_streaming(self, text: str):
        """
        Stream audio chunks with ZERO latency
        
        ⚡ KEY OPTIMIZATION:
        - Audio plays WHILE generating (not after)
        - Sub-50ms time to first audio
        - Smooth continuous playback
        """
        if not text.strip():
            return
        
        with self._lock:
            self._is_speaking = True
        
        try:
            # Start stream if not already active
            if self._stream is None or not self._stream.active:
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32'
                )
                self._stream.start()
            
            # ⚡ STREAM AUDIO CHUNKS AS THEY GENERATE
            for chunk in self.model.generate_audio_stream(self.voice_state, text):
                if self._should_stop:
                    break
                
                # Convert torch tensor to numpy
                audio_data = chunk.numpy()
                
                # Apply rate adjustment if needed
                if self.rate != 1.0:
                    # Simple rate adjustment (consider using librosa for better quality)
                    audio_data = self._adjust_rate(audio_data, self.rate)
                
                # Play immediately
                self._stream.write(audio_data)
        
        except Exception as e:
            print(f"⚠️ Streaming error: {e}")
        
        finally:
            with self._lock:
                self._is_speaking = False
    
    def _adjust_rate(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Simple rate adjustment by resampling"""
        if rate == 1.0:
            return audio
        
        # Simple linear interpolation for rate change
        # For better quality, use librosa.effects.time_stretch
        indices = np.arange(0, len(audio), rate)
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]
    
    def flush(self):
        """Flush remaining tokens"""
        with self._lock:
            if self._token_buffer:
                text = ''.join(self._token_buffer).strip()
                if text:
                    self._speak_chunk_streaming(text)
                self._token_buffer = []
        
        # Wait for playback to complete
        while self._is_speaking and not self._should_stop:
            time.sleep(0.01)
    
    def stop(self):
        """Emergency stop"""
        with self._lock:
            self._should_stop = True
            self._is_speaking = False
            self._token_buffer = []
        
        if self._stream and self._stream.active:
            self._stream.stop()
        
        # Reset after brief delay
        threading.Thread(target=self._reset_stop, daemon=True).start()
    
    def _reset_stop(self):
        time.sleep(0.05)
        with self._lock:
            self._should_stop = False
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        with self._lock:
            return self._is_speaking
    
    def shutdown(self):
        """Clean shutdown"""
        print("🛑 Shutting down POCKET-TTS...")
        self.stop()
        if self._stream:
            self._stream.close()
        print("✅ POCKET-TTS shutdown complete")


# ============================================================================
# PIPER TTS IMPLEMENTATIONS (for backwards compatibility)
# ============================================================================

class InstantStreamingTTS:
    """
    PIPER-BASED INSTANT TTS
    Kept for backwards compatibility
    """
    
    def __init__(self, model_path: str = "en_US-lessac-medium.onnx", 
                 config_path: str = None, rate: float = 1.0):
        print("⚡ Initializing PIPER TTS...")
        
        try:
            from piper.voice import PiperVoice
        except ImportError:
            raise ImportError(
                "piper-tts not installed!\n"
                "Install with: pip install piper-tts"
            )
        
        if config_path is None:
            config_path = model_path + ".json"
        
        self.voice = PiperVoice.load(model_path, config_path=config_path)
        self.rate = rate
        self.volume = 1.0
        
        # Pre-warm pipeline
        try:
            dummy_audio = self._synthesize_to_numpy("Hello")
            print("✓ Pipeline pre-warmed")
        except Exception as e:
            print(f"⚠️ Pre-warm warning: {e}")
        
        # State
        self.is_speaking_flag = False
        self.should_stop = False
        self.lock = threading.Lock()
        
        # Thread pool
        self.synthesis_executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="synth"
        )
        
        # Queue
        self.audio_queue = queue.PriorityQueue(maxsize=20)
        self.chunk_counter = 0
        
        # Playback worker
        self.is_running = True
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="playback"
        )
        self.playback_thread.start()
        
        print("✅ PIPER TTS ready")
    
    def process_token(self, token: str):
        if not token or self.should_stop:
            return
        
        if not hasattr(self, '_token_buffer'):
            self._token_buffer = []
        
        self._token_buffer.append(token)
        combined = ''.join(self._token_buffer)
        
        word_count = len(combined.split())
        has_punctuation = any(c in combined for c in '.!?,;:')
        
        if has_punctuation or word_count >= 4 or len(self._token_buffer) >= 8:
            text = combined.strip()
            if text:
                self._speak_chunk(text)
                self._token_buffer = []
    
    def _speak_chunk(self, text: str):
        if not text.strip():
            return
        
        with self.lock:
            self.is_speaking_flag = True
            self.chunk_counter += 1
        
        future = self.synthesis_executor.submit(self._synthesize_audio, text)
        priority = len(text)
        
        try:
            self.audio_queue.put_nowait((priority, self.chunk_counter, future))
        except queue.Full:
            try:
                self.audio_queue.put((priority, self.chunk_counter, future), timeout=0.1)
            except:
                pass
    
    def _synthesize_to_numpy(self, text: str) -> np.ndarray:
        audio_bytes = bytes()
        
        for audio_chunk in self.voice.synthesize_stream_raw(
            text, 
            length_scale=1.0 / self.rate
        ):
            audio_bytes += audio_chunk
        
        if not audio_bytes:
            return None
        
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        audio_float *= self.volume
        
        return audio_float
    
    def _synthesize_audio(self, text: str) -> tuple:
        try:
            clean_text = self._clean_text(text)
            if not clean_text:
                return None
            
            audio = self._synthesize_to_numpy(clean_text)
            if audio is None or len(audio) == 0:
                return None
            
            return (audio, self.voice.config.sample_rate)
        
        except Exception as e:
            print(f"⚠️ Synthesis error: {e}")
            return None
    
    def _playback_worker(self):
        while self.is_running:
            try:
                priority, counter, future = self.audio_queue.get(timeout=0.05)
                
                if future is None:
                    break
                
                if self.should_stop:
                    self.audio_queue.task_done()
                    continue
                
                result = future.result(timeout=3.0)
                
                if result is not None and not self.should_stop:
                    audio, sample_rate = result
                    sd.play(audio, samplerate=sample_rate, blocking=True)
                
                self.audio_queue.task_done()
            
            except queue.Empty:
                with self.lock:
                    if self.audio_queue.empty():
                        self.is_speaking_flag = False
                continue
            
            except Exception as e:
                print(f"⚠️ Playback error: {e}")
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        clean = ' '.join(text.split())
        clean = clean.replace('&', 'and').replace('@', 'at')
        return clean.strip()
    
    def flush(self):
        if hasattr(self, '_token_buffer') and self._token_buffer:
            text = ''.join(self._token_buffer).strip()
            if text:
                self._speak_chunk(text)
            self._token_buffer = []
        
        self.audio_queue.join()
    
    def stop(self):
        with self.lock:
            self.should_stop = True
            self.is_speaking_flag = False
        
        if hasattr(self, '_token_buffer'):
            self._token_buffer = []
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        sd.stop()
        threading.Thread(target=self._reset_stop, daemon=True).start()
    
    def _reset_stop(self):
        time.sleep(0.05)
        with self.lock:
            self.should_stop = False
    
    def is_speaking(self) -> bool:
        with self.lock:
            return self.is_speaking_flag or not self.audio_queue.empty()
    
    def shutdown(self):
        print("🛑 Shutting down TTS...")
        self.is_running = False
        self.flush()
        self.stop()
        
        try:
            self.audio_queue.put_nowait((0, 0, None))
        except:
            pass
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
        
        self.synthesis_executor.shutdown(wait=True)
        print("✅ TTS shutdown complete")


# ============================================================================
# WRAPPER CLASSES FOR DIFFERENT MODES
# ============================================================================

class AggressiveInstantTTS:
    """Speaks after just 2 words (FASTEST)"""
    
    def __init__(self, model_path: str = None, config_path: str = None, 
                 rate: float = 1.2, backend: str = "pocket"):
        """
        Args:
            backend: 'pocket' for PocketTTS, 'piper' for Piper
        """
        if backend == "pocket":
            self.tts = PocketTTSStreaming(rate=rate)
        else:
            self.tts = InstantStreamingTTS(
                model_path=model_path, 
                config_path=config_path, 
                rate=rate
            )
        
        self._buffer = []
        self._lock = threading.Lock()
    
    def process_token(self, token: str):
        if not token:
            return
        
        with self._lock:
            self._buffer.append(token)
            combined = ''.join(self._buffer)
        
        word_count = len(combined.split())
        has_punct = any(c in combined for c in '.!?,;:\n')
        
        if has_punct or word_count >= 2:
            text = combined.strip()
            if text:
                if hasattr(self.tts, '_speak_chunk'):
                    self.tts._speak_chunk(text)
                else:
                    self.tts._speak_chunk_streaming(text)
                
                with self._lock:
                    self._buffer = []
    
    def flush(self):
        with self._lock:
            if self._buffer:
                text = ''.join(self._buffer).strip()
                if text:
                    if hasattr(self.tts, '_speak_chunk'):
                        self.tts._speak_chunk(text)
                    else:
                        self.tts._speak_chunk_streaming(text)
                self._buffer = []
        self.tts.flush()
    
    def stop(self):
        self.tts.stop()
    
    def is_speaking(self):
        return self.tts.is_speaking()
    
    def shutdown(self):
        self.tts.shutdown()


class UltraSmoothTTS:
    """Balanced speed with natural flow (RECOMMENDED)"""
    
    def __init__(self, model_path: str = None, config_path: str = None, 
                 rate: float = 1.0, backend: str = "pocket"):
        """
        Args:
            backend: 'pocket' for PocketTTS (recommended), 'piper' for Piper
        """
        if backend == "pocket":
            self.tts = PocketTTSStreaming(rate=rate)
        else:
            self.tts = InstantStreamingTTS(
                model_path=model_path,
                config_path=config_path,
                rate=rate
            )
        
        self._buffer = []
        self._lock = threading.Lock()
    
    def process_token(self, token: str):
        if not token:
            return
        
        with self._lock:
            self._buffer.append(token)
            combined = ''.join(self._buffer)
        
        word_count = len(combined.split())
        ends_sentence = any(combined.rstrip().endswith(c) for c in '.!?')
        ends_clause = any(combined.rstrip().endswith(c) for c in ',;:')
        
        should_speak = (
            ends_sentence or
            (ends_clause and word_count >= 3) or
            word_count >= 5
        )
        
        if should_speak:
            text = combined.strip()
            if text:
                if hasattr(self.tts, '_speak_chunk'):
                    self.tts._speak_chunk(text)
                else:
                    self.tts._speak_chunk_streaming(text)
                
                with self._lock:
                    self._buffer = []
    
    def flush(self):
        with self._lock:
            if self._buffer:
                text = ''.join(self._buffer).strip()
                if text:
                    if hasattr(self.tts, '_speak_chunk'):
                        self.tts._speak_chunk(text)
                    else:
                        self.tts._speak_chunk_streaming(text)
                self._buffer = []
        self.tts.flush()
    
    def stop(self):
        self.tts.stop()
    
    def is_speaking(self):
        return self.tts.is_speaking()
    
    def shutdown(self):
        self.tts.shutdown()


# Alias for compatibility
SmartStreamingTTS = UltraSmoothTTS


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎤 TTS BACKEND COMPARISON")
    print("="*60)
    
    # Test PocketTTS
    print("\n1️⃣ Testing POCKET-TTS (Sub-50ms latency)...")
    tts1 = PocketTTSStreaming(voice="alba", rate=1.0)
    
    test_text = "Hello world, this is a test of the ultra low latency streaming system."
    
    start = time.perf_counter()
    for word in test_text.split():
        tts1.process_token(word + " ")
        time.sleep(0.05)  # Simulate token arrival
    
    tts1.flush()
    duration = time.perf_counter() - start
    
    print(f"✅ Total duration: {duration:.2f}s")
    tts1.shutdown()
    
    print("\n" + "="*60)
    print("Choose your TTS backend:")
    print("  • PocketTTS  - Sub-50ms latency (FASTEST)")
    print("  • Piper      - Fast ONNX-based (alternative)")
    print("="*60)