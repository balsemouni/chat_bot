"""
AI Voice Detector
=================
Detects if audio is AI-generated vs human voice to prevent feedback loops
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
import threading
from typing import Tuple, Optional


class AudioCNN(nn.Module):
    """CNN model for audio classification"""
    def __init__(self, n_mels=256, num_classes=2, input_channels=1):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu(self.bn4(self.conv4(x)))))
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc2(self.dropout5(self.relu(self.fc1(x))))
        return x


class AudioResNet(nn.Module):
    """ResNet model for audio classification"""
    def __init__(self, num_classes=2, input_channels=3):
        super(AudioResNet, self).__init__()
        import torchvision.models as models
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class AIVoiceDetector:
    """
    Detects AI-generated vs human voice to prevent feedback loops
    
    This prevents the system from:
    - Transcribing its own TTS output
    - Barging in on itself
    - Creating audio feedback loops
    """
    
    def __init__(self, model_path: str, device: str = "cuda", confidence_threshold: float = 0.7):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence to classify as AI (0.7 = 70%)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self._lock = threading.Lock()
        
        print(f"🔍 Loading AI voice detector on {self.device}...")
        
        # Load checkpoint to determine model architecture
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_type = checkpoint.get('model_type', 'resnet')
        self.use_delta = checkpoint.get('use_delta', True)
        self.n_mels = checkpoint.get('n_mels', 256)
        
        # Create model
        input_channels = 3 if self.use_delta else 1
        if self.model_type == 'resnet':
            self.model = AudioResNet(num_classes=2, input_channels=input_channels)
        else:
            self.model = AudioCNN(n_mels=self.n_mels, num_classes=2, input_channels=input_channels)
        
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ AI voice detector ready (threshold: {confidence_threshold:.0%})")
    
    def _audio_to_melspectrogram(self, audio: np.ndarray, sr: int = 16000, duration: int = 3) -> np.ndarray:
        """Convert audio to mel spectrogram"""
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        
        # Pad or truncate to target length
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db + 40) / 40
        return mel_spec_db
    
    def _get_delta_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract mel + delta + delta-delta features"""
        mel = self._audio_to_melspectrogram(audio, sr=sr)
        delta = librosa.feature.delta(mel)
        delta2 = librosa.feature.delta(mel, order=2)
        return np.stack([mel, delta, delta2])
    
    def is_ai_voice(self, audio: np.ndarray, sr: int = 16000) -> Tuple[bool, float, dict]:
        """
        Detect if audio is AI-generated
        
        Args:
            audio: Audio samples as numpy array
            sr: Sample rate
        
        Returns:
            Tuple of (is_ai, confidence, probabilities)
            - is_ai: True if AI voice detected
            - confidence: Confidence score (0-1)
            - probabilities: {'ai': prob, 'human': prob}
        """
        with self._lock:
            try:
                # Check minimum audio length
                min_samples = sr * 0.5  # At least 0.5 seconds
                if len(audio) < min_samples:
                    # Not enough audio - assume human (safer default)
                    return False, 0.0, {'ai': 0.0, 'human': 1.0}
                
                # Extract features
                if self.use_delta:
                    features = self._get_delta_features(audio, sr=sr)
                else:
                    mel_spec = self._audio_to_melspectrogram(audio, sr=sr)
                    features = mel_spec[np.newaxis, :, :]
                
                # Normalize
                features = (features - features.mean()) / (features.std() + 1e-8)
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    
                    ai_prob = probabilities[0].item()
                    human_prob = probabilities[1].item()
                    
                    is_ai = ai_prob >= self.confidence_threshold
                    confidence = ai_prob if is_ai else human_prob
                    
                    return is_ai, confidence, {'ai': ai_prob, 'human': human_prob}
            
            except Exception as e:
                print(f"⚠️ AI detection error: {e}")
                # On error, assume human (safer default to avoid blocking real users)
                return False, 0.0, {'ai': 0.0, 'human': 1.0}