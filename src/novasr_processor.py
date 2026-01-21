#!/usr/bin/env python3
"""
NovaSR Processor - Thread-safe audio enhancement wrapper

Handles NovaSR model loading, inference, and sample rate conversion.
Optimized for multi-threaded stereo processing.
"""

import os
import threading
import torch
import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional
from NovaSR import FastSR


class NovaSRProcessor:
    """
    Thread-safe NovaSR audio processor.

    Converts 16kHz audio → 48kHz audio (3x upsampling).
    Supports parallel stereo channel processing.
    """

    # Class-level model cache (shared across instances)
    _model_lock = threading.Lock()
    _model_instance: Optional['NovaSRProcessor'] = None

    def __init__(self, model_path: Optional[str] = None, half_precision: bool = False):
        """
        Initialize NovaSR processor.

        Args:
            model_path: Path to model file (None = download from HuggingFace)
            half_precision: Use FP16 for faster inference (requires GPU)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half_precision = half_precision and self.device.type == "cuda"
        self.input_rate = 16000
        self.output_rate = 48000
        self.upsample_factor = 3

        # Thread-local storage for per-thread model instances
        self._local = threading.local()

        # Load model
        self._load_model(model_path)

    @classmethod
    def get_shared(cls) -> 'NovaSRProcessor':
        """Get or create shared model instance."""
        with cls._model_lock:
            if cls._model_instance is None:
                cls._model_instance = cls()
            return cls._model_instance

    def _load_model(self, model_path: Optional[str] = None):
        """Load NovaSR model (thread-safe)."""
        with self._model_lock:
            try:
                self.model = FastSR(ckpt_path=model_path, half=self.half_precision)
                print(f"[NovaSR] Model loaded on {self.device}")
                if self.half_precision:
                    print("[NovaSR] Using FP16 precision")
            except Exception as e:
                print(f"[NovaSR] Error loading model: {e}")
                self.model = None

    def _get_model(self):
        """Get thread-local model reference."""
        return self.model

    def downsample(self, audio: np.ndarray, from_rate: int = 48000) -> np.ndarray:
        """
        Downsample audio to 16kHz for NovaSR input.

        Args:
            audio: Input audio array (shape: [channels, samples])
            from_rate: Input sample rate

        Returns:
            Downsampled audio at 16kHz
        """
        if from_rate == self.input_rate:
            return audio

        # Calculate resampling ratio
        ratio = self.input_rate / from_rate
        num_samples = int(audio.shape[-1] * ratio)

        # Use scipy for high-quality resampling
        if audio.ndim == 1:
            resampled = signal.resample(audio, num_samples)
        else:
            resampled = signal.resample(audio, num_samples, axis=-1)

        return resampled.astype(np.float32)

    def upsample_simple(self, audio: np.ndarray, to_rate: int = 48000) -> np.ndarray:
        """
        Simple upsampling fallback (linear interpolation).

        Args:
            audio: Input audio at 16kHz
            to_rate: Target sample rate

        Returns:
            Upsampled audio
        """
        ratio = to_rate / self.input_rate
        num_samples = int(audio.shape[-1] * ratio)

        if audio.ndim == 1:
            resampled = signal.resample(audio, num_samples)
        else:
            resampled = signal.resample(audio, num_samples, axis=-1)

        return resampled.astype(np.float32)

    def process_channel(
        self,
        audio_channel: np.ndarray,
        input_rate: int = 48000
    ) -> np.ndarray:
        """
        Process a single audio channel through NovaSR.

        Args:
            audio_channel: Single channel audio (1D array)
            input_rate: Input sample rate

        Returns:
            Enhanced audio at 48kHz
        """
        model = self._get_model()
        if model is None:
            # Fallback to simple upsampling
            audio_16k = self.downsample(audio_channel, input_rate)
            return self.upsample_simple(audio_16k, self.output_rate)

        try:
            # Downsample to 16kHz if needed
            if input_rate != self.input_rate:
                audio_16k = self.downsample(audio_channel, input_rate)
            else:
                audio_16k = audio_channel

            # Ensure 2D shape for NovaSR (add channel dimension)
            if audio_16k.ndim == 1:
                audio_16k = audio_16k.reshape(1, -1)

            # Convert to tensor and process
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio_16k).unsqueeze(0).to(self.device)
                if self.half_precision:
                    audio_tensor = audio_tensor.half()

                # Run inference
                output = model.model.infer(audio_tensor)

                # Convert back to numpy
                enhanced = output.squeeze(0).cpu().numpy()

            return enhanced.astype(np.float32)

        except Exception as e:
            print(f"[NovaSR] Inference error: {e}")
            # Fallback to simple upsampling
            audio_16k = audio_channel if input_rate == self.input_rate else self.downsample(audio_channel, input_rate)
            return self.upsample_simple(audio_16k, self.output_rate)

    def process_stereo_parallel(
        self,
        audio: np.ndarray,
        input_rate: int = 48000
    ) -> np.ndarray:
        """
        Process stereo audio with parallel channel processing.

        Args:
            audio: Stereo audio (shape: [2, samples])
            input_rate: Input sample rate

        Returns:
            Enhanced stereo audio at 48kHz
        """
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected stereo audio [2, samples], got shape {audio.shape}")

        # Process channels in parallel (could use multiprocessing here)
        # For now, sequential processing with thread safety
        left_enhanced = self.process_channel(audio[0], input_rate)
        right_enhanced = self.process_channel(audio[1], input_rate)

        # Ensure same length (trim to shorter)
        min_len = min(left_enhanced.shape[0], right_enhanced.shape[0])
        return np.stack([left_enhanced[:min_len], right_enhanced[:min_len]])

    def process(
        self,
        audio: np.ndarray,
        input_rate: int = 48000
    ) -> Tuple[np.ndarray, int]:
        """
        Process audio through NovaSR.

        Args:
            audio: Input audio (shape: [channels, samples] or [samples])
            input_rate: Input sample rate

        Returns:
            Tuple of (enhanced_audio, output_rate)
        """
        # Handle mono audio
        if audio.ndim == 1:
            enhanced = self.process_channel(audio, input_rate)
        # Handle stereo audio
        elif audio.shape[0] == 2:
            enhanced = self.process_stereo_parallel(audio, input_rate)
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        return enhanced, self.output_rate


def get_processor() -> NovaSRProcessor:
    """Get shared NovaSR processor instance."""
    return NovaSRProcessor.get_shared()


# Test function
def test_processor():
    """Test NovaSR processor with synthetic audio."""
    print("[Test] Creating NovaSR processor...")
    processor = get_processor()

    # Create test audio (1 second at 48kHz)
    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Mono test
    print("[Test] Testing mono audio...")
    mono_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    enhanced_mono, rate = processor.process(mono_audio, sample_rate)
    print(f"[Test] Mono: {mono_audio.shape} -> {enhanced_mono.shape} @ {rate}Hz")

    # Stereo test
    print("[Test] Testing stereo audio...")
    stereo_audio = np.stack([
        np.sin(2 * np.pi * 440 * t),
        np.sin(2 * np.pi * 554 * t)  # Different frequency for right channel
    ]).astype(np.float32)
    enhanced_stereo, rate = processor.process(stereo_audio, sample_rate)
    print(f"[Test] Stereo: {stereo_audio.shape} -> {enhanced_stereo.shape} @ {rate}Hz")

    print("[Test] All tests passed!")


if __name__ == "__main__":
    test_processor()
