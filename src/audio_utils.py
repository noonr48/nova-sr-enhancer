#!/usr/bin/env python3
"""
Audio Utilities - Audio I/O and format handling

Provides utilities for audio extraction, processing, and file handling.
Uses ffmpeg for video audio extraction and soundfile for audio I/O.
"""

import os
import subprocess
import tempfile
import soundfile as sf
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


# Supported audio formats
AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.opus'}
# Supported video formats (for audio extraction)
VIDEO_FORMATS = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.wmv', '.flv'}


def is_audio_file(path: str) -> bool:
    """Check if file is an audio format."""
    return Path(path).suffix.lower() in AUDIO_FORMATS


def is_video_file(path: str) -> bool:
    """Check if file is a video format."""
    return Path(path).suffix.lower() in VIDEO_FORMATS


def extract_audio_from_video(
    video_path: str,
    audio_path: Optional[str] = None,
    sample_rate: int = 48000
) -> str:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file
        audio_path: Output audio path (None = auto-generate)
        sample_rate: Output sample rate

    Returns:
        Path to extracted audio file
    """
    if audio_path is None:
        audio_path = str(Path(video_path).with_suffix('.wav'))

    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
        '-ar', str(sample_rate),  # Sample rate
        '-ac', '2',  # Stereo
        '-loglevel', 'error',  # Only show errors
        '-y',  # Overwrite output
        audio_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return audio_path


def load_audio(
    path: str,
    sample_rate: int = 48000
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate (resamples if different)

    Returns:
        Tuple of (audio_array, actual_sample_rate)
    """
    audio, sr = sf.read(path, always_2d=False)

    # Convert to float32 and normalize
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize to [-1, 1] if needed
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    # Resample if needed
    if sr != sample_rate:
        from scipy import signal as scipy_signal
        num_samples = int(len(audio) * sample_rate / sr)
        if audio.ndim == 1:
            audio = scipy_signal.resample(audio, num_samples)
        else:
            audio = scipy_signal.resample(audio, num_samples, axis=-1)
        sr = sample_rate

    # Ensure 2D array (channels, samples)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    else:
        audio = audio.T  # (samples, channels) -> (channels, samples)

    return audio, sr


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int = 48000
):
    """
    Save audio to file.

    Args:
        audio: Audio array (channels, samples)
        path: Output path
        sample_rate: Sample rate
    """
    # Convert to (samples, channels) for soundfile
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    else:
        audio = audio.T

    sf.write(path, audio, sample_rate)


def remux_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: str
):
    """
    Replace audio in video file with enhanced audio.

    Args:
        video_path: Original video file
        audio_path: Enhanced audio file
        output_path: Output video file
    """
    cmd = [
        'ffmpeg',
        '-i', video_path,  # Video input
        '-i', audio_path,  # Audio input
        '-c:v', 'copy',  # Copy video stream (no re-encode)
        '-c:a', 'aac',  # Encode audio as AAC
        '-map', '0:v:0',  # Use video from first input
        '-map', '1:a:0',  # Use audio from second input
        '-shortest',  # Match shortest duration
        '-loglevel', 'error',
        '-y',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg remux failed: {result.stderr}")


def get_audio_duration(path: str) -> float:
    """
    Get audio file duration in seconds.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds
    """
    info = sf.info(path)
    return info.duration


def find_media_files(
    directory: str,
    recursive: bool = True
) -> List[str]:
    """
    Find all audio and video files in directory.

    Args:
        directory: Directory to search
        recursive: Search subdirectories

    Returns:
        List of file paths
    """
    path = Path(directory)
    patterns = []

    if recursive:
        patterns = ['**/*' + ext for ext in AUDIO_FORMATS | VIDEO_FORMATS]
    else:
        patterns = ['*' + ext for ext in AUDIO_FORMATS | VIDEO_FORMATS]

    files = []
    for pattern in patterns:
        files.extend(path.glob(pattern))

    return [str(f) for f in files]


def format_duration(seconds: float) -> str:
    """Format duration as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def format_filesize(bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}TB"


if __name__ == "__main__":
    import sys

    # Test audio utilities
    print("[Audio Utils] Testing...")

    # Create test audio
    test_audio = np.random.randn(2, 48000).astype(np.float32) * 0.1

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        test_path = f.name

    try:
        save_audio(test_audio, test_path, 48000)
        print(f"[Audio Utils] Saved test audio: {test_path}")

        loaded, sr = load_audio(test_path)
        print(f"[Audio Utils] Loaded: {loaded.shape} @ {sr}Hz")

        duration = get_audio_duration(test_path)
        print(f"[Audio Utils] Duration: {format_duration(duration)}")

    finally:
        if os.path.exists(test_path):
            os.unlink(test_path)

    print("[Audio Utils] Tests passed!")
