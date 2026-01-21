#!/usr/bin/env python3
"""
Live Enhancer - Real-time audio enhancement using NovaSR

Captures system audio via PipeWire, processes through NovaSR,
and outputs to speakers. Uses multi-threaded pipeline.
"""

import sys
import signal
import time
import threading
import pyaudio
import numpy as np
from typing import Optional

from novasr_processor import get_processor
from thread_manager import ThreadManager, BoundedQueue, AffinityThread
from audio_utils import format_duration


class LiveEnhancer:
    """
    Real-time audio enhancement using NovaSR.

    Pipeline:
    [Capture] → [Queue A] → [NovaSR Process] → [Queue B] → [Playback]
    Thread 1     16 chunks     Thread 2-3          8 chunks    Thread 4
    """

    def __init__(
        self,
        chunk_size: int = 1920,  # 40ms @ 48kHz
        sample_rate: int = 48000,
        channels: int = 2,
        input_device: Optional[str] = None,
        output_device: Optional[str] = None
    ):
        """
        Initialize live enhancer.

        Args:
            chunk_size: Samples per chunk
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            input_device: Input device name (None = default)
            output_device: Output device name (None = default)
        """
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels

        # NovaSR processor
        self.processor = get_processor()

        # Audio I/O
        self.audio = pyaudio.PyAudio()

        # Find devices
        self.input_device_index = self._find_device('input', input_device)
        self.output_device_index = self._find_device('output', output_device)

        # Queues
        self.input_queue = BoundedQueue(maxsize=16)
        self.output_queue = BoundedQueue(maxsize=8)

        # Control
        self.running = False
        self.threads = []

        # Streams
        self.input_stream = None
        self.output_stream = None

    def _find_device(self, direction: str, device_name: Optional[str]) -> Optional[int]:
        """Find audio device by name."""
        if device_name is None:
            return None

        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if device_name.lower() in info['name'].lower():
                max_channels = info['maxInputChannels' if direction == 'input' else 'maxOutputChannels']
                if max_channels > 0:
                    print(f"[Device] Found {direction}: {info['name']} (index: {i})")
                    return i

        print(f"[Warning] Device '{device_name}' not found, using default")
        return None

    def list_devices(self):
        """List all available audio devices."""
        print("\n" + "=" * 60)
        print("Available Audio Devices")
        print("=" * 60)

        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            name = info['name']
            max_in = info['maxInputChannels']
            max_out = info['maxOutputChannels']
            rate = int(info['defaultSampleRate'])

            flags = []
            if max_in > 0:
                flags.append(f"IN:{max_in}")
            if max_out > 0:
                flags.append(f"OUT:{max_out}")

            print(f"  [{i:2d}] {name}")
            print(f"       {rate}Hz | {' | '.join(flags)}")

        print("=" * 60 + "\n")

    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio capture."""
        if status:
            print(f"[Warning] Input status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Reshape to (channels, samples)
        if self.channels == 2:
            audio_data = audio_data.reshape(frame_count, self.channels).T
        else:
            audio_data = audio_data.reshape(1, -1)

        # Put in queue for processing
        try:
            self.input_queue.put_nowait(audio_data)
        except Exception:
            # Queue full, drop frame
            pass

        return (None, pyaudio.paContinue)

    def audio_output_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio playback."""
        if status:
            print(f"[Warning] Output status: {status}")

        try:
            # Get processed audio from queue
            audio_chunk = self.output_queue.get_nowait()

            # Convert to bytes
            audio_bytes = audio_chunk.T.astype(np.float32).tobytes()

            # Pad if needed
            needed = frame_count * self.channels
            if len(audio_bytes) < needed:
                audio_bytes += b'\x00' * (needed - len(audio_bytes))

            return (audio_bytes[:needed], pyaudio.paContinue)

        except Exception:
            # Return silence if no data
            silence = np.zeros((frame_count, self.channels), dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)

    def processing_loop(self):
        """Main processing loop (runs in separate thread)."""
        while self.running:
            try:
                # Get audio chunk from input queue
                audio_chunk = self.input_queue.get(timeout=0.1)

                # Process through NovaSR
                enhanced_audio, _ = self.processor.process(
                    audio_chunk,
                    input_rate=self.sample_rate
                )

                # Put in output queue
                try:
                    self.output_queue.put_nowait(enhanced_audio)
                except Exception:
                    # Output queue full, drop frame
                    pass

            except Exception:
                continue

    def start(self):
        """Start live audio enhancement."""
        if self.running:
            print("[Error] Already running")
            return

        print("[Live] Starting NovaSR Live Enhancer...")
        print(f"[Live] Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.0f}ms)")
        print(f"[Live] Sample rate: {self.sample_rate}Hz")
        print(f"[Live] Channels: {self.channels}")

        try:
            # Create input stream
            self.input_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_input_callback
            )

            # Create output stream (account for upsampling)
            output_chunk_size = self.chunk_size * 3  # NovaSR does 3x upsampling
            self.output_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=output_chunk_size,
                stream_callback=self.audio_output_callback
            )

            # Start processing thread
            self.running = True
            proc_thread = threading.Thread(target=self.processing_loop, daemon=True)
            proc_thread.start()
            self.threads.append(proc_thread)

            # Start streams
            self.input_stream.start_stream()
            self.output_stream.start_stream()

            print("[Live] Enhancement active! Press Ctrl+C to stop.")
            print("[Live] Note: ~150-250ms latency is expected\n")

        except Exception as e:
            print(f"[Error] Failed to start: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop live audio enhancement."""
        if not self.running:
            return

        print("\n[Live] Stopping...")
        self.running = False

        # Stop streams
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=1.0)

        self.audio.terminate()
        print("[Live] Stopped")

    def run(self):
        """Run enhancer until interrupted."""
        self.start()

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time audio enhancement with NovaSR"
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available audio devices'
    )
    parser.add_argument(
        '--input-device',
        help='Input device name'
    )
    parser.add_argument(
        '--output-device',
        help='Output device name'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1920,
        help='Chunk size in samples (default: 1920 = 40ms @ 48kHz)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        choices=[16000, 48000],
        help='Sample rate (default: 48000)'
    )

    args = parser.parse_args()

    enhancer = LiveEnhancer(
        chunk_size=args.chunk_size,
        sample_rate=args.sample_rate,
        input_device=args.input_device,
        output_device=args.output_device
    )

    if args.list:
        enhancer.list_devices()
    else:
        # Setup signal handlers
        signal.signal(signal.SIGINT, lambda s, f: enhancer.stop())
        signal.signal(signal.SIGTERM, lambda s, f: enhancer.stop())

        enhancer.run()
