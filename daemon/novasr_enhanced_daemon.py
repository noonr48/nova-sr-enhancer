#!/usr/bin/env python3
"""
NovaSR Enhanced Audio Daemon
Multi-process audio enhancement system for PipeWire/PulseAudio

Creates virtual "Enhanced" versions of each physical device:
- USB Speakers (Enhanced)
- USB Headphones (Enhanced)
- USB S/PDIF (Enhanced)

Each virtual device has its own processing chain:
virtual_sink -> parec -> NovaSR -> paplay -> physical_sink
"""

import sys
import os
import subprocess
import threading
import time
import signal
import multiprocessing as mp
from queue import Empty
import numpy as np
import torch
from NovaSR import FastSR

# Configuration
NOVASR_RATE = 16000
OUTPUT_RATE = 48000
CHANNELS = 1
CHUNK_SIZE = 9600  # NovaSR processing chunk size
RESAMPLE_CHUNK_SIZE = CHUNK_SIZE * 3  # 48kHz chunk for downsampling

# Disable CUDA - use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = "cpu"

# Virtual sink configurations
VIRTUAL_SINKS = {
    "novasr_speakers": {
        "sink_name": "novasr_speakers",
        "sink_description": "NovaSR_Enhanced_Speakers",
        "physical_sink": "alsa_output.pci-0000_c1_00.6.HiFi__Speaker__sink",
    }
}


def create_virtual_sink(sink_name, sink_description):
    """Create a virtual null sink using pactl"""
    try:
        # Check if sink already exists
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            check=True
        )

        if sink_name in result.stdout:
            print(f"[Setup] Virtual sink already exists: {sink_name}")
            return True

        # Create the sink using pactl
        subprocess.run(
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={sink_name}",
                f"sink_properties=device.description={sink_description}",
                "rate=48000",
                "channels=2"
            ],
            capture_output=True,
            check=True,
            text=True
        )

        print(f"[Setup] Created virtual sink: {sink_description}")
        time.sleep(0.5)  # Give it time to initialize
        return True

    except subprocess.CalledProcessError as e:
        print(f"[Setup] Error creating sink {sink_name}: {e.stderr if e.stderr else e}")
        return False
    except Exception as e:
        print(f"[Setup] Unexpected error creating sink: {e}")
        return False


def get_sink_id(sink_name):
    """Get the numeric sink ID by name"""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2 and parts[1] == sink_name:
                sink_id = parts[0]
                print(f"[Setup] Found sink ID {sink_id} for {sink_name}")
                return sink_id

        print(f"[Setup] Sink {sink_name} not found")
        return None
    except Exception as e:
        print(f"[Setup] Error getting sink ID: {e}")
        return None


def set_default_sink(sink_name):
    """Set the default sink with retry logic"""
    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            # First verify the sink exists
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                capture_output=True,
                text=True,
                check=True
            )

            if sink_name not in result.stdout:
                print(f"[Setup] Sink {sink_name} not found, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue

            # Set the default sink
            result = subprocess.run(
                ["pactl", "set-default-sink", sink_name],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"[Setup] Set default sink to: {sink_name}")
                return True
            else:
                print(f"[Setup] Error setting default sink: {result.stderr}")
                return False

        except Exception as e:
            print(f"[Setup] Error setting default sink (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    print(f"[Setup] Failed to set default sink after {max_retries} attempts")
    return False


def create_all_virtual_sinks():
    """Create all virtual sinks and set the first one as default"""
    print("[Setup] Creating virtual sinks...")
    success_count = 0
    first_sink = None

    for key, config in VIRTUAL_SINKS.items():
        if create_virtual_sink(config["sink_name"], config["sink_description"]):
            success_count += 1
            if first_sink is None:
                first_sink = config["sink_name"]

    print(f"[Setup] Created {success_count}/{len(VIRTUAL_SINKS)} virtual sinks")

    # Set the first virtual sink as default
    if first_sink:
        set_default_sink(first_sink)

    return success_count > 0


def remove_virtual_sink(sink_name):
    """Remove a virtual sink"""
    try:
        # Find the module index for this sink
        result = subprocess.run(
            ["pactl", "list", "short", "modules"],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.split('\n'):
            if sink_name in line and 'module-null-sink' in line:
                module_index = line.split()[0]
                subprocess.run(
                    ["pactl", "unload-module", module_index],
                    capture_output=True,
                    check=True
                )
                print(f"[Cleanup] Removed virtual sink: {sink_name}")
                return True

    except Exception as e:
        print(f"[Cleanup] Error removing sink {sink_name}: {e}")

    return False


def remove_all_virtual_sinks():
    """Remove all virtual sinks"""
    print("[Cleanup] Removing virtual sinks...")
    for key, config in VIRTUAL_SINKS.items():
        remove_virtual_sink(config["sink_name"])


class NovaSRProcessor:
    """Process audio through NovaSR with resampling"""

    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        print(f"[NovaSR] Initializing on {DEVICE}...")

        try:
            self.model = FastSR()
            self.model.device = torch.device('cpu')
            self.model.model = self.model.model.to('cpu').float()
            self.model.half = False
            print("[NovaSR] Model loaded successfully")
        except Exception as e:
            print(f"[NovaSR] Error loading model: {e}")
            raise

        self.input_buffer = bytearray()
        self.bytes_per_sample = 4  # float32

    def downsample_48k_to_16k(self, audio_48k):
        """Downsample audio from 48kHz to 16kHz (3:1 ratio)"""
        return audio_48k[::3]

    def process_chunk(self, audio_chunk):
        """Process a chunk of audio through NovaSR"""
        try:
            audio_tensor = torch.FloatTensor(audio_chunk).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                output = self.model.model(audio_tensor)

            result = output.squeeze().cpu().numpy()
            return result.astype(np.float32)

        except Exception as e:
            print(f"[NovaSR] Processing error: {e}")
            return None

    def process(self, input_data):
        """Process input audio data (48kHz) and return output (48kHz)"""
        self.input_buffer.extend(input_data)

        needed_bytes = RESAMPLE_CHUNK_SIZE * self.bytes_per_sample

        if len(self.input_buffer) < needed_bytes:
            return None

        chunk_bytes = self.input_buffer[:needed_bytes]
        self.input_buffer = self.input_buffer[needed_bytes:]

        audio_48k = np.frombuffer(chunk_bytes, dtype=np.float32)
        audio_16k = self.downsample_48k_to_16k(audio_48k)

        output = self.process_chunk(audio_16k)
        return output

    def run(self):
        """Main processing loop"""
        print("[NovaSR] Processor ready")
        while True:
            try:
                data = self.input_queue.get(timeout=1.0)
                if data is None:  # Shutdown signal
                    break

                output = self.process(data)
                if output is not None:
                    self.output_queue.put(output.tobytes())

            except Empty:
                continue
            except Exception as e:
                print(f"[NovaSR] Error in processing loop: {e}")
                break


class AudioCaptureThread(threading.Thread):
    """Capture audio from virtual sink monitor"""

    def __init__(self, monitor_source, input_queue):
        super().__init__(daemon=True)
        self.monitor_source = monitor_source
        self.input_queue = input_queue
        self.running = True
        self.process = None

    def run(self):
        print(f"[Capture] Starting capture from {self.monitor_source}")

        try:
            self.process = subprocess.Popen(
                [
                    "parec",
                    "--format=float32le",
                    f"--rate={OUTPUT_RATE}",
                    "--channels=2",  # Capture stereo
                    "--latency-msec=50",
                    "-d", self.monitor_source
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=65536
            )

            while self.running:
                data = self.process.stdout.read(4096)
                if not data:
                    break

                # Convert stereo to mono (average channels)
                audio_array = np.frombuffer(data, dtype=np.float32)
                # Ensure we have an even number of samples for stereo
                if len(audio_array) % 2 != 0:
                    audio_array = audio_array[:-1]
                mono_audio = audio_array.reshape(-1, 2).mean(axis=1).astype(np.float32)
                self.input_queue.put(mono_audio.tobytes())

        except Exception as e:
            print(f"[Capture] Error: {e}")
        finally:
            if self.process:
                self.process.terminate()
            print("[Capture] Stopped")

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()


class AudioPlaybackThread(threading.Thread):
    """Play audio to physical sink using sink ID to avoid routing issues"""

    def __init__(self, physical_sink_id, output_queue, physical_sink_name):
        super().__init__(daemon=True)
        self.physical_sink_id = physical_sink_id
        self.physical_sink_name = physical_sink_name
        self.output_queue = output_queue
        self.running = True
        self.process = None

    def run(self):
        print(f"[Playback] Starting playback to sink ID {self.physical_sink_id} ({self.physical_sink_name})")

        try:
            # Pre-fill with silence to keep pacat alive
            silence = np.zeros(48000, dtype=np.float32).tobytes()  # 1 second of silence

            # Use -d with numeric ID to avoid routing to default sink
            self.process = subprocess.Popen(
                [
                    "pacat",
                    "--format=float32le",
                    f"--rate={OUTPUT_RATE}",
                    "--channels=1",
                    "--latency-msec=200",
                    "-d", str(self.physical_sink_id),
                    "--playback"
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=65536
            )

            # Write initial silence to start the stream
            self.process.stdin.write(silence)
            self.process.stdin.flush()
            print("[Playback] Started with silence buffer")

            # Now wait for real data
            while self.running:
                try:
                    data = self.output_queue.get(timeout=2.0)
                    if data is None:  # Shutdown signal
                        break
                    self.process.stdin.write(data)
                    self.process.stdin.flush()
                except Empty:
                    # Write some more silence to keep stream alive
                    self.process.stdin.write(np.zeros(24000, dtype=np.float32).tobytes())
                    self.process.stdin.flush()
                    # Check if process is still alive
                    if self.process.poll() is not None:
                        print("[Playback] Process died unexpectedly")
                        break
                    continue
                except (BrokenPipeError, OSError) as e:
                    print(f"[Playback] Pipe error: {e}")
                    break

        except Exception as e:
            print(f"[Playback] Error: {e}")
        finally:
            if self.process:
                self.process.terminate()
            print("[Playback] Stopped")

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()


class DeviceProcessor:
    """Manage audio processing for a single device"""

    def __init__(self, config):
        self.config = config
        self.sink_name = config["sink_name"]
        self.monitor_source = config["sink_name"] + ".monitor"
        self.physical_sink = config["physical_sink"]
        # Get the physical sink ID to avoid routing issues
        self.physical_sink_id = get_sink_id(self.physical_sink)
        if not self.physical_sink_id:
            raise ValueError(f"Could not find sink ID for {self.physical_sink}")
        self.description = config["sink_description"]

        self.input_queue = mp.Queue(maxsize=10)
        self.output_queue = mp.Queue(maxsize=10)
        self.capture_thread = None
        self.playback_thread = None
        self.processor_process = None
        self.active = False

    def start(self):
        """Start processing for this device"""
        if self.active:
            return

        print(f"[{self.description}] Starting audio processing...")

        # Start NovaSR processor in separate process
        self.processor_process = mp.Process(
            target=self._run_processor,
            args=(self.input_queue, self.output_queue)
        )
        self.processor_process.start()

        # Start capture and playback threads
        self.capture_thread = AudioCaptureThread(
            self.monitor_source, self.input_queue
        )
        self.playback_thread = AudioPlaybackThread(
            self.physical_sink_id, self.output_queue, self.physical_sink
        )

        self.capture_thread.start()
        self.playback_thread.start()

        self.active = True
        print(f"[{self.description}] Processing active")

    def stop(self):
        """Stop processing for this device"""
        if not self.active:
            return

        print(f"[{self.description}] Stopping audio processing...")

        # Signal shutdown
        try:
            self.input_queue.put(None, timeout=1)
        except:
            pass
        try:
            self.output_queue.put(None, timeout=1)
        except:
            pass

        # Stop threads
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2)
            self.capture_thread = None

        if self.playback_thread:
            self.playback_thread.stop()
            self.playback_thread.join(timeout=2)
            self.playback_thread = None

        # Stop processor process
        if self.processor_process:
            self.processor_process.terminate()
            self.processor_process.join(timeout=2)
            if self.processor_process.is_alive():
                self.processor_process.kill()
            self.processor_process = None

        # Clear queues to prevent old data
        try:
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
        except:
            pass
        try:
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
        except:
            pass

        self.active = False
        print(f"[{self.description}] Processing stopped")

    def _run_processor(self, input_queue, output_queue):
        """Run NovaSR processor in separate process"""
        try:
            processor = NovaSRProcessor(input_queue, output_queue)
            processor.run()
        except Exception as e:
            print(f"[{self.description}] Processor error: {e}")

    def restart(self):
        """Restart processing with fresh queues"""
        print(f"[{self.description}] Restarting audio processing...")
        self.stop()
        # Create fresh queues
        self.input_queue = mp.Queue(maxsize=10)
        self.output_queue = mp.Queue(maxsize=10)
        time.sleep(0.5)  # Brief pause to ensure cleanup
        self.start()


class NovaSRDaemon:
    """Main daemon managing all device processors"""

    def __init__(self):
        self.processors = {}
        self.running = True
        self.check_interval = 2.0  # Check every 2 seconds

    def get_active_sink(self):
        """Get the currently active/default sink"""
        try:
            result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"[Daemon] Error getting default sink: {e}")
            return None

    def monitor_and_process(self):
        """Always keep audio processing running and periodically ensure default sink"""
        print("[Daemon] Starting monitor loop...")

        # Start all processors immediately
        for name, config in VIRTUAL_SINKS.items():
            if name not in self.processors:
                self.processors[name] = DeviceProcessor(config)
                self.processors[name].start()
                print(f"[Daemon] Started processing for {config['sink_description']}")

        while self.running:
            try:
                # Periodically ensure the virtual sink is the default (every 10 seconds)
                default_sink = self.get_active_sink()
                for name, config in VIRTUAL_SINKS.items():
                    virtual_sink = config["sink_name"]
                    if default_sink != virtual_sink:
                        print(f"[Daemon] Re-enforcing default sink to {virtual_sink}")
                        set_default_sink(virtual_sink)
                        break

                time.sleep(self.check_interval)

            except Exception as e:
                print(f"[Daemon] Error in monitor loop: {e}")
                time.sleep(self.check_interval)

    def shutdown(self):
        """Shutdown all processors"""
        print("[Daemon] Shutting down...")
        self.running = False

        for name, processor in self.processors.items():
            if processor.active:
                processor.stop()

        # Remove virtual sinks
        remove_all_virtual_sinks()

        print("[Daemon] Shutdown complete")

    def run(self):
        """Run the daemon"""
        print("=" * 60)
        print("NovaSR Enhanced Audio Daemon")
        print("=" * 60)
        print(f"Device: {DEVICE}")
        print(f"Processing: {NOVASR_RATE}Hz -> {OUTPUT_RATE}Hz")
        print()
        print("Virtual Devices:")
        for name, config in VIRTUAL_SINKS.items():
            print(f"  - {config['sink_description']}")
            print(f"    Virtual: {config['sink_name']}")
            print(f"    Physical: {config['physical_sink']}")
        print("=" * 60)
        print()

        # Create virtual sinks
        if not create_all_virtual_sinks():
            print("[Daemon] Failed to create virtual sinks!")
            sys.exit(1)

        # Handle signals
        def signal_handler(sig, frame):
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            self.monitor_and_process()
        except Exception as e:
            print(f"[Daemon] Fatal error: {e}")
            self.shutdown()
            sys.exit(1)


def main():
    """Main entry point"""
    daemon = NovaSRDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
