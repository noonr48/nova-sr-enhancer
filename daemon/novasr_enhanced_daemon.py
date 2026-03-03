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
import hashlib
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

# Prefix for our virtual sinks (to identify them)
NOVASR_PREFIX = "novasr_enhanced_"

# Sinks to ignore (virtual sinks, monitors, etc.)
IGNORE_PATTERNS = ["novasr_", ".monitor", "auto_null"]


def get_physical_sinks():
    """Get list of physical output sinks (excluding our virtual sinks)"""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            check=True
        )

        sinks = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                sink_name = parts[1]
                # Skip sinks that match ignore patterns
                if any(pattern in sink_name for pattern in IGNORE_PATTERNS):
                    continue
                sinks.append({
                    "id": parts[0],
                    "name": sink_name,
                })

        return sinks
    except Exception as e:
        print(f"[Setup] Error getting physical sinks: {e}")
        return []


def generate_virtual_sink_config(physical_sink):
    """Generate a virtual sink config for a physical sink"""
    name = physical_sink["name"]

    # Create a friendly description and short name
    if "HiFi" in name:
        if "Headphones" in name:
            friendly = "Headphones"
        elif "Speaker" in name:
            friendly = "Speakers"
        else:
            friendly = "HiFi"
    elif "hdmi" in name.lower():
        friendly = "HDMI"
    elif "usb" in name.lower():
        friendly = "USB_Audio"
    elif "bluez" in name.lower() or "bluetooth" in name.lower():
        friendly = "Bluetooth"
    else:
        friendly = name.split(".")[-1][:12]

    # Create unique but short sink name using hash of physical name
    name_hash = hashlib.md5(name.encode()).hexdigest()[:4]
    sink_name = f"novasr_{friendly.lower()}_{name_hash}"

    return {
        "sink_name": sink_name,
        "sink_description": f"NovaSR_{friendly}",
        "physical_sink": physical_sink["name"],
        "physical_sink_id": physical_sink["id"],
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
    """Create virtual sinks for all current physical sinks"""
    print("[Setup] Scanning for physical sinks...")
    physical_sinks = get_physical_sinks()
    print(f"[Setup] Found {len(physical_sinks)} physical sink(s)")

    configs = {}
    for sink in physical_sinks:
        config = generate_virtual_sink_config(sink)
        configs[config["sink_name"]] = config
        if create_virtual_sink(config["sink_name"], config["sink_description"]):
            print(f"[Setup] Created enhanced sink for: {sink['name']}")

    return configs


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
    """Remove all NovaSR virtual sinks"""
    print("[Cleanup] Removing all NovaSR virtual sinks...")
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2 and NOVASR_PREFIX in parts[1]:
                remove_virtual_sink(parts[1])
    except Exception as e:
        print(f"[Cleanup] Error removing virtual sinks: {e}")


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
        self.description = config["sink_description"]
        # Queues created fresh in start()
        self.input_queue = None
        self.output_queue = None
        self.capture_thread = None
        self.playback_thread = None
        self.processor_process = None
        self.active = False

    def get_physical_sink_id(self):
        """Get the physical sink ID (look up fresh each time)"""
        return get_sink_id(self.physical_sink)

    def start(self):
        """Start processing for this device. Returns True on success."""
        if self.active:
            return True

        # Get fresh sink ID each time (can change after PipeWire restart)
        physical_sink_id = self.get_physical_sink_id()
        if not physical_sink_id:
            print(f"[{self.description}] Physical sink not found: {self.physical_sink}")
            return False

        print(f"[{self.description}] Starting audio processing...")

        # Create fresh queues
        self.input_queue = mp.Queue(maxsize=10)
        self.output_queue = mp.Queue(maxsize=10)

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
            physical_sink_id, self.output_queue, self.physical_sink
        )

        self.capture_thread.start()
        self.playback_thread.start()

        self.active = True
        print(f"[{self.description}] Processing active")
        return True

    def stop(self):
        """Stop processing for this device"""
        if not self.active:
            return

        print(f"[{self.description}] Stopping audio processing...")

        # Signal shutdown
        if self.input_queue:
            try:
                self.input_queue.put(None, timeout=1)
            except:
                pass
        if self.output_queue:
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

        # Clear queues
        self.input_queue = None
        self.output_queue = None
        self.active = False
        print(f"[{self.description}] Processing stopped")

    def _run_processor(self, input_queue, output_queue):
        """Run NovaSR processor in separate process"""
        try:
            processor = NovaSRProcessor(input_queue, output_queue)
            processor.run()
        except Exception as e:
            print(f"[{self.description}] Processor error: {e}")


class NovaSRDaemon:
    """Main daemon managing all device processors with dynamic device detection"""

    def __init__(self):
        self.processors = {}
        self.virtual_configs = {}  # Maps virtual sink name -> config
        self.running = True
        self.check_interval = 2.0  # Check every 2 seconds
        self.device_scan_interval = 5.0  # Scan for new devices every 5 seconds
        self.last_device_scan = 0

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

    def wait_for_audio_system(self, timeout=30):
        """Wait for PipeWire/PulseAudio to be ready"""
        print("[Setup] Waiting for audio system...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    ["pactl", "list", "short", "sinks"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Check if we have at least one non-virtual sink
                sinks = [line for line in result.stdout.strip().split('\n') if line.strip()]
                real_sinks = [s for s in sinks if NOVASR_PREFIX not in s]
                if real_sinks:
                    print(f"[Setup] Audio system ready ({len(real_sinks)} physical sink(s) found)")
                    return True
            except:
                pass

            time.sleep(1)

        print("[Setup] Warning: Audio system not ready after timeout, proceeding anyway")
        return False

    def scan_and_update_devices(self):
        """Scan for physical devices and update virtual sinks accordingly"""
        physical_sinks = get_physical_sinks()
        current_physical_names = {s["name"] for s in physical_sinks}

        # Get current physical sinks we have configs for
        known_physical = {c["physical_sink"] for c in self.virtual_configs.values()}

        # Find new devices
        new_devices = []
        for sink in physical_sinks:
            if sink["name"] not in known_physical:
                new_devices.append(sink)

        # Find removed devices
        removed_devices = []
        for physical_name in known_physical:
            if physical_name not in current_physical_names:
                removed_devices.append(physical_name)

        # Add new devices
        for sink in new_devices:
            config = generate_virtual_sink_config(sink)
            if create_virtual_sink(config["sink_name"], config["sink_description"]):
                self.virtual_configs[config["sink_name"]] = config
                self.processors[config["sink_name"]] = DeviceProcessor(config)
                print(f"[Daemon] NEW DEVICE: Created enhanced sink for {sink['name']}")

        # Remove disconnected devices
        for physical_name in removed_devices:
            # Find the config for this physical device
            for vname, config in list(self.virtual_configs.items()):
                if config["physical_sink"] == physical_name:
                    # Stop processing if active
                    if vname in self.processors:
                        if self.processors[vname].active:
                            self.processors[vname].stop()
                        del self.processors[vname]
                    # Remove virtual sink
                    remove_virtual_sink(vname)
                    del self.virtual_configs[vname]
                    print(f"[Daemon] DEVICE REMOVED: {physical_name}")
                    break

        return len(new_devices) > 0 or len(removed_devices) > 0

    def stop_all_processors(self):
        """Stop all active processors"""
        for vname, processor in self.processors.items():
            if processor.active:
                processor.stop()

    def monitor_and_process(self):
        """Monitor sink selection and device changes - only ONE processor active at a time"""
        print("[Daemon] Starting monitor loop...")
        print("[Daemon] Select any 'NovaSR_*' device in KDE to enable enhancement")
        print("[Daemon] Select any other device to disable enhancement")
        print("[Daemon] Hot-plugging enabled: new devices detected automatically")

        # Do initial device scan
        self.scan_and_update_devices()

        last_sink = None
        last_active = None

        while self.running:
            try:
                current_time = time.time()

                # Periodically scan for new devices
                if current_time - self.last_device_scan >= self.device_scan_interval:
                    self.scan_and_update_devices()
                    self.last_device_scan = current_time

                current_sink = self.get_active_sink()

                # Log sink changes
                if current_sink != last_sink:
                    print(f"[Daemon] Default sink: {current_sink}")
                    last_sink = current_sink

                # Check if current sink is one of our virtual sinks
                active_virtual = None
                for vname, config in self.virtual_configs.items():
                    if config["sink_name"] == current_sink:
                        active_virtual = vname
                        break

                # Stop previous processor if different (ensures only ONE active)
                if last_active and last_active != active_virtual:
                    if last_active in self.processors and self.processors[last_active].active:
                        print(f"[Daemon] Stopping {last_active}")
                        self.processors[last_active].stop()
                    last_active = None

                # Start new processor if needed (with retry if physical sink not ready)
                if active_virtual and active_virtual != last_active:
                    if active_virtual in self.processors and not self.processors[active_virtual].active:
                        print(f"[Daemon] Starting {active_virtual} (selected)")
                        if self.processors[active_virtual].start():
                            last_active = active_virtual
                        else:
                            print(f"[Daemon] Failed to start {active_virtual}, will retry...")

                time.sleep(self.check_interval)

            except Exception as e:
                print(f"[Daemon] Error in monitor loop: {e}")
                time.sleep(self.check_interval)

    def shutdown(self):
        """Shutdown all processors"""
        print("[Daemon] Shutting down...")
        self.running = False

        self.stop_all_processors()

        # Remove virtual sinks
        remove_all_virtual_sinks()

        print("[Daemon] Shutdown complete")

    def run(self):
        """Run the daemon"""
        print("=" * 60)
        print("NovaSR Enhanced Audio Daemon (Dynamic Mode)")
        print("=" * 60)
        print(f"Device: {DEVICE}")
        print(f"Processing: {NOVASR_RATE}Hz -> {OUTPUT_RATE}Hz")
        print("=" * 60)
        print()

        # Wait for audio system to be ready
        self.wait_for_audio_system()

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
