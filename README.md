# NovaSR Enhanced Audio for KDE Plasma

System-wide audio enhancement for KDE Plasma on Linux using PipeWire/PulseAudio. Automatically upscales audio from 16kHz to 48kHz using the NovaSR neural network model.

## How It Works

```
KDE Audio Selection
     ↓
[Speakers (NovaSR_Enhanced_Speakers)]  ← You select this in KDE
     ↓
[Virtual Sink Capture]
     ↓
[NovaSR Processing: 16kHz → 48kHz]
     ↓
[Enhanced Audio → Physical Speakers]
```

## Quick Start

1. **Enable and start the service:**
   ```bash
   systemctl --user enable --now novasr-enhancer
   ```

2. **Select the enhanced device in KDE:**
   - Open KDE Audio Settings (right-click volume icon)
   - Under "Output Devices", select **"NovaSR_Enhanced_Speakers"**
   - Done! All audio is now enhanced

3. **Service auto-manages processing:**
   - When you select the NovaSR device → processing starts
   - When you select another device → processing stops
   - Runs automatically in background

## System Requirements

- Linux with PipeWire (Arch, Fedora, Bazzite, etc.)
- KDE Plasma 6 (or other Wayland compositors)
- Python 3.14+
- Working audio output device(s)

## Project Structure

```
~/.local/share/nova-sr-enhancer/
├── daemon/
│   ├── novasr_enhanced_daemon.py   # Main daemon (monitors & processes audio)
│   └── start_enhanced.sh            # Service launch script
├── src/
│   ├── novasr_processor.py          # Thread-safe NovaSR wrapper
│   ├── thread_manager.py            # CPU affinity & thread pools
│   ├── audio_utils.py               # Audio I/O utilities
│   ├── batch_processor.py           # Batch file processing
│   └── live_enhancer.py             # Real-time enhancement (standalone)
├── bin/
│   ├── nova-sr-batch                # Batch processor CLI
│   ├── nova-sr-live                 # Live enhancer CLI
│   └── nova-sr-start                # Interactive launcher
├── models/                          # Cached NovaSR model
├── lib/ & bin/                      # Python virtual environment
└── README.md                        # This file
```

## Configuration

The daemon is configured in `daemon/novasr_enhanced_daemon.py`:

```python
VIRTUAL_SINKS = {
    "novasr_speakers": {
        "sink_name": "novasr_speakers",
        "sink_description": "NovaSR_Enhanced_Speakers",
        "physical_sink": "alsa_output.pci-0000_c1_00.6.HiFi__Speaker__sink",
    }
}
```

To add more devices (headphones, SPDIF, etc.), add entries to `VIRTUAL_SINKS`.

## Service Commands

```bash
# Check status
systemctl --user status novasr-enhancer

# View logs
journalctl --user -u novasr-enhancer -f

# Restart service
systemctl --user restart novasr-enhancer

# Stop service
systemctl --user stop novasr-enhancer
```

## Batch Processing (Downloaded Lectures)

For processing audio/video files offline:

```bash
# Single file
~/.local/share/nova-sr-enhancer/bin/nova-sr-batch lecture.mp4 -o enhanced.mp4

# Directory (parallel processing)
~/.local/share/nova-sr-enhancer/bin/nova-sr-batch ~/Lectures/ -o ~/Lectures_Enhanced/ -j 4

# Supported formats: MP4, MP3, M4A, WEBM, MKV, WAV, FLAC, OGG
```

## Architecture Details

### Daemon (System Service)

- **Multi-process design**: Each device has isolated processing
- **Auto-detection**: Monitors `pactl get-default-sink` to detect device selection
- **Processing pipeline**: `virtual_sink → parec → NovaSR → paplay → physical_sink`
- **CPU optimization**: Uses multiprocessing for NovaSR inference

### Virtual Sinks

Created using `pactl load-module module-null-sink`:
- `novasr_speakers` - Virtual sink that apps output to
- `.monitor` - Capture source for the daemon
- Routes to physical device after enhancement

### NovaSR Model

- **Size**: 52KB
- **Input**: 16kHz audio (downsampled from 48kHz)
- **Output**: 48kHz enhanced audio (3x upsampling)
- **Source**: https://github.com/ysharma3501/NovaSR
- **Cached in**: `~/.cache/huggingface/hub/models--YatharthS--NovaSR/`

## Performance

- **Latency**: ~200-300ms (acceptable for lectures, videos)
- **CPU Usage**: ~40% on 6-core system (1 core for NovaSR, others for I/O)
- **Memory**: ~300MB (PyTorch model + processing buffers)

## Troubleshooting

### NovaSR device not showing in KDE

1. Check service is running:
   ```bash
   systemctl --user status novasr-enhancer
   ```

2. Check virtual sink exists:
   ```bash
   pactl list short sinks | grep novasr
   ```

3. Restart service:
   ```bash
   systemctl --user restart novasr-enhancer
   ```

### No sound enhancement

1. Verify the NovaSR device is selected as default sink:
   ```bash
   pactl get-default-sink
   # Should return: novasr_speakers
   ```

2. Check logs for processing activity:
   ```bash
   journalctl --user -u novasr-enhancer -f
   # Look for: "[Speakers (NovaSR Enhanced)] Starting audio processing..."
   ```

### High CPU usage

Normal behavior during audio playback. NovaSR runs on CPU (no CUDA).

### Audio glitches

Increase buffer sizes in `novasr_enhanced_daemon.py`:
- `CHUNK_SIZE = 9600` → increase to `19200`

## Dependencies

Installed in virtual environment:
```bash
# Core
torch>=2.0.0
torchaudio
einops

# Audio
pyaudio
soundfile
scipy
numpy

# NovaSR
git+https://github.com/ysharma3501/NovaSR.git
```

System packages:
```bash
# Arch
pacman -S python-pip python-virtualenv pulseaudio pipewire

# Fedora
dnf install python3-pip python3-virtualenv pulseaudio pipewire
```

## Credits

- **NovaSR**: https://github.com/ysharma3501/NovaSR
- **Original inspiration**: Similar setup on another system

## License

MIT
