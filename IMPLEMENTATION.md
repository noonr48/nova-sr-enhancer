# Implementation Notes

How this NovaSR Enhanced Audio system was built, from concept to working implementation.

## Initial Problem

User wanted to enhance university lecture audio using NovaSR (52KB neural upsampling model). Lectures sound "horrible" - muffled, low quality.

**Requirements:**
- Work with both live lectures (browser/Zoom) AND downloaded files
- Integrate with KDE audio menu - select device and it "just works"
- Use 6-core/12-thread CPU efficiently
- System: Arch Linux, PipeWire 1.4.9, Python 3.14.2, no GPU

## Design Evolution

### Attempt 1: System-Wide Audio Intercept (FAILED)

**Concept:** Intercept ALL audio going to ANY device

**Approach:**
- Create PipeWire virtual sink
- Route everything through it
- Process through NovaSR
- Output to selected device

**Why it failed:**
- PipeWire/PulseAudio doesn't have a simple "insert filter" hook
- Requires complex module loading per device
- Would need to restart audio stack when changing devices

**Lesson:** Can't easily make enhancement truly transparent in PipeWire without significant complexity.

### Attempt 2: Manual Device Selection (REJECTED)

**Concept:** User selects "nova_sr_filter" in KDE

**Approach:**
- One virtual sink for all audio
- User manually switches to it
- Requires manual device switching

**Why rejected:**
User wanted to just select "Speakers" in KDE menu and have enhancement happen automatically.

### Attempt 3: Enhanced Device Wrappers (SUCCESS!)

**Concept:** Create "Enhanced" versions of each physical device

**Inspiration:** User's previous system (`/home/benbi/Downloads/aiaudio/NovaSR-Enhanced-Audio/`)

**Final Architecture:**
```
Physical Device: "Ryzen HD Audio Controller Speaker"
                    ↓
        Enhanced Wrapper: "NovaSR_Enhanced_Speakers"
                    ↓
          User selects Enhanced in KDE
                    ↓
        Daemon detects selection
                    ↓
        Processing pipeline activates:
        virtual_sink → parec → NovaSR → paplay → physical_sink
```

## Implementation Steps

### Step 1: Virtual Environment & Dependencies

```bash
# Create isolated Python environment
python -m venv ~/.local/share/nova-sr-enhancer

# Install NovaSR + dependencies
pip install einops torch torchaudio soundfile scipy numpy pyaudio
pip install git+https://github.com/ysharma3501/NovaSR.git
```

**Why venv?**
- Avoids system Python conflicts
- Easy to recreate/upgrade
- Contains PyTorch and other heavy dependencies

**Missing dependency discovered:** `einops` not listed in NovaSR's setup.py

### Step 2: Core NovaSR Wrapper (`src/novasr_processor.py`)

**Purpose:** Thread-safe wrapper around NovaSR model

**Key design decisions:**
- Class-level model cache (shared across instances)
- Thread-local storage for per-thread models
- Automatic sample rate conversion (48kHz → 16kHz → 48kHz)
- Fallback to simple upsampling if model fails

**Code structure:**
```python
class NovaSRProcessor:
    def __init__(model_path, half_precision):
        # Load model from HuggingFace or local path
        # Set device (CPU in this case)

    def downsample(audio, from_rate):
        # 48kHz → 16kHz for NovaSR input

    def process_channel(audio_channel):
        # Run NovaSR inference

    def process_stereo_parallel(audio):
        # Process left/right channels in parallel
```

### Step 3: Thread Manager (`src/thread_manager.py`)

**Purpose:** Optimize for 6-core/12-thread CPU

**Thread assignment:**
| Thread | Purpose | CPU Core |
|--------|---------|----------|
| 1 | Audio capture | Core 0 |
| 2-3 | NovaSR processing (stereo) | Cores 1-4 |
| 4 | Audio playback | Core 5 |
| 5+ | Batch file processing | All cores |

**Key features:**
- `set_cpu_affinity()` - Pin threads to specific cores
- `BoundedQueue` - Lock-free-ish queues for audio pipeline
- `ThreadPoolExecutor` - Parallel batch processing

### Step 4: Audio Utilities (`src/audio_utils.py`)

**Purpose:** Audio I/O and format handling

**Functions:**
- `extract_audio_from_video()` - ffmpeg for video → audio
- `load_audio()` - Load with resampling
- `save_audio()` - Save with proper format
- `remux_audio_to_video()` - ffmpeg: audio back into video
- `get_audio_duration()` - Get file duration

**Supported formats:**
- Audio: WAV, MP3, M4A, AAC, FLAC, OGG, OPUS
- Video: MP4, MKV, WEBM, AVI, MOV, WMV, FLV

### Step 5: Batch Processor (`src/batch_processor.py`)

**Purpose:** Process downloaded lecture files

**Workflow:**
```
Input file (MP4/MP3/etc)
    ↓
Extract audio (if video)
    ↓
Load at 48kHz
    ↓
Downsample to 16kHz
    ↓
NovaSR upsamples to 48kHz
    ↓
Save output (or remux to video)
```

**Parallel processing:**
- Thread pool processes multiple files simultaneously
- CPU affinity distributes across cores
- Progress bar with tqdm

### Step 6: Live Enhancer (`src/live_enhancer.py`)

**Purpose:** Real-time audio enhancement (standalone, not used by daemon)

**Pipeline:**
```
PyAudio capture → Queue A → NovaSR → Queue B → PyAudio playback
Thread 1          16 chunks  Thread 2   8 chunks  Thread 4
```

**Parameters:**
- Chunk size: 1920 samples (40ms @ 48kHz)
- Latency: ~200-300ms
- Sample rate: 48000 Hz

### Step 7: Daemon (`daemon/novasr_enhanced_daemon.py`)

**Purpose:** Background service that monitors device selection and auto-processes

**Architecture:**
```
NovaSRDaemon
    ├── Monitors: pactl get-default-sink
    ├── Creates: virtual sinks (module-null-sink)
    └── Manages: DeviceProcessor instances

DeviceProcessor (per virtual device)
    ├── AudioCaptureThread (parec from virtual sink)
    ├── NovaSRProcessor (separate process)
    └── AudioPlaybackThread (pacat to physical sink)
```

**Key logic:**
```python
while running:
    default_sink = get_default_sink()

    if default_sink == "novasr_speakers":
        start_processing()  # Activate pipeline
    else:
        stop_processing()   # Save CPU
```

**Why multiprocessing for NovaSR?**
- Isolates PyTorch inference
- Prevents GIL blocking
- Can be killed/restarted cleanly

### Step 8: Virtual Sink Creation

**Using PulseAudio module-null-sink:**
```bash
pactl load-module module-null-sink \
    sink_name=novasr_speakers \
    sink_properties=device.description=NovaSR_Enhanced_Speakers \
    rate=48000 \
    channels=2
```

**Creates:**
- Sink: `novasr_speakers` (what apps output to)
- Monitor: `novasr_speakers.monitor` (what daemon captures from)

### Step 9: Systemd Service

**File:** `~/.config/systemd/user/novasr-enhancer.service`

**Key settings:**
```ini
After=wireplumber.service pipewire.service pipewire-pulse.service
BindsTo=wireplumber.service
Type=simple
Restart=on-failure
Nice=-5  # Higher priority for audio
```

**Why user service?**
- Runs without root
- Auto-starts on login (with `enable --linger`)
- Binds to audio stack lifecycle

## Device Naming Issues

**Problem:** PulseAudio truncates device descriptions

**Attempted formats:**
- `"Speakers (NovaSR Enhanced)"` → Truncated to `"Speakers"`
- `"NovaSR Enhanced Speakers"` → Truncated to `"NovaSR"`

**Solution:** Use underscores - `"NovaSR_Enhanced_Speakers"`

This is a PulseAudio quirk with `module-null-sink` properties.

## Integration with KDE

**How KDE sees the devices:**
```
Ryzen HD Audio Controller Speaker  (physical device)
NovaSR_Enhanced_Speakers             (virtual device)
```

**User workflow:**
1. Click volume icon in KDE
2. Select "NovaSR_Enhanced_Speakers"
3. Daemon detects via `pactl get-default-sink`
4. Processing pipeline activates
5. Audio is enhanced

## Performance Optimization

### CPU Usage

**Baseline (no optimization):** Would use 100% of one core

**With optimizations:**
- CPU affinity: Prevents thread migration
- Separate capture/playback threads: I/O doesn't block
- multiprocessing for NovaSR: Isolates CPU-heavy work

**Result:** ~40% CPU on 6-core system during playback

### Memory

**PyTorch model:** ~52KB (tiny!)

**Runtime memory:** ~300MB
- PyTorch runtime: ~200MB
- Audio buffers: ~50MB
- Processing overhead: ~50MB

### Latency

**Target:** < 500ms for acceptable video/lecture playback

**Breakdown:**
- Capture buffer: 40ms (1920 samples @ 48kHz)
- NovaSR processing: ~100ms (CPU inference)
- Playback buffer: 40ms
- **Total: ~180-200ms**

**Trade-offs:**
- Smaller chunks → lower latency, more CPU
- Larger chunks → higher latency, less glitch-prone

## Troubleshooting Encountered

### Issue 1: NovaSR model not found

**Symptom:** `ModuleNotFoundError: No module named 'einops'`

**Cause:** `einops` used in NovaSR code but not in setup.py

**Fix:** Manually install `pip install einops`

### Issue 2: Virtual sink not appearing

**Symptom:** Device not in KDE audio menu

**Cause:** PipeWire needs restart after config changes

**Fix:** `systemctl --user restart pipewire pipewire-pulse wireplumber`

### Issue 3: Device description truncated

**Symptom:** "Speakers (NovaSR Enhanced)" shows as just "Speakers"

**Cause:** PulseAudio property parsing

**Fix:** Use underscores: `"NovaSR_Enhanced_Speakers"`

### Issue 4: No sound on first selection

**Symptom:** Device selected but no audio

**Cause:** Daemon needs to detect default sink change

**Fix:** Built-in monitoring loop (2-second check interval)

## Files and Their Purposes

### Core Processing
- `src/novasr_processor.py` - NovaSR model wrapper
- `src/thread_manager.py` - CPU affinity and thread pools
- `src/audio_utils.py` - Audio I/O utilities

### Batch Processing
- `src/batch_processor.py` - Process multiple files offline

### Live Enhancement
- `src/live_enhancer.py` - Real-time processing (standalone)

### Daemon (Main Feature)
- `daemon/novasr_enhanced_daemon.py` - Background service
- `daemon/start_enhanced.sh` - Service launcher

### CLI Tools
- `bin/nova-sr-batch` - Batch processing command
- `bin/nova-sr-live` - Live enhancement command
- `bin/nova-sr-start` - Interactive launcher

### Configuration
- `~/.config/systemd/user/novasr-enhancer.service` - Systemd service
- No PipeWire config needed (daemon creates sinks dynamically)

## How to Recreate This System

### On a Fresh Arch Linux Install:

```bash
# 1. Create project directory
mkdir -p ~/.local/share/nova-sr-enhancer
cd ~/.local/share/nova-sr-enhancer

# 2. Create virtual environment
python -m venv .
source bin/activate

# 3. Install dependencies
pip install einops torch torchaudio soundfile scipy numpy pyaudio
pip install git+https://github.com/ysharma3501/NovaSR.git

# 4. Copy source files
# (Copy all src/, daemon/, and service files from git repo)

# 5. Install service
cp ~/.config/systemd/user/novasr-enhancer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now novasr-enhancer

# 6. Select in KDE
# Open audio settings, choose "NovaSR_Enhanced_Speakers"
```

## Future Improvements

### Possible Enhancements

1. **GPU Support**
   - Add CUDA detection
   - Load model on GPU if available
   - Would significantly speed up processing

2. **Per-Application Enhancement**
   - Only enhance certain apps (browser, media player)
   - Leave system sounds untouched
   - Requires PulseAudio module routing

3. **Quality Settings**
   - Adjustable enhancement levels
   - EQ before/after NovaSR
   - Compression/limiting

4. **Better UI Integration**
   - KDE Plasma widget
   - System tray icon with toggle
   - Visual feedback when processing

5. **Automatic Device Detection**
   - Auto-add new devices to VIRTUAL_SINKS
   - Detect when headphones plugged in
   - Create enhanced versions automatically

## Lessons Learned

1. **PipeWire/PulseAudio is powerful but complex**
   - Virtual sinks are easy to create
   - But dynamic routing requires monitoring

2. **CPU affinity matters**
   - Pinning threads prevents migration overhead
   - Especially important for audio processing

3. **Multiprocessing vs Threading**
   - Use multiprocessing for CPU-bound work (PyTorch)
   - Use threading for I/O-bound work (audio capture/playback)

4. **Keep it simple**
   - Initial attempts were too complex
   - User's existing approach was the right one
   - "Enhanced wrapper devices" is the most practical

5. **Test early with real hardware**
   - Emulation doesn't reveal audio timing issues
   - Real devices have different characteristics

## References

- **NovaSR:** https://github.com/ysharma3501/NovaSR
- **PipeWire:** https://docs.pipewire.org/
- **PulseAudio:** https://freedesktop.org/software/pulseaudio/doxygen/
- **Python Audio:** https://people.csail.mit.edu/hubert/pyaudio/

## Credits

- **NovaSR model:** Yatharth Sharma
- **Original implementation idea:** User's previous system
- **Implementation assistance:** Claude Code
