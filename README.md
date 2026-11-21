# WhisperOverlay

WhisperOverlay is a Windows desktop app that reads video from a file or RTSP
source, transcribes its audio with faster-whisper, and overlays the current
caption on an H.264 output file or RTSP stream.

## Requirements

- Windows 10 or later
- A working internet connection the first time a faster-whisper model is used;
  faster-whisper downloads the selected model

To build or run from source, also install Python 3.10 or later and FFmpeg on
`PATH` (the FFmpeg build must include `libx264` and `drawtext`). The packaged
executable bundles FFmpeg and does not require Python on the target PC.

## Build the single-file Windows app

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-build.txt
# Install an FFmpeg build with libx264 and drawtext, then:
$env:FFMPEG_EXE = "C:\path\to\ffmpeg.exe"
.\build_windows.ps1
```

The standalone executable is `dist\WhisperOverlay.exe`. FFmpeg and the Python
runtime/dependencies are bundled into that single file; the target PC does not
need Python or FFmpeg installed. When opened normally, the app hides its
standalone console window; headless command-line runs retain the console.

The faster-whisper model is downloaded on first use and cached in the user's
local Hugging Face cache, so the first run needs internet access. Models are
large and are intentionally not embedded in the executable.
In the desktop app, select a model to see whether it is already cached, or use
**Download model** to fetch it before starting a stream. The progress bar shows
the percentage downloaded for the current model file.

To run from source during development, install `requirements.txt` and run
`python app.py` (with FFmpeg available on `PATH`).

## Headless command-line mode

The same executable can run without opening the desktop UI:

```powershell
.\dist\WhisperOverlay.exe --headless `
  --input-kind FILE --input "C:\video\input.mp4" `
  --output-kind RTSP --output "rtsp://server/live" `
  --model small --audio-mode aac --loop-file
```

Use `--input-kind RTSP` and an RTSP URL for live input. `--loop-file` is only
valid with FILE input. Audio mode is `aac` (256-kbit/s AAC) or `copy` (pass
through). Stop a headless run with Ctrl+C. Run with `--help` to see all options.

Select FILE or RTSP for each endpoint. File output is written using the
container inferred from its extension (MP4 and MKV are offered by the file
picker). RTSP destinations must be publishable by FFmpeg, such as a path on an
RTSP media server.

For FILE input, enable **Loop file indefinitely (real-time)** to repeat the
source until you stop the pipeline. This option is disabled for RTSP inputs.

Video is encoded as H.264 using `libx264`, the `ultrafast` preset, and
`zerolatency`. Audio can either be passed through unchanged or encoded as
256-kbit/s AAC. Use pass-through when the input codec is supported by the
chosen output container/server.

The app reads the source once and routes a PCM audio copy from that same FFmpeg
session to faster-whisper. RTSP cameras need only one client connection. Audio
is transcribed in five-second chunks; if inference falls behind, older queued
chunks are discarded to keep captions near live. The caption represents the
most recently transcribed chunk.
