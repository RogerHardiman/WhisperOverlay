"""
Realtime subtitles for an RTSP stream.

  RTSP source --> ffmpeg --+--> video (drawtext overlay, re-encoded) + audio --> RTSP server (e.g. mediamtx)
                           |
                           +--> 16 kHz mono PCM --> faster-whisper --> subtitle text file
                                                                            ^
                                      ffmpeg's drawtext re-reads this file every frame (reload=1)

Setup:
    pip install -U faster-whisper numpy
    # ffmpeg on PATH (built with libx264 and freetype/drawtext, which standard builds have)
    # a running mediamtx (https://github.com/bluenviron/mediamtx), default RTSP port 8554

The source can be an RTSP URL or a local video file. A file is read at real-time
speed (ffmpeg -re) and plays once by default. Use --loop to repeat it continuously.

Usage:
    python rtsp_live_subtitles.py rtsp://camera/stream rtsp://localhost:8554/subtitled
    python rtsp_live_subtitles.py video.mp4 rtsp://localhost:8554/subtitled
    python rtsp_live_subtitles.py video.mp4 OUT                  # play the file once
    python rtsp_live_subtitles.py video.mp4 OUT --loop           # loop the file
    python rtsp_live_subtitles.py IN OUT --model small.en --device cuda
    python rtsp_live_subtitles.py IN OUT --encoder h264_nvenc

Watch:
    ffplay rtsp://localhost:8554/subtitled
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

SAMPLE_RATE = 16000
CHUNK_SEC = 0.1
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_SEC) * 2  # s16le mono


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for timestamped transcripts."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def write_timestamped_transcript(path: Path, segments) -> int:
    """Write segment timestamps and text, returning the number of segments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as transcript:
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            start = format_timestamp(float(segment.start))
            end = format_timestamp(float(segment.end))
            transcript.write(f"[{start} --> {end}] {text}\n")
            count += 1
    return count


def default_font() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    return next((c for c in candidates if Path(c).exists()), "")


def is_rtsp(src: str) -> bool:
    return src.lower().startswith(("rtsp://", "rtsps://"))


def ffmpeg_escape(path: str) -> str:
    """Escape a path for use inside a filtergraph option value."""
    return path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


class SubtitleFile:
    """Atomically updated text file that ffmpeg's drawtext reloads every frame."""

    def __init__(self, path: Path, width: int, show_overlay: bool = False):
        self.path, self.width, self.current = path, width, None
        self.show_overlay = show_overlay
        self.set("")

    def set(self, text: str):
        text = "\n".join(textwrap.wrap(text.strip(), self.width)[-2:]) or " "  # last 2 lines only
        if text == self.current:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: drawtext never sees a half-written file
        self.current = text
        if self.show_overlay and text != " ":
            print(text, flush=True)


def build_ffmpeg_cmd(args, subfile: Path) -> list:
    drawtext = (
        f"drawtext=textfile='{ffmpeg_escape(str(subfile))}':reload=1:expansion=none"
        f":fontcolor=white:fontsize={args.font_size}:borderw=3:bordercolor=black"
        f":x=(w-text_w)/2:y=h-text_h-{args.margin}:line_spacing=6"
    )
    if args.font:
        drawtext += f":fontfile='{ffmpeg_escape(args.font)}'"

    venc = ["-c:v", args.encoder, "-pix_fmt", "yuv420p", "-g", str(args.gop), "-b:v", args.bitrate]
    if args.encoder == "libx264":
        venc += ["-preset", "veryfast", "-tune", "zerolatency"]

    if is_rtsp(args.input):
        # Live source: low-latency flags
        input_opts = ["-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay"]
    else:
        # File source: -re paces reading at native frame rate (real time);
        # -stream_loop -1 repeats forever with continuous timestamps.
        input_opts = ["-re"]
        if args.loop:
            input_opts += ["-stream_loop", "-1"]

    cmd = [
        "ffmpeg", "-loglevel", "warning", "-nostats",
    ]
    if not is_rtsp(args.output) and not is_rtsp(args.input) and not args.loop:
        cmd += ["-progress", "pipe:2"]
    cmd += [
        *input_opts,
        "-i", args.input,
        # Output 1: overlaid video + audio
        "-map", "0:v:0", "-map", "0:a:0?", "-vf", drawtext, *venc,
        "-c:a", "aac", "-b:a", "128k",
    ]

    if is_rtsp(args.output):
        cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", args.output]
    else:
        cmd += ["-y"]
        if Path(args.output).suffix.lower() in (".mp4", ".mov", ".m4v"):
            cmd += ["-movflags", "+faststart"]
        cmd += [args.output]

    cmd += [
        # Output 2: raw audio to stdout for Whisper
        "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
    ]
    return cmd


def decode_audio_file(path: str) -> "np.ndarray":
    """Decode local media to mono 16 kHz float32 PCM without passing the file path to PyAV."""
    if np is None:
        raise RuntimeError("numpy is required")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH")

    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", path,
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or f"ffmpeg exited with status {result.returncode}")
    if not result.stdout:
        raise RuntimeError("input file contained no decodable audio")

    return np.frombuffer(result.stdout, np.int16).copy().astype(np.float32) / 32768.0


def probe_duration(path: str) -> float:
    """Return a media file duration in seconds, or 0 when it cannot be determined."""
    if shutil.which("ffprobe") is None:
        return 0.0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
        return max(0.0, float(result.stdout.strip()))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def progress_thread(pipe, duration: float):
    """Consume FFmpeg stderr and print percentage progress for finite file runs."""
    last_percent = -1
    values = {}

    while True:
        raw_line = pipe.readline()
        if not raw_line:
            break
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = raw_line.strip()

        if duration <= 0:
            if line and not line.startswith(("frame=", "out_time=", "progress=")):
                print(line, file=sys.stderr)
            continue

        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value

        if key in ("progress", "out_time_us"):
            elapsed = 0.0
            if values.get("out_time_us", "").strip():
                try:
                    elapsed = float(values["out_time_us"]) / 1_000_000.0
                except ValueError:
                    elapsed = 0.0
            elif values.get("out_time", "").strip():
                try:
                    hh, mm, ss = values["out_time"].split(":")
                    elapsed = float(hh) * 3600 + float(mm) * 60 + float(ss)
                except (ValueError, AttributeError):
                    elapsed = 0.0

            percent = min(100, max(0, int(elapsed / duration * 100)))
            if percent != last_percent:
                print(f"\rProcessing: {percent}%", end="", file=sys.stderr, flush=True)
                last_percent = percent
            if value == "end":
                print("\rProcessing: 100%", file=sys.stderr)
                values.clear()


def stop_ffmpeg(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Ask FFmpeg to quit gracefully so local muxers can write the MP4 moov atom."""
    if proc.poll() is not None:
        return

    if proc.stdin is not None:
        try:
            proc.stdin.write(b"q\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def reader_thread(pipe, q: queue.Queue):
    """Drain ffmpeg stdout and preserve partial PCM reads across read boundaries."""
    pending = bytearray()
    while True:
        data = pipe.read(CHUNK_BYTES)
        if not data:
            if pending:
                padded = bytes(pending).ljust(CHUNK_BYTES, b"\0")
                chunk = np.frombuffer(padded, np.int16).astype(np.float32) / 32768.0
                put_chunk(q, chunk)
            q.put(None)
            return

        pending.extend(data)
        while len(pending) >= CHUNK_BYTES:
            chunk_bytes = bytes(pending[:CHUNK_BYTES])
            del pending[:CHUNK_BYTES]
            put_chunk(q, np.frombuffer(chunk_bytes, np.int16).astype(np.float32) / 32768.0)


def put_chunk(q: queue.Queue, chunk: "np.ndarray"):
    """Keep the newest audio when the consumer is too slow."""
    try:
        q.put_nowait(chunk)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(chunk)
        except queue.Full:
            pass


def check_ffmpeg(args, p: argparse.ArgumentParser) -> None:
    """Fail fast for missing FFmpeg executables, filters, or encoders."""
    if shutil.which("ffmpeg") is None:
        p.error("ffmpeg is not on PATH")

    try:
        filters = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).stdout
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).stdout
    except OSError as exc:
        p.error(f"could not inspect ffmpeg: {exc}")

    if " drawtext " not in filters:
        p.error("ffmpeg was built without the drawtext filter")
    if args.encoder not in encoders:
        p.error(f"ffmpeg does not provide the requested {args.encoder!r} encoder")


def main():
    p = argparse.ArgumentParser(description="Realtime Whisper subtitles on an RTSP stream.")
    p.add_argument("input", help="source: an rtsp:// URL, or a path to an MP4/video file (played once unless --loop is used)")
    p.add_argument(
        "output",
        nargs="?",
        metavar="OVERLAY_DESTINATION",
        help="required overlay destination: RTSP URL or local output file (omit only with --transcribe-only)",
    )
    p.add_argument("--loop", action="store_true", help="loop a local file source continuously")
    p.add_argument("--transcribe-only", nargs="?", const="", metavar="TXT", help="transcribe a local input to timestamped text instead of creating output media; optional TXT path")
    p.add_argument("--showtranscriptions", action="store_true", help="print finalized transcripts with UTC timestamps for RTSP sources or relative timestamps for files")
    p.add_argument("--showoverlay", action="store_true", help="print the text strings written to the ffmpeg drawtext overlay")
    p.add_argument(
        "--model",
        choices=["tiny.en", "base.en", "small.en", "medium.en"],
        default="base.en",
        help="English-only Whisper model (default: base.en)",
    )
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--compute-type", default="default", help="int8 (CPU), float16 (GPU), ...")
    # segmentation
    p.add_argument("--vad-threshold", type=float, default=0.01, help="RMS level counted as speech (0-1)")
    p.add_argument("--end-silence", type=float, default=0.7, help="seconds of silence that end an utterance")
    p.add_argument("--max-utterance", type=float, default=12.0, help="force a cut after this many seconds")
    p.add_argument("--partial-interval", type=float, default=1.0, help="seconds between live partial updates")
    p.add_argument("--hold", type=float, default=3.0, help="seconds to keep text on screen after speech ends")
    # look and encoding
    p.add_argument("--font", default=default_font(), help="path to a .ttf font")
    p.add_argument("--font-size", default="36")
    p.add_argument("--margin", type=int, default=60, help="pixels from bottom edge")
    p.add_argument("--wrap", type=int, default=42, help="characters per subtitle line")
    p.add_argument("--encoder", default="libx264", help="libx264, h264_nvenc, h264_qsv, h264_videotoolbox...")
    p.add_argument("--bitrate", default="3M")
    p.add_argument("--gop", type=int, default=50)
    args = p.parse_args()

    if args.vad_threshold < 0 or args.vad_threshold > 1:
        p.error("--vad-threshold must be between 0 and 1")
    if args.end_silence < 0:
        p.error("--end-silence must be non-negative")
    if args.max_utterance <= 0:
        p.error("--max-utterance must be greater than 0")
    if args.partial_interval <= 0:
        p.error("--partial-interval must be greater than 0")
    if args.hold < 0:
        p.error("--hold must be non-negative")
    if args.wrap <= 0:
        p.error("--wrap must be greater than 0")

    if not is_rtsp(args.input) and not Path(args.input).is_file():
        p.error(f"input is neither an rtsp:// URL nor an existing file: {args.input}")
    if args.transcribe_only is not None:
        if args.output:
            p.error("with --transcribe-only, specify the text file using --transcribe-only TXT, not the positional output")
    elif args.output is None:
        p.error("OVERLAY_DESTINATION is required unless --transcribe-only is used")

    if args.output is not None and not is_rtsp(args.output):
        output_path = Path(args.output)
        if not output_path.suffix:
            p.error("local output files must include a filename extension, such as output.mp4")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            p.error(f"could not create output directory: {exc}")
    if np is None:
        p.error("numpy is required; install it with 'pip install numpy'")
    if WhisperModel is None:
        p.error("faster-whisper is required; install it with 'pip install faster-whisper'")

    if args.transcribe_only is None:
        check_ffmpeg(args, p)

    print(f"Loading model {args.model}...")
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    except Exception as exc:
        p.error(f"could not load Whisper model: {exc}")

    if args.transcribe_only is not None and is_rtsp(args.input):
        transcript_path = Path(args.transcribe_only) if args.transcribe_only else Path("rtsp_transcript.txt")
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text("", encoding="utf-8")
        print(f"Transcribing {args.input}...")

        proc = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-nostats",
                "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay",
                "-i", args.input,
                "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "s16le", "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )

        q: queue.Queue = queue.Queue(maxsize=50)
        threading.Thread(target=reader_thread, args=(proc.stdout, q), daemon=True).start()

        preroll = deque(maxlen=3)
        buf: list = []
        active = False
        silence = 0.0
        run_start = time.monotonic()
        utterance_start = 0.0
        utterance_started_utc = None

        try:
            while True:
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    chunk = np.zeros(0, np.float32)
                    if proc.poll() is not None:
                        break
                if chunk is None:
                    break

                now = time.monotonic()
                if chunk.size:
                    is_speech = float(np.sqrt(np.mean(chunk ** 2))) > args.vad_threshold
                    if not active:
                        preroll.append(chunk)
                        if is_speech:
                            active, silence = True, 0.0
                            utterance_start = max(0.0, now - run_start)
                            utterance_started_utc = datetime.now(timezone.utc)
                            buf = list(preroll)
                            preroll.clear()
                    else:
                        buf.append(chunk)
                        silence = 0.0 if is_speech else silence + CHUNK_SEC

                if active:
                    dur = len(buf) * CHUNK_SEC
                    if silence >= args.end_silence or dur >= args.max_utterance:
                        segments, _ = model.transcribe(
                            np.concatenate(buf),
                            beam_size=1,
                            vad_filter=True,
                            condition_on_previous_text=False,
                            no_speech_threshold=0.6,
                        )
                        text = " ".join(s.text.strip() for s in segments).strip()
                        if text:
                            cue_end = max(0.0, time.monotonic() - run_start)
                            line = f"[{format_timestamp(utterance_start)} --> {format_timestamp(cue_end)}] {text}\n"
                            with transcript_path.open("a", encoding="utf-8") as transcript:
                                transcript.write(line)
                                transcript.flush()
                            utc_start = utterance_started_utc or datetime.now(timezone.utc)
                            utc_end = datetime.now(timezone.utc)
                            if args.showtranscriptions:
                                start_text = utc_start.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                                end_text = utc_end.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                                print(f"[{start_text} --> {end_text}] {text}", flush=True)
                        buf, active, silence = [], False, 0.0
                        utterance_started_utc = None
        except KeyboardInterrupt:
            pass
        finally:
            stop_ffmpeg(proc)
            print(f"Stopped. Wrote transcript to {transcript_path}")
        return

    if args.transcribe_only is not None:
        transcript_path = Path(args.transcribe_only) if args.transcribe_only else Path(args.input).with_suffix(".txt")
        print(f"Transcribing {args.input}...")
        try:
            audio = decode_audio_file(args.input)
            segments, _ = model.transcribe(
                audio,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            segments = list(segments)
            if args.showtranscriptions:
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        print(f"[{format_timestamp(float(segment.start))} --> {format_timestamp(float(segment.end))}] {text}", flush=True)
            count = write_timestamped_transcript(transcript_path, segments)
        except Exception as exc:
            p.error(f"transcription failed: {exc}")
        print(f"Wrote {count} timestamped segment(s) to {transcript_path}")
        return

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(
            audio, beam_size=1, vad_filter=True,
            condition_on_previous_text=False, no_speech_threshold=0.6,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    tmpdir = Path(tempfile.mkdtemp(prefix="rtsp_subs_"))
    subs = SubtitleFile(tmpdir / "subtitle.txt", args.wrap, args.showoverlay)

    cmd = build_ffmpeg_cmd(args, subs.path)
    print("Starting ffmpeg...")
    progress_duration = 0.0
    if not is_rtsp(args.output) and not is_rtsp(args.input) and not args.loop:
        progress_duration = probe_duration(args.input)

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
    except OSError as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        p.error(f"could not start ffmpeg: {exc}")

    q: queue.Queue = queue.Queue(maxsize=50)
    transcription_q: queue.Queue = queue.Queue(maxsize=2)
    transcription_results: queue.Queue = queue.Queue()

    def transcription_worker():
        while True:
            job = transcription_q.get()
            if job is None:
                return
            kind, request_id, audio, cue_start, cue_end = job
            try:
                text = transcribe(audio)
            except Exception as exc:
                print(f"warning: transcription failed: {exc}", file=sys.stderr)
                text = ""
            transcription_results.put((kind, request_id, text, cue_start, cue_end))

    threading.Thread(target=reader_thread, args=(proc.stdout, q), daemon=True).start()
    if progress_duration > 0:
        threading.Thread(target=progress_thread, args=(proc.stderr, progress_duration), daemon=True).start()
    else:
        threading.Thread(target=progress_thread, args=(proc.stderr, 0.0), daemon=True).start()
    threading.Thread(target=transcription_worker, daemon=True).start()

    preroll = deque(maxlen=3)
    buf: list = []
    active = False
    silence = 0.0
    last_partial = 0.0
    clear_at = 0.0
    request_id = 0
    latest_partial_id = -1
    run_start = time.monotonic()
    utterance_start = 0.0
    print(f"Running. Output: {args.output}   (Ctrl+C to stop)")

    def enqueue_transcription(kind: str, audio: np.ndarray) -> int:
        nonlocal request_id
        request_id += 1
        cue_end = max(0.0, time.monotonic() - run_start)
        cue_start = max(0.0, utterance_start)
        job = (kind, request_id, np.ascontiguousarray(audio), cue_start, cue_end)
        if kind == "final":
            try:
                transcription_q.put_nowait(job)
            except queue.Full:
                try:
                    transcription_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    transcription_q.put_nowait(job)
                except queue.Full:
                    pass
        else:
            try:
                transcription_q.put_nowait(job)
            except queue.Full:
                pass
        return request_id

    try:
        while True:
            while True:
                try:
                    kind, result_id, text, cue_start, cue_end = transcription_results.get_nowait()
                except queue.Empty:
                    break
                if kind == "partial":
                    if result_id == latest_partial_id and text:
                        subs.set(text)
                elif text:
                    subs.set(text)
                    if args.showtranscriptions:
                        print(f"[{format_timestamp(cue_start)} --> {format_timestamp(cue_end)}] {text}", flush=True)
                    clear_at = time.monotonic() + args.hold
                else:
                    subs.set("")
                    clear_at = 0.0

            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                chunk = np.zeros(0, np.float32)
                if proc.poll() is not None:
                    break
            if chunk is None:
                break

            now = time.monotonic()
            queue_depth = q.qsize()
            if queue_depth > 25:
                print(f"warning: audio queue is {queue_depth * CHUNK_SEC:.1f}s behind")

            if chunk.size:
                is_speech = float(np.sqrt(np.mean(chunk ** 2))) > args.vad_threshold
                if not active:
                    preroll.append(chunk)
                    if is_speech:
                        active, silence, last_partial = True, 0.0, now
                        utterance_start = max(0.0, now - run_start)
                        buf = list(preroll)
                        preroll.clear()
                else:
                    buf.append(chunk)
                    silence = 0.0 if is_speech else silence + CHUNK_SEC

            if active:
                dur = len(buf) * CHUNK_SEC
                if silence >= args.end_silence or dur >= args.max_utterance:
                    enqueue_transcription("final", np.concatenate(buf))
                    buf, active, silence = [], False, 0.0
                elif now - last_partial >= args.partial_interval and dur >= 0.8:
                    latest_partial_id = enqueue_transcription("partial", np.concatenate(buf))
                    last_partial = time.monotonic()
            elif clear_at and time.monotonic() > clear_at:
                subs.set("")
                clear_at = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        subs.set("")
        transcription_q.put(None)
        stop_ffmpeg(proc)
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("Stopped.")


if __name__ == "__main__":
    main()