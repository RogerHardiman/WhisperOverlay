"""
Transcribe a local audio or video file to timestamped text.

Usage:
    python app2.py input.mp4 transcript.txt
    python app2.py input.mp4 transcript.txt --model small.en --device cuda
"""
import argparse
import shutil
import subprocess
import sys
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


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def iter_audio_chunks(path: str, chunk_seconds: float, overlap_seconds: float):
    """Yield fixed-size PCM chunks while FFmpeg streams the file."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH")

    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", path,
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    chunk_bytes = int(chunk_seconds * SAMPLE_RATE) * 2
    overlap_samples = int(overlap_seconds * SAMPLE_RATE)
    pending = bytearray()
    overlap = np.empty(0, np.float32)
    position = 0.0

    try:
        while True:
            data = proc.stdout.read(1024 * 1024)
            if not data:
                break
            pending.extend(data)

            while len(pending) >= chunk_bytes:
                new_bytes = bytes(pending[:chunk_bytes])
                del pending[:chunk_bytes]
                new_audio = np.frombuffer(new_bytes, np.int16).copy().astype(np.float32) / 32768.0
                chunk_start = max(0.0, position - overlap_seconds)
                chunk = np.concatenate((overlap, new_audio)) if overlap.size else new_audio
                position += len(new_audio) / SAMPLE_RATE
                overlap = new_audio[-overlap_samples:] if overlap_samples > 0 else np.empty(0, np.float32)
                yield chunk, chunk_start, overlap_seconds if chunk_start > 0 else 0.0

        if pending:
            new_audio = np.frombuffer(bytes(pending), np.int16).copy().astype(np.float32) / 32768.0
            chunk_start = max(0.0, position - overlap_seconds)
            chunk = np.concatenate((overlap, new_audio)) if overlap.size else new_audio
            yield chunk, chunk_start, overlap_seconds if chunk_start > 0 else 0.0
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            error = proc.stderr.read().decode("utf-8", errors="replace").strip()
            proc.stderr.close()
        else:
            error = ""
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(error or f"ffmpeg exited with status {return_code}")


def write_timestamped_transcript(path: Path, segments) -> int:
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


def append_timestamped_segments(path: Path, segments, chunk_start: float, overlap_seconds: float) -> int:
    """Append segments, skipping text already supplied by the overlap region."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    overlap_cutoff = overlap_seconds
    with path.open("a", encoding="utf-8") as transcript:
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            segment_start = float(segment.start)
            segment_end = float(segment.end)
            if chunk_start > 0 and segment_end <= overlap_cutoff:
                continue
            start = format_timestamp(chunk_start + segment_start)
            end = format_timestamp(chunk_start + segment_end)
            transcript.write(f"[{start} --> {end}] {text}\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a local audio or video file using faster-whisper."
    )
    parser.add_argument("input", help="path to the input audio or video file")
    parser.add_argument("output", help="path to the timestamped text file")
    parser.add_argument(
        "--model",
        choices=["tiny.en", "base.en", "small.en", "medium.en"],
        default="base.en",
        help="English-only Whisper model (default: base.en)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--compute-type", default="default", help="int8 (CPU), float16 (GPU), ...")
    parser.add_argument("--chunk-seconds", type=float, default=60.0, help="audio seconds transcribed per chunk (default: 60)")
    parser.add_argument("--overlap-seconds", type=float, default=5.0, help="audio seconds reused between chunks (default: 5)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input is not an existing file: {args.input}")
    if np is None:
        parser.error("numpy is required; install it with 'pip install numpy'")
    if WhisperModel is None:
        parser.error("faster-whisper is required; install it with 'pip install faster-whisper'")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required and must be on PATH")
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be greater than 0")
    if args.overlap_seconds < 0:
        parser.error("--overlap-seconds must be non-negative")
    if args.overlap_seconds >= args.chunk_seconds:
        parser.error("--overlap-seconds must be smaller than --chunk-seconds")

    print(f"Loading model {args.model}...")
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    except Exception as exc:
        parser.error(f"could not load Whisper model: {exc}")

    print(f"Transcribing {args.input}...")
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        count = 0
        for audio_chunk, chunk_start, overlap_seconds in iter_audio_chunks(
            args.input,
            args.chunk_seconds,
            args.overlap_seconds,
        ):
            segments, _ = model.transcribe(
                audio_chunk,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            count += append_timestamped_segments(
                output_path,
                list(segments),
                chunk_start,
                overlap_seconds,
            )
    except Exception as exc:
        parser.error(f"transcription failed: {exc}")

    print(f"Wrote {count} timestamped segment(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
