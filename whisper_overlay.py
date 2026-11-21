from __future__ import annotations

import os
import queue
import subprocess
import sys
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


LogCallback = Callable[[str], None]
ModelProgressCallback = Callable[[int, str], None]

MODEL_REPOSITORIES = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def _snapshot_download_function():
    from huggingface_hub import snapshot_download

    return snapshot_download


def is_model_downloaded(model_size: str) -> bool:
    repo_id = MODEL_REPOSITORIES[model_size]
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        _snapshot_download_function()(repo_id, local_files_only=True)
    except LocalEntryNotFoundError:
        return False
    return True


def download_model(model_size: str, on_progress: ModelProgressCallback) -> None:
    from huggingface_hub.utils import tqdm

    repo_id = MODEL_REPOSITORIES[model_size]

    class ProgressBar(tqdm):
        def update(self, n: int = 1) -> bool:
            updated = super().update(n)
            if self.total:
                percent = min(100, int(self.n * 100 / self.total))
                on_progress(percent, str(self.desc or "model files"))
            return updated

    _snapshot_download_function()(repo_id, tqdm_class=ProgressBar)


def default_ffmpeg_path() -> str:
    extraction_dir = getattr(sys, "_MEIPASS", None)
    if extraction_dir:
        bundled_ffmpeg = Path(extraction_dir) / "ffmpeg.exe"
        if bundled_ffmpeg.is_file():
            return str(bundled_ffmpeg)
    return "ffmpeg"


@dataclass(frozen=True)
class PipelineConfig:
    input_kind: str
    input_source: str
    output_kind: str
    output_destination: str
    audio_mode: str
    model_size: str
    loop_input: bool = False
    ffmpeg_path: str = default_ffmpeg_path()
    font_path: str = "C:/Windows/Fonts/arial.ttf"
    chunk_seconds: int = 5


def _filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _caption_filter(text_path: Path, font_path: str) -> str:
    text = _filter_value(text_path.as_posix())
    font = _filter_value(font_path)
    return (
        f"drawtext=fontfile='{font}':textfile='{text}':reload=1:"
        "fontcolor=white:fontsize=28:box=1:boxcolor=black@0.65:"
        "boxborderw=12:x=(w-text_w)/2:y=h-text_h-40:expansion=none"
    )


def _input_arguments(config: PipelineConfig) -> list[str]:
    if config.input_kind == "RTSP":
        return ["-rtsp_transport", "tcp", "-i", config.input_source]
    if config.loop_input:
        return ["-stream_loop", "-1", "-re", "-i", config.input_source]
    return ["-i", config.input_source]


def build_output_command(config: PipelineConfig, caption_path: Path) -> list[str]:
    command = [config.ffmpeg_path, "-hide_banner", "-y"]
    command.extend(_input_arguments(config))

    command.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            _caption_filter(caption_path, config.font_path),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "50",
            "-keyint_min",
            "50",
            "-sc_threshold",
            "0",
        ]
    )
    if config.audio_mode == "AAC":
        command.extend(["-c:a", "aac", "-b:a", "256k"])
    else:
        command.extend(["-c:a", "copy"])

    if config.output_kind == "RTSP":
        command.extend(["-f", "rtsp", "-rtsp_transport", "tcp", "-muxdelay", "0.1"])
    command.append(config.output_destination)
    command.extend(_audio_pipe_arguments())
    return command


def _audio_pipe_arguments() -> list[str]:
    return [
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]


class WhisperOverlayPipeline:
    def __init__(self, config: PipelineConfig, on_log: LogCallback) -> None:
        self.config = config
        self.on_log = on_log
        self._stop = threading.Event()
        self._processes: set[subprocess.Popen[bytes]] = set()
        self._process_lock = threading.Lock()
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._caption_path: Path | None = None

    def _log(self, message: str) -> None:
        self.on_log(message)

    def _start_process(self, command: list[str], *, capture_audio: bool = False) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture_audio else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_audio else subprocess.PIPE,
            bufsize=0,
        )
        with self._process_lock:
            self._processes.add(process)
            should_stop = self._stop.is_set()
        if should_stop:
            process.terminate()
        return process

    def _forget_process(self, process: subprocess.Popen[bytes]) -> None:
        with self._process_lock:
            self._processes.discard(process)

    def stop(self) -> None:
        self._stop.set()
        with self._process_lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def run(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="whisper-overlay-")
        self._caption_path = Path(self._temp_dir.name) / "caption.txt"
        self._caption_path.write_text("", encoding="utf-8")
        output: subprocess.Popen[bytes] | None = None
        audio_thread: threading.Thread | None = None
        output_error_thread: threading.Thread | None = None
        try:
            output = self._start_process(
                build_output_command(self.config, self._caption_path), capture_audio=True
            )
            self._log("FFmpeg output started.")
            output_error_thread = threading.Thread(
                target=self._drain_process_errors, args=(output, "FFmpeg output"),
                name="output-errors", daemon=True
            )
            output_error_thread.start()
            audio_thread = threading.Thread(
                target=self._transcribe_audio,
                args=(output,),
                name="whisper-audio",
                daemon=True,
            )
            audio_thread.start()
            while output.poll() is None and not self._stop.wait(0.2):
                pass
            return_code = output.wait()
            if return_code and not self._stop.is_set():
                self._log(f"FFmpeg output exited with code {return_code}.")
        except OSError as error:
            self._log(f"Could not start FFmpeg output: {error}")
        finally:
            self.stop()
            if output is not None:
                try:
                    output.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    output.kill()
                    output.wait()
            if audio_thread is not None:
                audio_thread.join(timeout=3)
            if output is not None:
                self._forget_process(output)
            if output_error_thread is not None:
                output_error_thread.join(timeout=1)
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None
            self._log("Pipeline stopped.")

    def _drain_process_errors(self, process: subprocess.Popen[bytes], label: str) -> None:
        if process.stderr is None:
            return
        while True:
            try:
                line = process.stderr.readline()
            except OSError:
                return
            if not line:
                return
            message = line.decode("utf-8", errors="replace").strip()
            if message:
                self._log(f"{label}: {message}")

    def _transcribe_audio(self, process: subprocess.Popen[bytes]) -> None:
        assert self._caption_path is not None
        if process.stdout is None:
            self._log("FFmpeg did not provide an audio stream.")
            self.stop()
            return

        pending: queue.Queue[bytes] = queue.Queue(maxsize=2)
        samples_per_chunk = 16000 * self.config.chunk_seconds
        bytes_per_chunk = samples_per_chunk * 2
        reader = threading.Thread(
            target=self._read_audio_chunks,
            args=(process, pending, bytes_per_chunk),
            name="audio-reader",
            daemon=True,
        )
        reader.start()
        try:
            self._log(f"Loading faster-whisper model '{self.config.model_size}'...")
            from faster_whisper import WhisperModel

            model = WhisperModel(self.config.model_size, device="auto", compute_type="auto")
            self._log("Transcription is running.")
            while not self._stop.is_set():
                try:
                    chunk = pending.get(timeout=0.25)
                except queue.Empty:
                    if process.poll() is not None and not reader.is_alive():
                        break
                    continue
                text = self._transcribe_chunk(model, chunk)
                self._write_caption(text)
        except Exception as error:
            self._log(f"Transcription failed: {error}")
            self.stop()
        finally:
            reader.join(timeout=1)

    def _read_audio_chunks(
        self, process: subprocess.Popen[bytes], chunks: queue.Queue[bytes], chunk_size: int
    ) -> None:
        assert process.stdout is not None
        while not self._stop.is_set():
            data = bytearray()
            while len(data) < chunk_size and not self._stop.is_set():
                part = process.stdout.read(chunk_size - len(data))
                if not part:
                    break
                data.extend(part)
            if len(data) < chunk_size:
                return
            if chunks.full():
                try:
                    chunks.get_nowait()
                except queue.Empty:
                    pass
            try:
                chunks.put_nowait(bytes(data))
            except queue.Full:
                pass

    def _transcribe_chunk(self, model: object, pcm: bytes) -> str:
        assert self._temp_dir is not None
        wav_path = Path(self._temp_dir.name) / "chunk.wav"
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm)

        segments, _info = model.transcribe(  # type: ignore[attr-defined]
            str(wav_path), vad_filter=True, beam_size=3
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _write_caption(self, text: str) -> None:
        assert self._caption_path is not None
        safe_text = " ".join(text.replace("\x00", "").split())[:300]
        temporary_path = self._caption_path.with_suffix(".tmp")
        temporary_path.write_text(safe_text, encoding="utf-8")
        os.replace(temporary_path, self._caption_path)
