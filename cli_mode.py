from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

from whisper_overlay import PipelineConfig, WhisperOverlayPipeline


MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WhisperOverlay video captioning")
    parser.add_argument(
        "--headless", action="store_true", help="Run without opening the desktop window"
    )
    parser.add_argument("--input-kind", choices=("FILE", "RTSP"))
    parser.add_argument("--input", dest="input_source", help="Input file path or RTSP URL")
    parser.add_argument("--output-kind", choices=("FILE", "RTSP"))
    parser.add_argument("--output", dest="output_destination", help="Output file path or RTSP URL")
    parser.add_argument("--model", choices=MODEL_SIZES, default="small")
    parser.add_argument("--audio-mode", choices=("aac", "copy"), default="aac")
    parser.add_argument("--loop-file", action="store_true", help="Loop FILE input in real time")
    return parser


def hide_standalone_console() -> None:
    if os.name != "nt" or sys.argv[1:]:
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_processes = kernel32.GetConsoleProcessList
    get_processes.argtypes = [ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    get_processes.restype = ctypes.c_ulong
    processes = (ctypes.c_ulong * 16)()
    if get_processes(processes, len(processes)) != 1:
        return

    get_window = kernel32.GetConsoleWindow
    get_window.restype = ctypes.c_void_p
    show_window = kernel32.ShowWindow
    show_window.argtypes = [ctypes.c_void_p, ctypes.c_int]
    show_window.restype = ctypes.c_bool
    window = get_window()
    if window:
        show_window(window, 0)


def run_headless_if_requested(argv: list[str]) -> bool:
    parser = _argument_parser()
    args = parser.parse_args(argv)

    if not args.headless:
        if argv:
            parser.error("Command-line options require --headless.")
        return False

    required = {
        "--input-kind": args.input_kind,
        "--input": args.input_source,
        "--output-kind": args.output_kind,
        "--output": args.output_destination,
    }
    missing = [option for option, value in required.items() if not value]
    if missing:
        parser.error(f"--headless requires: {', '.join(missing)}")
    if args.loop_file and args.input_kind != "FILE":
        parser.error("--loop-file can only be used with --input-kind FILE.")
    if args.input_kind == "FILE" and not Path(args.input_source).is_file():
        parser.error(f"Input file does not exist: {args.input_source}")
    if args.output_kind == "FILE" and not Path(args.output_destination).parent.is_dir():
        parser.error("The output folder must already exist.")

    audio_mode = "AAC" if args.audio_mode == "aac" else "Pass through"
    config = PipelineConfig(
        input_kind=args.input_kind,
        input_source=args.input_source,
        output_kind=args.output_kind,
        output_destination=args.output_destination,
        audio_mode=audio_mode,
        model_size=args.model,
        loop_input=args.loop_file,
    )
    pipeline = WhisperOverlayPipeline(config, lambda message: print(message, flush=True))
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()
        print("Interrupted; stopping pipeline.", flush=True)
    return True
