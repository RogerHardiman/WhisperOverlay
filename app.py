from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from cli_mode import hide_standalone_console, run_headless_if_requested
from whisper_overlay import PipelineConfig, WhisperOverlayPipeline


class WhisperOverlayApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("WhisperOverlay")
        self.root.minsize(720, 540)
        self.pipeline: WhisperOverlayPipeline | None = None

        self.input_kind = tk.StringVar(value="FILE")
        self.input_source = tk.StringVar()
        self.output_kind = tk.StringVar(value="FILE")
        self.output_destination = tk.StringVar()
        self.audio_mode = tk.StringVar(value="AAC")
        self.model_size = tk.StringVar(value="small")
        self.loop_file = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Input").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(frame, text="Source type").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        input_type = ttk.Combobox(
            frame, textvariable=self.input_kind, values=("FILE", "RTSP"), state="readonly", width=12
        )
        input_type.grid(row=1, column=1, sticky="w", pady=4)
        input_type.bind("<<ComboboxSelected>>", self._update_browse_buttons)

        ttk.Label(frame, text="Video source").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(frame, textvariable=self.input_source).grid(row=2, column=1, sticky="ew", pady=4)
        self.input_browse = ttk.Button(frame, text="Browse...", command=self._browse_input)
        self.input_browse.grid(row=2, column=2, padx=(8, 0), pady=4)

        self.loop_checkbox = ttk.Checkbutton(
            frame, text="Loop file indefinitely (real-time)", variable=self.loop_file
        )
        self.loop_checkbox.grid(row=3, column=1, sticky="w", pady=4)

        ttk.Separator(frame).grid(row=4, column=0, columnspan=3, sticky="ew", pady=12)
        ttk.Label(frame, text="Output").grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(frame, text="Destination type").grid(row=6, column=0, sticky="w", padx=(0, 10), pady=4)
        output_type = ttk.Combobox(
            frame, textvariable=self.output_kind, values=("FILE", "RTSP"), state="readonly", width=12
        )
        output_type.grid(row=6, column=1, sticky="w", pady=4)
        output_type.bind("<<ComboboxSelected>>", self._update_browse_buttons)

        ttk.Label(frame, text="Video destination").grid(row=7, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(frame, textvariable=self.output_destination).grid(row=7, column=1, sticky="ew", pady=4)
        self.output_browse = ttk.Button(frame, text="Browse...", command=self._browse_output)
        self.output_browse.grid(row=7, column=2, padx=(8, 0), pady=4)

        ttk.Label(frame, text="Audio").grid(row=8, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(
            frame, textvariable=self.audio_mode, values=("AAC", "Pass through"), state="readonly", width=16
        ).grid(row=8, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Whisper model").grid(row=9, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.model_size,
            values=("tiny", "base", "small", "medium", "large-v3"),
            state="readonly",
            width=16,
        ).grid(row=9, column=1, sticky="w", pady=4)

        actions = ttk.Frame(frame)
        actions.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        self.start_button = ttk.Button(actions, text="Start", command=self._start)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status).pack(side=tk.RIGHT)

        ttk.Label(frame, text="Activity").grid(row=11, column=0, columnspan=3, sticky="w", pady=(6, 4))
        self.log = tk.Text(frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.log.grid(row=12, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(12, weight=1)
        self._update_browse_buttons()

    def _update_browse_buttons(self, _event: object = None) -> None:
        self.input_browse.configure(state=tk.NORMAL if self.input_kind.get() == "FILE" else tk.DISABLED)
        self.loop_checkbox.configure(state=tk.NORMAL if self.input_kind.get() == "FILE" else tk.DISABLED)
        self.output_browse.configure(state=tk.NORMAL if self.output_kind.get() == "FILE" else tk.DISABLED)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=(("Video files", "*.mp4 *.mkv *.mov *.avi *.ts"), ("All files", "*.*"))
        )
        if path:
            self.input_source.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=(("MP4 video", "*.mp4"), ("Matroska video", "*.mkv"), ("All files", "*.*")),
        )
        if path:
            self.output_destination.set(path)

    def _start(self) -> None:
        source = self.input_source.get().strip()
        destination = self.output_destination.get().strip()
        if not source or not destination:
            messagebox.showerror("Missing path", "Enter both an input source and an output destination.")
            return
        if self.input_kind.get() == "FILE" and not Path(source).is_file():
            messagebox.showerror("Input file not found", f"The input file does not exist:\n{source}")
            return
        if self.output_kind.get() == "FILE" and not Path(destination).parent.is_dir():
            messagebox.showerror("Output folder not found", "Choose an existing folder for the output file.")
            return

        config = PipelineConfig(
            input_kind=self.input_kind.get(),
            input_source=source,
            output_kind=self.output_kind.get(),
            output_destination=destination,
            audio_mode=self.audio_mode.get(),
            model_size=self.model_size.get(),
            loop_input=self.input_kind.get() == "FILE" and self.loop_file.get(),
        )
        self.pipeline = WhisperOverlayPipeline(config, self._log_from_thread)
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status.set("Starting")
        threading.Thread(target=self._run_pipeline, args=(self.pipeline,), daemon=True).start()

    def _run_pipeline(self, pipeline: WhisperOverlayPipeline) -> None:
        pipeline.run()
        self.root.after(0, self._pipeline_finished, pipeline)

    def _pipeline_finished(self, pipeline: WhisperOverlayPipeline) -> None:
        if self.pipeline is pipeline:
            self.pipeline = None
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.status.set("Ready")

    def _stop(self) -> None:
        if self.pipeline is not None:
            self.status.set("Stopping")
            self.pipeline.stop()

    def _log_from_thread(self, message: str) -> None:
        self.root.after(0, self._append_log, message)

    def _append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.root.destroy()


def main() -> None:
    if run_headless_if_requested(sys.argv[1:]):
        return
    hide_standalone_console()
    root = tk.Tk()
    WhisperOverlayApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
