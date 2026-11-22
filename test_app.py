import io
import queue
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace

import app


class AppTests(unittest.TestCase):
    @patch("app.shutil.which", return_value="ffmpeg")
    @patch("app.subprocess.run")
    def test_decode_mp4_audio_uses_ffmpeg_pcm(self, run, _which):
        run.return_value = Mock(
            returncode=0,
            stdout=b"\x00\x40\xff\xbf",
            stderr=b"",
        )

        audio = app.decode_audio_file("input.mp4")

        command = run.call_args.args[0]
        self.assertIn("-i", command)
        self.assertIn("input.mp4", command)
        self.assertIn("-ar", command)
        self.assertIn("16000", command)
        self.assertIn("s16le", command)
        self.assertEqual(audio.dtype.name, "float32")

    def test_format_timestamp(self):
        self.assertEqual(app.format_timestamp(3661.234), "01:01:01.234")

    def test_write_timestamped_transcript(self):
        segment = SimpleNamespace(start=1.25, end=2.5, text=" Hello world ")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "transcript.txt"
            count = app.write_timestamped_transcript(path, [segment])
            self.assertEqual(count, 1)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "[00:00:01.250 --> 00:00:02.500] Hello world\n",
            )

    def test_stop_ffmpeg_sends_quit_before_wait(self):
        stdin = Mock()
        proc = Mock()
        proc.poll.return_value = None
        proc.stdin = stdin
        proc.wait.return_value = 0

        app.stop_ffmpeg(proc)

        stdin.write.assert_called_once_with(b"q\n")
        stdin.flush.assert_called_once_with()
        proc.wait.assert_called_once_with(timeout=10.0)
        proc.terminate.assert_not_called()

    def test_is_rtsp(self):
        self.assertTrue(app.is_rtsp("RTSP://camera/stream"))
        self.assertTrue(app.is_rtsp("rtsps://camera/stream"))
        self.assertFalse(app.is_rtsp("video.mp4"))

    def test_ffmpeg_escape(self):
        self.assertEqual(app.ffmpeg_escape("C:\\Video\\one'two.txt"), r"C\:/Video/one\'two.txt")

    def test_subtitle_file_wraps_and_atomically_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subtitle.txt"
            subs = app.SubtitleFile(path, 10)
            subs.set("one two three four five")
            self.assertEqual(path.read_text(encoding="utf-8"), "three four\nfive")

    def test_subtitle_file_prints_overlay_text_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subtitle.txt"
            subs = app.SubtitleFile(path, 20, show_overlay=True)
            with patch("builtins.print") as print_mock:
                subs.set("Hello world")
                subs.set("")
            print_mock.assert_called_once_with("Hello world", flush=True)

    def test_build_ffmpeg_cmd_uses_optional_audio_maps(self):
        args = SimpleNamespace(
            font_size="36",
            margin=60,
            font="",
            encoder="libx264",
            gop=50,
            bitrate="3M",
            input="video.mp4",
            loop=True,
            output="rtsp://localhost:8554/subtitles",
        )
        subfile = Path("/tmp/subtitles.txt")
        cmd = app.build_ffmpeg_cmd(args, subfile)
        self.assertIn("0:a:0?", cmd)
        self.assertIn("-stream_loop", cmd)
        drawtext = cmd[cmd.index("-vf") + 1]
        self.assertIn(":x=60:", drawtext)

    def test_build_ffmpeg_cmd_supports_rtsp_output(self):
        args = SimpleNamespace(
            font_size="36", margin=60, font="", encoder="libx264",
            gop=50, bitrate="3M", input="video.mp4", loop=False,
            output="rtsp://localhost:8554/subtitles",
        )
        cmd = app.build_ffmpeg_cmd(args, Path("/tmp/subtitles.txt"))
        self.assertIn("-f", cmd)
        self.assertIn("rtsp", cmd)
        self.assertIn("rtsp://localhost:8554/subtitles", cmd)

    def test_progress_thread_prints_nothing(self):
        pipe = io.StringIO("out_time_us=5000000\nprogress=continue\nout_time_us=10000000\nprogress=end\n")
        with patch("builtins.print") as print_mock:
            app.progress_thread(pipe, 10.0)
        print_mock.assert_not_called()

    def test_windows_drive_path_is_escaped(self):
        escaped = app.ffmpeg_escape("C:\\Users\\Roger\\subtitles.txt")
        self.assertEqual(escaped, r"C\:/Users/Roger/subtitles.txt")

    def test_progress_thread_accepts_bytes(self):
        pipe = io.BytesIO(b"out_time_us=5000000\nprogress=continue\nout_time_us=10000000\nprogress=end\n")
        app.progress_thread(pipe, 10.0)

    def test_build_ffmpeg_cmd_supports_file_output(self):
        args = SimpleNamespace(
            font_size="36", margin=60, font="", encoder="libx264",
            gop=50, bitrate="3M", input="video.mp4", loop=False,
            output="/tmp/subtitles.mp4",
        )
        cmd = app.build_ffmpeg_cmd(args, Path("/tmp/subtitles.txt"))
        self.assertIn("-y", cmd)
        self.assertIn("+faststart", cmd)
        self.assertIn("-progress", cmd)
        self.assertIn("pipe:2", cmd)
        self.assertIn("/tmp/subtitles.mp4", cmd)
        self.assertNotIn("-rtsp_transport", cmd)

    def test_build_ffmpeg_cmd_supports_ts_output(self):
        args = SimpleNamespace(
            font_size="36", margin=60, font="", encoder="libx264",
            gop=50, bitrate="3M", input="video.mp4", loop=False,
            output="/tmp/subtitles.ts",
        )
        cmd = app.build_ffmpeg_cmd(args, Path("/tmp/subtitles.txt"))
        output_index = cmd.index("/tmp/subtitles.ts")
        self.assertEqual(cmd[output_index - 2:output_index], ["-f", "mpegts"])
        self.assertNotIn("+faststart", cmd)


if __name__ == "__main__":
    unittest.main()
