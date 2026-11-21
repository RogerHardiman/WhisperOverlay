import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import app


class AppTests(unittest.TestCase):
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

    def test_build_ffmpeg_cmd_uses_optional_audio_maps(self):
        args = SimpleNamespace(
            font_size="36",
            margin=60,
            font="",
            encoder="libx264",
            gop=50,
            bitrate="3M",
            input="video.mp4",
            no_loop=False,
            output="rtsp://localhost:8554/subtitles",
        )
        subfile = Path("/tmp/subtitles.txt")
        cmd = app.build_ffmpeg_cmd(args, subfile)
        self.assertIn("0:a:0?", cmd)
        self.assertIn("-stream_loop", cmd)


if __name__ == "__main__":
    unittest.main()
