import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import app2


class App2Tests(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(app2.format_timestamp(3661.234), "01:01:01.234")

    def test_write_timestamped_transcript(self):
        segment = SimpleNamespace(start=1.25, end=2.5, text=" Hello world ")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "transcript.txt"
            count = app2.write_timestamped_transcript(path, [segment])
            self.assertEqual(count, 1)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "[00:00:01.250 --> 00:00:02.500] Hello world\n",
            )

    def test_append_timestamped_segments_skips_overlap(self):
        segment = SimpleNamespace(start=1.0, end=4.0, text="Inside overlap")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.txt"
            count = app2.append_timestamped_segments(path, [segment], 60.0, 5.0)
            self.assertEqual(count, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
