import unittest
from pathlib import Path

from cli_mode import _argument_parser
from whisper_overlay import PipelineConfig, build_output_command


class PipelineCommandTests(unittest.TestCase):
    def make_config(self, **overrides: str) -> PipelineConfig:
        values = {
            "input_kind": "FILE",
            "input_source": "input.mp4",
            "output_kind": "FILE",
            "output_destination": "output.mp4",
            "audio_mode": "AAC",
            "model_size": "small",
        }
        values.update(overrides)
        return PipelineConfig(**values)

    def test_file_output_encodes_low_latency_h264_and_aac(self) -> None:
        command = build_output_command(self.make_config(), Path("C:/Temp/caption.txt"))

        self.assertIn("libx264", command)
        self.assertIn("zerolatency", command)
        self.assertIn("ultrafast", command)
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[command.index("-b:a") + 1], "256k")
        self.assertIn("reload=1", command[command.index("-vf") + 1])
        self.assertEqual(command[command.index("output.mp4") + 1], "-map")
        self.assertEqual(command[-1], "pipe:1")

    def test_rtsp_output_uses_tcp_and_can_copy_audio(self) -> None:
        config = self.make_config(
            input_kind="RTSP",
            input_source="rtsp://camera/live",
            output_kind="RTSP",
            output_destination="rtsp://server/live",
            audio_mode="Pass through",
        )
        command = build_output_command(config, Path("caption.txt"))

        self.assertEqual(command[command.index("-rtsp_transport") + 1], "tcp")
        self.assertEqual(command.count("-rtsp_transport"), 2)
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertIn("zerolatency", command)
        self.assertEqual(command[command.index("rtsp://server/live") + 1], "-map")

    def test_rtsp_uses_one_input_and_sends_pcm_to_transcriber(self) -> None:
        command = build_output_command(
            self.make_config(input_kind="RTSP", input_source="rtsp://camera/live"),
            Path("caption.txt"),
        )

        self.assertEqual(command.count("-i"), 1)
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[-1], "pipe:1")
        self.assertIn("0:a:0?", command)
        self.assertEqual(command[command.index("output.mp4") + 1], "-map")

    def test_looping_file_input_is_realtime_for_both_ffmpeg_processes(self) -> None:
        config = self.make_config(loop_input=True)
        command = build_output_command(config, Path("caption.txt"))
        input_index = command.index("-i")
        self.assertEqual(command[input_index - 3 : input_index], ["-stream_loop", "-1", "-re"])

    def test_rtsp_input_does_not_get_file_loop_options(self) -> None:
        config = self.make_config(input_kind="RTSP", loop_input=True)

        self.assertNotIn("-stream_loop", build_output_command(config, Path("caption.txt")))

    def test_headless_arguments_expose_audio_model_and_file_looping(self) -> None:
        args = _argument_parser().parse_args(
            [
                "--headless",
                "--input-kind",
                "FILE",
                "--input",
                "input.mp4",
                "--output-kind",
                "RTSP",
                "--output",
                "rtsp://server/live",
                "--model",
                "medium",
                "--audio-mode",
                "copy",
                "--loop-file",
            ]
        )

        self.assertTrue(args.headless)
        self.assertEqual(args.input_kind, "FILE")
        self.assertEqual(args.output_kind, "RTSP")
        self.assertEqual(args.model, "medium")
        self.assertEqual(args.audio_mode, "copy")
        self.assertTrue(args.loop_file)


if __name__ == "__main__":
    unittest.main()
