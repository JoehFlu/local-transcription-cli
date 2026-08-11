import contextlib
import io
import unittest

import transcribe


class PlatformSupportTests(unittest.TestCase):
    def test_runtime_platform(self):
        self.assertEqual(transcribe.runtime_platform("darwin"), "macos")
        self.assertEqual(transcribe.runtime_platform("linux"), "linux")
        self.assertEqual(transcribe.runtime_platform("linux2"), "linux")
        self.assertEqual(transcribe.runtime_platform("win32"), "windows")
        self.assertEqual(transcribe.runtime_platform("freebsd"), "other")

    def test_macos_install_hints(self):
        self.assertEqual(transcribe.ffmpeg_install_hint("darwin"), "brew install ffmpeg")
        self.assertEqual(
            transcribe.ollama_install_hint("darwin"),
            "brew install --cask ollama",
        )

    def test_wsl_install_hints(self):
        self.assertIn("apt install", transcribe.ffmpeg_install_hint("linux"))
        self.assertIn("ollama.com/install.sh", transcribe.ollama_install_hint("linux"))

    def test_native_windows_is_rejected_with_wsl_hint(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            transcribe.ensure_supported_platform("win32")

        self.assertIn("WSL 2", output.getvalue())

    def test_macos_and_linux_are_supported(self):
        transcribe.ensure_supported_platform("darwin")
        transcribe.ensure_supported_platform("linux")


if __name__ == "__main__":
    unittest.main()
