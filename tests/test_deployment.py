"""Deployment boundaries without downloading weights or changing Python packages."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.name == "posix" and shutil.which("bash") and shutil.which("flock"),
    "Linux shell deployment tests",
)
class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="msv2 deployment ")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.app = self.folder / "application with spaces"
        self.app.mkdir()
        (self.app / "configs").mkdir()
        (self.app / "configs/default.json").write_text(
            '{"sample_fps":3,"scales":[1,2,4]}\n'
        )
        self.root = self.folder / "data with spaces"
        self.root.mkdir()
        self.environment = os.environ.copy()
        self.environment["MSV2_ROOT"] = str(self.root)
        for key in (
            "MSV2_PYTHON",
            "MSV2_SOURCE_DIR",
            "MSV2_SWINIR_SOURCE",
            "MSV2_SWINIR_WEIGHT",
            "MSV2_EXTENSIONS_DIR",
        ):
            self.environment.pop(key, None)

    def script(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + text, encoding="utf-8")
        path.chmod(0o755)

    def invoke(self, script, *arguments):
        return subprocess.run(
            ["bash", str(script), *arguments],
            env=self.environment,
            cwd=self.folder,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_wrapper_preserves_spaces_and_arguments(self):
        shutil.copy2(PROJECT / "msv2.sh", self.app / "msv2.sh")
        interpreter = self.root / ".venv/bin/python"
        self.script(
            interpreter,
            'printf "%s\\0" "$@" > "$MSV2_ROOT/arguments"\nprintf "%s" "$PYTHONPATH" > "$MSV2_ROOT/module-path"\n',
        )
        result = self.invoke(self.app / "msv2.sh", "train", "video name.mp4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.root / "arguments").read_bytes().split(b"\0")[:-1],
            [b"-s", b"-m", b"msv2", b"train", b"video name.mp4"],
        )
        self.assertEqual(
            (self.root / "module-path").read_text().split(":")[0], str(self.app)
        )

    def test_wrapper_explicit_environment_overrides_saved_pointer(self):
        shutil.copy2(PROJECT / "msv2.sh", self.app / "msv2.sh")
        interpreter = self.folder / "chosen interpreter"
        self.script(
            interpreter, 'printf "%s" "$MSV2_SOURCE_DIR" > "$MSV2_ROOT/chosen-source"\n'
        )
        (self.root / "runtime.env").write_text(
            "export MSV2_PYTHON=/nonexistent/old-python\nexport MSV2_SOURCE_DIR=/saved/source\n"
        )
        self.environment["MSV2_PYTHON"] = str(interpreter)
        self.environment["MSV2_SOURCE_DIR"] = "/explicit/source with spaces"
        result = self.invoke(self.app / "msv2.sh", "status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.root / "chosen-source").read_text(), "/explicit/source with spaces"
        )

    def bootstrap_fixture(self, tools=True):
        scripts = self.app / "scripts"
        scripts.mkdir()
        shutil.copy2(PROJECT / "scripts/bootstrap.sh", scripts / "bootstrap.sh")
        command_bin = self.folder / "commands"
        command_bin.mkdir()
        if tools:
            for name in ("ffmpeg", "ffprobe", "colmap"):
                self.script(command_bin / name, "exit 0\n")
        self.command_bin = command_bin
        self.environment["PATH"] = (
            str(command_bin) + os.pathsep + self.environment["PATH"]
        )
        interpreter = self.folder / "existing python"
        self.script(
            interpreter,
            'printf "%s\\0" "$@" >> "$MSV2_ROOT/interpreter-calls"\ncat >/dev/null\n',
        )
        self.environment["MSV2_PYTHON"] = str(interpreter)
        lock = {}
        for key, directory in (
            ("mini_splatting2", "mini-splatting2"),
            ("swinir", "SwinIR"),
        ):
            source = self.root / "vendor" / directory
            source.mkdir(parents=True)
            content = b"fixed upstream source\n"
            (source / "source.py").write_bytes(content)
            item = dict(
                repository="https://example.invalid/source",
                commit="fixed",
                sha256="archive",
                directory=directory,
            )
            lock[key] = item
            receipt = dict(
                commit=item["commit"],
                archive_sha256=item["sha256"],
                files={"source.py": hashlib.sha256(content).hexdigest()},
            )
            (source / ".msv2-source.json").write_text(json.dumps(receipt))
        weight = (
            self.root
            / "weights"
            / "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
        )
        weight.parent.mkdir()
        weight.write_bytes(b"fixture weight, never loaded")
        lock["swinir4_weight"] = dict(
            url="https://example.invalid/never-requested",
            sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        )
        (self.app / "upstream.lock.json").write_text(json.dumps(lock))
        return scripts / "bootstrap.sh"

    def test_existing_interpreter_is_read_only_and_sources_are_pinned(self):
        bootstrap = self.bootstrap_fixture()
        for attempt in range(2):
            result = self.invoke(bootstrap)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("MSV2_RUNTIME_READY", result.stdout)
            if attempt == 0:
                self.assertEqual(
                    (self.root / "config.json").read_bytes(),
                    (self.app / "configs/default.json").read_bytes(),
                )
                (self.root / "config.json").write_text('{"sample_fps":1.5}\n')
            else:
                self.assertEqual(
                    (self.root / "config.json").read_text(), '{"sample_fps":1.5}\n'
                )
        calls = (self.root / "interpreter-calls").read_bytes().split(b"\0")[:-1]
        self.assertEqual(calls, [b"-s", b"-", b"-s", b"-"])
        self.assertFalse((self.root / ".venv").exists())
        self.assertTrue((self.root / "runtime.env").is_file())
        (self.root / "vendor/mini-splatting2/source.py").write_text("changed")
        rejected = self.invoke(bootstrap)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("Installed source was changed", rejected.stderr)
        self.assertEqual(
            (self.root / "interpreter-calls").read_bytes().split(b"\0")[:-1], calls
        )

    def test_setuptools_metadata_rewrite_preserves_source_integrity(self):
        bootstrap = self.bootstrap_fixture()
        source = self.root / "vendor/mini-splatting2"
        name = "submodules/simple-knn/simple_knn.egg-info/PKG-INFO"
        metadata_file = source / name
        metadata_file.parent.mkdir(parents=True)
        original = b"Metadata-Version: 2.1\n"
        metadata_file.write_bytes(original)
        receipt_path = source / ".msv2-source.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][name] = hashlib.sha256(original).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
        # A successful wheel build rewrites packaging metadata in-place.
        metadata_file.write_text("Metadata-Version: 2.4\n")
        for _ in range(2):
            result = self.invoke(bootstrap)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = json.loads(receipt_path.read_text())
            self.assertNotIn(name, saved["files"])
            self.assertIn(name, saved["generated_metadata"])
            self.assertEqual(saved["files"]["source.py"], receipt["files"]["source.py"])
        # A similarly named real file outside .egg-info must still be checked.
        protected = source / "PKG-INFO"
        protected.write_bytes(original)
        saved["files"]["PKG-INFO"] = hashlib.sha256(original).hexdigest()
        receipt_path.write_text(json.dumps(saved))
        protected.write_text("changed")
        rejected = self.invoke(bootstrap)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("Installed source was changed", rejected.stderr)

    def cuda_discovery_fixture(self):
        bootstrap = self.bootstrap_fixture()
        self.environment.pop("MSV2_PYTHON")
        self.environment.pop("CUDA_HOME", None)
        self.environment["HOME"] = str(self.folder / "host home")
        host_python = self.folder / "host python"
        self.script(
            host_python,
            'if [[ "$1" == -c ]]; then exit 0; fi\n'
            'printf "%s" "$CUDA_HOME" > "$MSV2_ROOT/discovered-cuda"\n'
            "cat >/dev/null\nexit 67\n",
        )
        self.environment["MSV2_BOOTSTRAP_PYTHON"] = str(host_python)
        wrong = self.folder / "wrong toolkit/bin/nvcc"
        self.script(
            wrong, 'printf "Cuda compilation tools, release 12.4, V12.4.93\\n"\n'
        )
        self.environment["PATH"] = (
            str(wrong.parent) + os.pathsep + self.environment["PATH"]
        )
        toolkit = (
            Path(self.environment["HOME"]) / ".local/share/runtime/cuda-toolkit-12.8"
        )
        self.script(
            toolkit / "bin/nvcc",
            'printf "Cuda compilation tools, release 12.8, V12.8.93\\n"\n',
        )
        return bootstrap, toolkit

    def test_cuda_discovery_finds_user_toolkit_and_skips_wrong_path_version(self):
        bootstrap, toolkit = self.cuda_discovery_fixture()
        result = self.invoke(bootstrap)
        self.assertEqual(result.returncode, 67, result.stdout + result.stderr)
        self.assertEqual((self.root / "discovered-cuda").read_text(), str(toolkit))
        self.assertFalse((self.root / ".venv").exists())

    def test_explicit_incompatible_cuda_home_is_not_replaced(self):
        bootstrap, _ = self.cuda_discovery_fixture()
        explicit = self.folder / "explicit wrong toolkit"
        self.script(
            explicit / "bin/nvcc",
            'printf "Cuda compilation tools, release 12.4, V12.4.93\\n"\n',
        )
        self.environment["CUDA_HOME"] = str(explicit)
        result = self.invoke(bootstrap)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires CUDA Toolkit 12.8", result.stderr)
        self.assertFalse((self.root / "discovered-cuda").exists())

    def test_missing_system_tools_use_noninteractive_sudo_without_touching_host(self):
        bootstrap = self.bootstrap_fixture(tools=False)
        for name in (
            "bash",
            "dirname",
            "uname",
            "mkdir",
            "flock",
            "env",
            "cat",
            "mv",
            "cp",
            "chmod",
        ):
            (self.command_bin / name).symlink_to(shutil.which(name))
        self.script(self.command_bin / "id", 'printf "1000\\n"\n')
        self.script(
            self.command_bin / "sudo",
            '[[ "$1" == -n ]]\nshift\nif [[ "$1" == true ]]; then exit 0; fi\nexec "$@"\n',
        )
        self.script(
            self.command_bin / "apt-get",
            'printf "%s\\n" "$*" >> "$MSV2_ROOT/apt-calls"\nif [[ "$1" == install ]]; then\nfor tool in ffmpeg ffprobe colmap git; do\nprintf \'#!/usr/bin/env bash\\nexit 0\\n\' > "$MSV2_TEST_BIN/$tool"\nchmod +x "$MSV2_TEST_BIN/$tool"\ndone\nfi\n',
        )
        self.environment["PATH"] = str(self.command_bin)
        self.environment["MSV2_TEST_BIN"] = str(self.command_bin)
        self.environment["MSV2_BOOTSTRAP_PYTHON"] = sys.executable
        result = self.invoke(bootstrap)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = (self.root / "apt-calls").read_text().splitlines()
        self.assertEqual(
            calls, ["update", "install --yes --no-install-recommends ffmpeg colmap git"]
        )
        second = self.invoke(bootstrap)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual((self.root / "apt-calls").read_text().splitlines(), calls)

    def test_missing_system_tools_without_privilege_report_copyable_command(self):
        bootstrap = self.bootstrap_fixture(tools=False)
        for name in ("bash", "dirname", "uname", "mkdir", "flock"):
            (self.command_bin / name).symlink_to(shutil.which(name))
        self.script(self.command_bin / "id", 'printf "1000\\n"\n')
        self.environment["PATH"] = str(self.command_bin)
        self.environment["MSV2_BOOTSTRAP_PYTHON"] = sys.executable
        result = self.invoke(bootstrap)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "sudo apt-get update && sudo apt-get install -y ffmpeg colmap git",
            result.stderr,
        )
        self.assertFalse((self.root / "interpreter-calls").exists())

    def test_deploy_starts_only_one_watcher_and_detects_failed_start(self):
        shutil.copy2(PROJECT / "deploy.sh", self.app / "deploy.sh")
        self.script(self.app / "scripts/bootstrap.sh", "exit 0\n")
        self.script(
            self.app / "msv2.sh",
            'if [[ "$1" == doctor ]]; then exit 0; fi\nsleep 0.3\nexec 8>"$MSV2_ROOT/queue/watcher.lock"\nflock -n 8\nprintf \'{"state":"watching","pid":%s}\\n\' "$$" > "$MSV2_ROOT/queue/watcher.json"\nexec sleep 30\n',
        )
        first = subprocess.Popen(
            ["bash", str(self.app / "deploy.sh")],
            env=self.environment,
            cwd=self.folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            output, error = first.communicate(timeout=10)
            self.assertEqual(first.returncode, 0, error)
            self.assertIn("MSV2_DEPLOYED", output)
            second = self.invoke(self.app / "deploy.sh")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("MSV2_ALREADY_WATCHING", second.stdout)
        finally:
            try:
                os.killpg(first.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.script(
            self.app / "msv2.sh",
            'if [[ "$1" == doctor ]]; then exit 0; fi\nexec 8>"$MSV2_ROOT/queue/watcher.lock"\nflock -n 8\nsleep 0.3\nexit 1\n',
        )
        failed = self.invoke(self.app / "deploy.sh")
        self.assertNotEqual(failed.returncode, 0)
        self.assertTrue(
            "Watcher failed" in failed.stderr
            or "Watcher did not report" in failed.stderr
        )


if __name__ == "__main__":
    unittest.main()
