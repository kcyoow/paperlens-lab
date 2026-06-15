import subprocess
import tempfile
import unittest
from pathlib import Path

from paperlens_lab.implementation_repo import inspect_implementation_repository


class ImplementationRepoTests(unittest.TestCase):
    def test_rejects_non_source_listed_github_root_shape(self):
        result = inspect_implementation_repository(
            {"url": "https://example.com/not/github"},
            base_dir=Path(tempfile.mkdtemp()),
            runner=_failing_runner,
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["execution"], "none")

    def test_inspects_manifest_without_executing_repo_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calls = []

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                if command[:2] == ["git", "clone"]:
                    clone_dir = Path(command[-1])
                    clone_dir.mkdir(parents=True)
                    (clone_dir / ".git").mkdir()
                    (clone_dir / "README.md").write_text("# LoRA\nRead-only manifest.\n", encoding="utf-8")
                    (clone_dir / "LICENSE").write_text("MIT License\n", encoding="utf-8")
                    (clone_dir / "pyproject.toml").write_text("[project]\nname='lora'\n", encoding="utf-8")
                    (clone_dir / "examples").mkdir()
                    (clone_dir / "examples" / "demo.py").write_text("print('not executed')\n", encoding="utf-8")
                    self.assertEqual(kwargs["env"]["GIT_LFS_SKIP_SMUDGE"], "1")
                    self.assertIn("--no-recurse-submodules", command)
                    return subprocess.CompletedProcess(command, 0, "", "")
                if "--abbrev-ref" in command:
                    return subprocess.CompletedProcess(command, 0, "main\n", "")
                if command[-1] == "HEAD":
                    return subprocess.CompletedProcess(command, 0, "abc123\n", "")
                return subprocess.CompletedProcess(command, 1, "", "unexpected command")

            result = inspect_implementation_repository(
                {
                    "source_id": "implementation:github:1",
                    "url": "https://github.com/microsoft/LoRA",
                    "source_url": "https://github.com/microsoft/LoRA",
                },
                base_dir=Path(tmpdir),
                runner=runner,
            )

        self.assertEqual(result["status"], "inspected")
        self.assertEqual(result["execution"], "none")
        self.assertEqual(result["commit"], "abc123")
        self.assertEqual(result["default_branch"], "main")
        self.assertEqual(result["readme"]["path"], "README.md")
        self.assertEqual(result["license"]["path"], "LICENSE")
        self.assertIn("README.md", [item["path"] for item in result["files"]])
        self.assertIn("pyproject.toml", [item["path"] for item in result["files"]])
        self.assertEqual([call[0][1] for call in calls].count("clone"), 1)


def _failing_runner(*args, **kwargs):
    raise AssertionError("runner should not be called for rejected URLs")
