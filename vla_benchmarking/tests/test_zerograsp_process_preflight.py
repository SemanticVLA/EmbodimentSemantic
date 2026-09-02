from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "legion" / "preflight_zerograsp_process.py"


def test_process_preflight_waits_ready_then_closes_without_inference(tmp_path):
    repo = tmp_path / "zerograsp"
    repo.mkdir()
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixture-not-deserialized")
    config = tmp_path / "demo.yaml"
    config.write_text("model_name: test\n", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    if message.get('type') == 'handshake':\n"
        "        print(json.dumps({'type': 'ready', 'protocol': 'zerograsp-jsonl-v1'}), flush=True)\n"
        "    elif message.get('type') == 'infer':\n"
        "        raise SystemExit('inference must not be sent by preflight')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--python",
            sys.executable,
            "--worker",
            str(worker),
            "--repo",
            str(repo),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["protocol"] == "zerograsp-jsonl-v1"
    assert payload["protocol_ready"] is True
    assert payload["closed"] is True
    assert payload["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_process_preflight_rejects_missing_or_mismatched_checkpoint_digest(tmp_path):
    repo = tmp_path / "zerograsp"
    repo.mkdir()
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixture-not-deserialized")
    config = tmp_path / "demo.yaml"
    config.write_text("model_name: test\n", encoding="utf-8")
    started = tmp_path / "worker-started"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "from pathlib import Path\n"
        f"Path({str(started)!r}).write_text('started')\n"
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    print(json.dumps({'type': 'ready', 'protocol': 'zerograsp-jsonl-v1'}), flush=True)\n",
        encoding="utf-8",
    )
    common = [
        sys.executable,
        str(HELPER),
        "--python", sys.executable,
        "--worker", str(worker),
        "--repo", str(repo),
        "--checkpoint", str(checkpoint),
        "--config", str(config),
    ]
    missing = subprocess.run(common, capture_output=True, text=True, check=False)
    assert missing.returncode != 0
    assert not started.exists()
    mismatch = subprocess.run(common + ["--checkpoint-sha256", "0" * 64], capture_output=True, text=True, check=False)
    assert mismatch.returncode == 2
    assert "checkpoint SHA-256 mismatch" in mismatch.stderr
    assert not started.exists()


def test_process_preflight_reaches_ready_with_production_worker_cli(tmp_path):
    """The helper must not pass its private digest flag to worker argparse."""
    repo = tmp_path / "zerograsp"
    repo.mkdir()
    (repo / "fixture_backend.py").write_text(
        "def factory(repo=None, checkpoint=None, config=None):\n"
        "    return lambda request: {}\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixture-not-deserialized")
    config = tmp_path / "demo.yaml"
    config.write_text("model_name: test\n", encoding="utf-8")
    worker = Path(__file__).resolve().parents[1] / "zerograsp_worker.py"
    env = dict(os.environ)
    env["ZEROGRASP_ENTRYPOINT"] = "fixture_backend:factory"
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--python", sys.executable,
            "--worker", str(worker),
            "--repo", str(repo),
            "--checkpoint", str(checkpoint),
            "--checkpoint-sha256", hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--config", str(config),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["protocol_ready"] is True
    assert payload["closed"] is True
