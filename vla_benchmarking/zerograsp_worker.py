"""External-runtime JSONL worker for the ZeroGrasp experiment.

This file is intentionally an adapter, not a vendored copy of the official
implementation.  The external entry point is supplied at runtime and loaded
once.  Keeping all optional framework imports inside that process preserves
the LIBERO controller's dependency and anti-cheating boundary.

Attribution: this worker is designed to invoke the official ZeroGrasp release
from https://github.com/sh8/ZeroGrasp (Iwase et al., CVPR 2025).  The official
repository and checkpoint must be provided by the operator; neither is copied
into this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from zerograsp_contracts import (
        GRASPNET_CAMERA_FRAME,
        GRASPNET_TRANSLATION_REFERENCE,
        decode_array,
        stable_json_hash,
    )
except ImportError:  # direct invocation from a checkout
    from vla_benchmarking.zerograsp_contracts import (
        GRASPNET_CAMERA_FRAME,
        GRASPNET_TRANSLATION_REFERENCE,
        decode_array,
        stable_json_hash,
    )


PROTOCOL = "zerograsp-jsonl-v1"


def camera_yaml_payload(K: Any) -> dict[str, list[float]]:
    """Build the released fetch_data-compatible YAML camera payload."""
    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("K must be a finite 3x3 matrix")
    return {"left_p": np.c_[matrix, np.zeros(3)].reshape(-1).tolist()}


def _set_seed(seed: int) -> dict[str, Any]:
    """Reset all available inference RNGs and conservative CUDA determinism.

    Torch is imported only inside the external worker process.  Deterministic
    flags constrain cuDNN where supported; they do not claim that every
    third-party CUDA kernel is deterministic.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    status: dict[str, Any] = {"python": True, "numpy": True, "torch": False, "cuda": False, "cudnn_deterministic": False}
    try:
        import torch
    except ImportError:
        return status
    torch.manual_seed(int(seed))
    status["torch"] = True
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and bool(cuda.is_available()):
        cuda.manual_seed_all(int(seed))
        status["cuda"] = True
    try:
        cudnn = torch.backends.cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
        status["cudnn_deterministic"] = True
    except (AttributeError, RuntimeError):
        # Some minimal/test Torch builds have no cuDNN backend.  Seeded RNGs
        # remain active and the status makes the limitation auditable.
        pass
    return status


def _load_entrypoint(spec: str, *, repo: str | None, checkpoint: str | None, config: str | None) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("entrypoint must be module:function")
    module_name, function_name = spec.split(":", 1)
    if repo:
        repo_path = str(Path(repo).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    with contextlib.redirect_stdout(io.StringIO()):
        module = importlib.import_module(module_name)
    loader = getattr(module, function_name)
    if not callable(loader):
        raise TypeError("configured ZeroGrasp entrypoint is not callable")
    # The entrypoint may either be a factory accepting these keyword arguments
    # or a no-argument callable returning an inference callable.
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            runtime = loader(repo=repo, checkpoint=checkpoint, config=config)
        except TypeError:
            runtime = loader()
    if not callable(runtime):
        raise TypeError("configured ZeroGrasp entrypoint must return a callable")
    return runtime


def _load_official_backend(*, repo: str, checkpoint: str, config: str) -> Callable[..., Any]:
    """Load the released demo path without copying its implementation.

    Imports are intentionally local to this external process.  This backend
    follows the public ``demo.py`` path: ``fetch_data`` -> model forward ->
    octree reconstruction -> grasp signal decoding -> collision filtering and
    NMS.  The caller supplies only the explicit image/mask/depth contract.
    """
    import tempfile

    import torch
    from PIL import Image

    repo_path = str(Path(repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from main import BaseTrainer  # type: ignore
    from zerograsp.nets.utils import get_xyz_from_octree  # type: ignore
    from zerograsp.utils.collision_detector import ModelFreeCollisionDetector  # type: ignore
    from zerograsp.utils.dataset import fetch_data  # type: ignore
    from zerograsp.utils.math import rotation_6d_to_matrix, unnormalize_pts  # type: ignore

    try:
        from zerograsp.utils.config import parse_config  # type: ignore
        # Use the release parser so all defaults/CLI compatibility are kept.
        with contextlib.redirect_stdout(io.StringIO()):
            model_config = parse_config(config)
    except Exception as exc:
        raise RuntimeError("official runtime config could not be parsed by the released parser") from exc
    model_config.checkpoint = checkpoint
    model_config.update_octree = True
    model_config.img_height = int(getattr(model_config, "img_height", 1024))
    model_config.img_width = int(getattr(model_config, "img_width", 1280))
    model_config.use_gt_depth = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        model = BaseTrainer.load_from_checkpoint(checkpoint, config=model_config, strict=False)
    model = model.to(device).eval()

    def predict(request: Mapping[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="zerograsp-request-") as work:
            rgb_path, depth_path, mask_path, camera_path = (Path(work) / name for name in ("input.png", "depth.png", "mask.png", "camera.yml"))
            Image.fromarray(np.asarray(request["clean_rgb"], dtype=np.uint8)).save(rgb_path)
            # The released fetch_data path consumes depth in the configured
            # source units (the official demo uses millimetres).
            depth_mm = np.nan_to_num(np.asarray(request["depth_m"], dtype=np.float32) * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
            Image.fromarray(np.clip(depth_mm, 0, 65535).astype(np.uint16)).save(depth_path)
            Image.fromarray(np.asarray(request["source_mask"], dtype=np.uint8)).save(mask_path)
            # The released fetch_data implementation's extension branch
            # requires YAML ``left_p`` (its JSON branch falls through to the
            # YAML error path), so use a valid 3x4 projection matrix here.
            K = np.asarray(request["K_model"], dtype=float)
            with open(camera_path, "w", encoding="utf-8") as handle:
                json.dump(camera_yaml_payload(K), handle)
            batch = fetch_data(str(rgb_path), str(depth_path), str(mask_path), str(camera_path), model_config, 1.0, device=device)
            with torch.no_grad():
                output = model.model(batch)
                z_min = batch[-2][0]
                pts_3d_in = batch[3][0]
                rays_3d = batch[4][0]
                octrees_out = output["octrees_out"]
                grid_res = 1 << int(model_config.min_lod)
                pcd, batch_id = get_xyz_from_octree(octrees_out, int(model_config.max_lod), nempty=True, return_batch=True)
                pcd = unnormalize_pts(pcd, z_min, float(model_config.grid_size), grid_res)
                normals = octrees_out.normals[int(model_config.max_lod)]
                signal = octrees_out.features[int(model_config.max_lod)]
                sdf = signal[:, :1]
                batch_id = batch_id.cpu().numpy()
                pcd = pcd.cpu().numpy()
                normals = torch.nn.functional.normalize(normals, dim=-1).cpu().numpy()
                sdf = sdf.cpu().numpy()
                pcd = pcd - normals * sdf
                # fetch_data receives one source mask, so object label zero is
                # the only candidate group exposed to this adapter.
                object_mask = batch_id == 0
                masked_pcd = pcd[object_mask]
                if len(masked_pcd) == 0:
                    raise RuntimeError("official model returned no reconstructed source points")
                masked_signal = signal[object_mask, 1:]
                quality = masked_signal[:, :1].cpu().numpy()
                tangent = masked_signal[:, 2:5]
                gnormal = masked_signal[:, 5:8]
                R = rotation_6d_to_matrix(torch.cat([-gnormal, tangent], dim=-1)).cpu().numpy()
                depth = masked_signal[:, 8:9].cpu().numpy()
                width = masked_signal[:, 9:10].cpu().numpy()
                translation = masked_pcd / 1000.0
                grasp_preds = np.concatenate([quality, np.clip(width * 0.1, 0.0, 0.1), 0.02 * np.ones_like(quality), np.clip(depth * 0.04, 0.0, 0.04), R.reshape(-1, 9), translation.reshape(-1, 3), -1 * np.ones_like(quality)], axis=-1)
                try:
                    from graspnetAPI import GraspGroup  # type: ignore
                    gg = GraspGroup(grasp_preds).sort_by_score()
                    depth_pcd = rays_3d.reshape(-1, 3)[::5].cpu().numpy()
                    cloud = torch.from_numpy(pcd / 1000.0).float().to(device)
                    cloud_nrm = torch.from_numpy(normals).float().to(device)
                    depth_cloud = torch.from_numpy(depth_pcd / 1000.0).float().to(device)
                    detector = ModelFreeCollisionDetector(cloud, cloud_nrm, depth_cloud)
                    collision_mask, delta_width, refined_depth = detector.detect(gg)
                    gg.grasp_group_array[:, 1] = gg.grasp_group_array[:, 1] + delta_width
                    gg.grasp_group_array[:, 3] = refined_depth
                    gg = gg[~collision_mask] if (~collision_mask).sum() > 0 else gg[:0]
                    gg = gg.nms(0.03, 30.0 / 180.0 * np.pi).sort_by_score()
                    grasp_array = np.asarray(gg.grasp_group_array, dtype=np.float64)
                    collision_free = [True] * len(grasp_array)
                except Exception:
                    # Collision tooling is optional in some external builds;
                    # do not claim collision acceptance without it.
                    grasp_array = grasp_preds
                    collision_free = [False] * len(grasp_array)
                matrices = []
                widths_out, heights_out, depths_out, scores_out = [], [], [], []
                for row in grasp_array:
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = row[4:13].reshape(3, 3)
                    T[:3, 3] = row[13:16]
                    matrices.append(T.tolist())
                    scores_out.append(float(row[0])); widths_out.append(float(row[1])); heights_out.append(float(row[2])); depths_out.append(float(row[3]))
                bounds_min = np.min(masked_pcd, axis=0) / 1000.0
                bounds_max = np.max(masked_pcd, axis=0) / 1000.0
                return {"grasps": matrices, "scores": scores_out, "width_m": widths_out, "height_m": heights_out, "depth_m": depths_out, "collision_free": collision_free, "translation_reference": [GRASPNET_TRANSLATION_REFERENCE] * len(matrices), "translation_frame": [GRASPNET_CAMERA_FRAME] * len(matrices), "rotation_frame": [GRASPNET_CAMERA_FRAME] * len(matrices), "reconstruction": {"dimensions_m": (bounds_max - bounds_min).tolist(), "centroid_camera_m": (np.mean(masked_pcd, axis=0) / 1000.0).tolist(), "bounds_camera_m": np.r_[bounds_min, bounds_max].tolist(), "confidence": 1.0, "source_frame": "camera"}, "diagnostics": {"backend": "official_demo_path", "collision_filter": bool(any(collision_free))}}
    return predict


def _decode_request(request: Mapping[str, Any]) -> dict[str, Any]:
    required = {"clean_rgb", "depth_m", "source_mask", "destination_mask", "K_model", "source_px_model", "destination_px_model", "request_hash", "fixed_seed"}
    request_keys = set(request)
    missing = required - request_keys
    if missing:
        raise ValueError(f"inference request missing fields: {sorted(missing)}")
    # Keep the worker boundary closed: only the documented RGB-D inference
    # payload may cross into the external runtime.  ``type`` and
    # ``request_id`` are protocol metadata emitted by the adapter and are
    # accepted for framing, but task/evaluator/simulator metadata is never
    # silently forwarded (or ignored).
    allowed = required | {"type", "request_id"}
    unknown = request_keys - allowed
    if unknown:
        raise ValueError(f"inference request contains unsupported fields: {sorted(unknown)}")
    if "type" in request and request["type"] != "infer":
        raise ValueError("inference request type must be 'infer'")
    if "request_id" in request:
        request_id = request["request_id"]
        if isinstance(request_id, bool) or not isinstance(request_id, (int, np.integer)) or int(request_id) < 0:
            raise ValueError("request_id must be a non-negative integer")
    decoded = {name: decode_array(request[name]) for name in ("clean_rgb", "depth_m", "source_mask", "destination_mask", "K_model")}
    if decoded["clean_rgb"].dtype != np.uint8 or decoded["clean_rgb"].ndim != 3 or decoded["clean_rgb"].shape[-1] != 3:
        raise ValueError("clean_rgb must be uint8 HxWx3")
    if decoded["depth_m"].ndim != 2 or decoded["depth_m"].shape != decoded["clean_rgb"].shape[:2]:
        raise ValueError("depth_m must align with clean_rgb")
    if decoded["source_mask"].dtype != np.bool_ or decoded["destination_mask"].dtype != np.bool_:
        raise ValueError("source and destination masks must be boolean")
    if decoded["source_mask"].shape != decoded["depth_m"].shape or decoded["destination_mask"].shape != decoded["depth_m"].shape:
        raise ValueError("masks must align with depth_m")
    if np.any(decoded["source_mask"] & decoded["destination_mask"]):
        raise ValueError("source and destination masks must be disjoint")
    decoded.update({"source_px_model": tuple(float(x) for x in request["source_px_model"]), "destination_px_model": tuple(float(x) for x in request["destination_px_model"]), "request_hash": str(request["request_hash"]), "fixed_seed": int(request["fixed_seed"])})
    return decoded


def _normalise_output(raw: Any) -> dict[str, Any]:
    """Convert supported released-inference return forms to strict JSON data."""
    if not isinstance(raw, Mapping):
        raise ValueError("external ZeroGrasp inference must return a mapping")
    allowed = {"grasps", "scores", "width_m", "height_m", "depth_m", "clearance_m", "collision_free", "source_index", "eef_frame", "translation_reference", "translation_frame", "rotation_frame", "reconstruction", "diagnostics"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"external inference returned unsupported fields: {sorted(unknown)}")
    if "grasps" not in raw:
        raise ValueError("external inference did not return grasps")
    grasps = np.asarray(raw["grasps"])
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or not np.issubdtype(grasps.dtype, np.number) or not np.all(np.isfinite(grasps)):
        raise ValueError("external grasps must have finite numeric shape (N,4,4)")
    result: dict[str, Any] = {"grasps": grasps.tolist(), "scores": None if raw.get("scores") is None else np.asarray(raw["scores"], dtype=float).reshape(-1).tolist()}
    def per_grasp_values(name: str, *, required: bool = False) -> list[Any] | None:
        if name not in raw or raw[name] is None:
            if required:
                raise ValueError(f"external inference requires per-grasp {name}")
            return None
        if isinstance(raw[name], (str, bytes)):
            raise ValueError(f"{name} must contain one value per grasp")
        values = list(raw[name])
        if len(values) != len(grasps):
            raise ValueError(f"{name} must contain one value per grasp")
        return values

    for name in ("width_m", "height_m", "depth_m", "clearance_m", "collision_free", "source_index", "eef_frame"):
        values = per_grasp_values(name)
        if values is not None:
            result[name] = values
    for name, expected in (
        ("translation_reference", GRASPNET_TRANSLATION_REFERENCE),
        ("translation_frame", GRASPNET_CAMERA_FRAME),
        ("rotation_frame", GRASPNET_CAMERA_FRAME),
    ):
        values = per_grasp_values(name, required=True)
        assert values is not None
        if any(not isinstance(value, str) or value != expected for value in values):
            raise ValueError(f"{name} must be exactly {expected}")
        result[name] = values
    reconstruction = raw.get("reconstruction")
    if reconstruction is not None:
        if not isinstance(reconstruction, Mapping) or set(reconstruction) - {"dimensions_m", "confidence", "source_frame", "centroid_camera_m", "bounds_camera_m"}:
            raise ValueError("unsupported reconstruction fields")
        if "source_frame" not in reconstruction or reconstruction["source_frame"] is None:
            raise ValueError("external reconstruction requires explicit source_frame")
        if reconstruction["source_frame"] != "camera":
            raise ValueError("external reconstruction source_frame must be exactly camera")
        dimensions = np.asarray(reconstruction.get("dimensions_m"), dtype=float).reshape(-1)
        if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
            raise ValueError("reconstruction dimensions_m must be three positive finite values")
        result["reconstruction"] = {"dimensions_m": dimensions.tolist(), "confidence": float(reconstruction.get("confidence", 0.0)), "source_frame": "camera"}
        for name, size in (("centroid_camera_m", 3), ("bounds_camera_m", 6)):
            if reconstruction.get(name) is not None:
                values = np.asarray(reconstruction[name], dtype=float).reshape(-1)
                if values.size != size or not np.all(np.isfinite(values)):
                    raise ValueError(f"invalid reconstruction {name}")
                result["reconstruction"][name] = values.tolist()
    result["diagnostics"] = dict(raw.get("diagnostics", {}))
    return result


class ExternalZeroGraspRuntime:
    def __init__(self, entrypoint: str, *, repo: str | None = None, checkpoint: str | None = None, config: str | None = None, seed: int = 0):
        self.seed = int(seed)
        self.determinism = _set_seed(self.seed)
        if entrypoint:
            self.predictor = _load_entrypoint(entrypoint, repo=repo, checkpoint=checkpoint, config=config)
        elif repo and checkpoint and config:
            self.predictor = _load_official_backend(repo=repo, checkpoint=checkpoint, config=config)
        else:
            raise ValueError("official backend requires repo, checkpoint, and config, or provide --entrypoint")

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        decoded = _decode_request(request)
        self.determinism = _set_seed(self.seed)
        # External runtime sees only the contract payload.  No task metadata
        # is added here and no simulator objects can cross this boundary.
        raw = self.predictor(decoded)
        result = _normalise_output(raw)
        diagnostics = dict(result.get("diagnostics", {}))
        diagnostics.setdefault("determinism", {"fixed_seed": self.seed, **self.determinism})
        result["diagnostics"] = diagnostics
        return result


def serve(*, entrypoint: str | None = None, repo: str | None = None, checkpoint: str | None = None, config: str | None = None, seed: int = 0) -> int:
    runtime: ExternalZeroGraspRuntime | None = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if not isinstance(message, Mapping):
                raise ValueError("JSONL message must be an object")
            if message.get("type") == "handshake":
                if message.get("protocol") != PROTOCOL:
                    raise ValueError("unsupported protocol")
                if int(message.get("seed", seed)) != int(seed):
                    raise ValueError("worker seed mismatch")
                determinism = _set_seed(seed)
                if not entrypoint and not (repo and checkpoint and config):
                    raise ValueError("external repo, checkpoint, and config are required when no entrypoint is supplied")
                if runtime is None:
                    runtime = ExternalZeroGraspRuntime(entrypoint, repo=repo, checkpoint=checkpoint, config=config, seed=seed)
                print(json.dumps({"type": "ready", "protocol": PROTOCOL, "seed": seed, "determinism": determinism, "attribution": "ZeroGrasp official release; external repository/checkpoint not vendored"}, separators=(",", ":")), flush=True)
            elif message.get("type") == "infer":
                if runtime is None:
                    raise RuntimeError("handshake required before inference")
                result = runtime.infer(message)
                request_hash = str(message["request_hash"])
                result.update({"type": "result", "request_hash": request_hash})
                result["output_hash"] = stable_json_hash(result)
                print(json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
            else:
                raise ValueError("unknown JSONL message type")
        except Exception as exc:
            print(json.dumps({"type": "error", "error": str(exc), "traceback": traceback.format_exc(limit=2)}, separators=(",", ":")), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="External ZeroGrasp JSONL worker")
    parser.add_argument("--entrypoint", default=os.environ.get("ZEROGRASP_ENTRYPOINT"))
    parser.add_argument("--repo", "--root", dest="repo", default=os.environ.get("ZERO_GRASP_ROOT"))
    parser.add_argument("--checkpoint", default=os.environ.get("ZERO_GRASP_CHECKPOINT"))
    parser.add_argument("--config", default=os.environ.get("ZERO_GRASP_CONFIG"))
    parser.add_argument("--env-lock", default=None, help="optional external environment lock file recorded in handshake")
    parser.add_argument("--handshake", action="store_true", help="validate external paths and print a machine-readable preflight result")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ZERO_GRASP_SEED", os.environ.get("ZEROGRASP_FIXED_SEED", "0"))))
    args = parser.parse_args(argv)
    if args.handshake:
        result = {"ok": bool(args.repo and args.checkpoint and args.config and Path(args.repo).exists() and Path(args.checkpoint).exists() and Path(args.config).exists()), "protocol": PROTOCOL, "root": args.repo, "checkpoint": args.checkpoint, "config": args.config, "env_lock": args.env_lock, "seed": args.seed}
        print(json.dumps(result, separators=(",", ":")))
        return 0 if result["ok"] else 2
    return serve(entrypoint=args.entrypoint, repo=args.repo, checkpoint=args.checkpoint, config=args.config, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
