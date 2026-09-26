"""First-frame SAM text-name probe for the SO101 task inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from samgraph_core.automatic_scene import AutomaticMaskAcquirer

from .config import SO101Config
from .dataset import SO101Dataset, sha256_file
from .render import render_mask_overlay
from .scene import install_so101_catalog


def run_name_probe(dataset: SO101Dataset, config: SO101Config, checkpoint: Path,
                   output: Path, task_ids: list[str], episode: int = 0) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    install_so101_catalog(config)
    from samgraph_core import LocalSam31Segmenter, OfficialSam31Runtime
    runtime = OfficialSam31Runtime(checkpoint, text_detection_threshold=0.2)
    segmenter = LocalSam31Segmenter(runtime)
    warmup = getattr(runtime, "warmup", None)
    if callable(warmup):
        warmup()
    results = {}
    complete = True
    try:
        prompts = config.prompts_for_tasks(task_ids)
        for task in task_ids:
            iterator = dataset.iter_frames(task, episode, "agent_view")
            try:
                frame = next(iterator)
            finally:
                iterator.close()
            acquirer = AutomaticMaskAcquirer(
                segmenter, capture_candidates=True, prompts_by_class=prompts[task],
            )
            selected = acquirer.acquire(frame.rgb)
            selected_masks = {
                class_id: proposals[0].mask for class_id, proposals in selected.items()
                if proposals
            }
            missing = sorted(set(config.object_ids) - set(selected_masks))
            candidate_conflicts = {}
            for missing_id in missing:
                rows = []
                for evidence in acquirer.candidate_evidence:
                    if evidence["class_id"] != missing_id:
                        continue
                    candidate = acquirer.candidate_masks[evidence["mask_key"]]
                    candidate_area = int(candidate.sum())
                    conflicts = []
                    for selected_id, selected_mask in selected_masks.items():
                        intersection = int(np.logical_and(candidate, selected_mask).sum())
                        if not intersection:
                            continue
                        selected_area = int(selected_mask.sum())
                        conflicts.append({
                            "object_id": selected_id,
                            "intersection_pixels": intersection,
                            "candidate_overlap_fraction": intersection / candidate_area,
                            "selected_overlap_fraction": intersection / selected_area,
                            "minimum_area_overlap_fraction": intersection / min(
                                candidate_area, selected_area,
                            ),
                        })
                    rows.append({
                        "prompt": evidence["prompt"], "score": evidence["score"],
                        "area": candidate_area, "window_xywh": evidence["window_xywh"],
                        "conflicts_with_selected": conflicts,
                    })
                candidate_conflicts[missing_id] = sorted(
                    rows,
                    key=lambda row: (
                        -max((item["minimum_area_overlap_fraction"]
                              for item in row["conflicts_with_selected"]), default=0.0),
                        -(row["score"] if row["score"] is not None else -1.0),
                    ),
                )
            complete &= not missing
            task_root = output / task / f"episode_{episode}"
            task_root.mkdir(parents=True)
            np.savez_compressed(task_root / "selected_masks.npz", **selected_masks)
            states = [{"track_id": key, "output_id": key, "status": "observed"}
                      for key in selected_masks]
            Image.fromarray(render_mask_overlay(frame.rgb, selected_masks, states)).save(
                task_root / "selected_overlay.png"
            )
            candidates_root = task_root / "candidates"
            candidates_root.mkdir()
            candidate_rows = []
            for index, evidence in enumerate(acquirer.candidate_evidence):
                mask = acquirer.candidate_masks[evidence["mask_key"]]
                name = f"{index:03d}_{evidence['class_id']}.png"
                overlay = render_mask_overlay(
                    frame.rgb, {evidence["class_id"]: mask},
                    [{"track_id": evidence["class_id"], "output_id": evidence["class_id"],
                      "status": "observed"}],
                )
                Image.fromarray(overlay).save(candidates_root / name)
                candidate_rows.append({**evidence, "overlay": f"candidates/{name}"})
            selected_rows = {
                class_id: [{
                    "prompt": proposal.prompt,
                    "score": proposal.score,
                    "prompt_votes": proposal.prompt_votes,
                    "area": proposal.area,
                    "mask_sha256": hashlib.sha256(proposal.mask.tobytes()).hexdigest(),
                } for proposal in proposals]
                for class_id, proposals in selected.items()
            }
            result = {
                "task": task,
                "episode": episode,
                "frame": 0,
                "source_sha256": frame.source_sha256,
                "trajectory_timestamp_s": frame.trajectory_timestamp_s,
                "video_timestamp_s": frame.video_timestamp_s,
                "timestamp_error_s": frame.timestamp_error_s,
                "selected": selected_rows,
                "missing_object_ids": missing,
                "complete": not missing,
                "diagnostics": acquirer.diagnostics,
                "candidate_conflicts_with_selected": candidate_conflicts,
                "duplicate_mask_rejection_threshold": 0.90,
                "candidates": candidate_rows,
                "selected_masks_sha256": sha256_file(task_root / "selected_masks.npz"),
            }
            (task_root / "probe.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            results[task] = result
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
    manifest = {
        "schema": "samgraph.so101_name_probe.v1",
        "tasks": task_ids,
        "episode": episode,
        "objects": list(config.object_ids),
        "object_config_sha256": config.sha256,
        "complete_initial_localization": complete,
        "results": {
            task: {"complete": row["complete"], "missing_object_ids": row["missing_object_ids"]}
            for task, row in results.items()
        },
        "sam": dict(getattr(runtime, "model_identity", {})),
        "manual_points_boxes_or_masks": False,
        "evaluation_performed": False,
    }
    (output / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["run_name_probe"]
