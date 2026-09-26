import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).parents[1] / "scripts/evaluate_libero_single_episode.py"
spec = importlib.util.spec_from_file_location("saved_graph_scorer", path)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def test_provisional_bowls_use_existing_vlm_swap_without_mutating_input():
    from vlm_bench.eval import FrameResult
    prediction = [["black_bowl_track_1", "is_left_of", "plate_1"],
                  ["black_bowl_track_2", "is_right_of", "plate_1"]]
    normalized = scorer.scoring_triplets(prediction, bowl_policy="vlm-swap")
    gt = {("akita_black_bowl_2", "is_left_of", "plate_1"),
          ("akita_black_bowl_1", "is_right_of", "plate_1")}
    result = FrameResult(task="task", demo="demo_0", frame=0,
                         camera="agentview", gt=gt, pred=normalized)
    assert result.f1 == 1.0
    assert result.pred == gt
    assert prediction[0][0] == "black_bowl_track_1"
    assert normalized != gt  # the VLM scorer, not the alias map, did the swap
    strict = scorer.scoring_triplets(prediction, bowl_policy="strict")
    assert ("black_bowl_track_1", "is_left_of", "plate_1") in strict


def test_scoring_rejects_colliding_bowl_vocabulary():
    with pytest.raises(ValueError, match="collide"):
        scorer.scoring_triplets([
            ["black_bowl_track_1", "is_left_of", "plate_1"],
            ["black_bowl_1", "is_right_of", "plate_1"],
        ], bowl_policy="vlm-swap")
