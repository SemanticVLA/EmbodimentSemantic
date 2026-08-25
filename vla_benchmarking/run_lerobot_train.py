"""Run LeRobot training with deterministic cuDNN flags set before imports."""
from __future__ import annotations

import runpy
import sys

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
