"""Minimal single mission: nominal scene, seed 0, full autonomy stack, no options.

    python 25879875.py

Builds the room, runs LiDAR mapping + A* planning + EKF (encoders + ArUco pose) + LQR tracking
until the robot docks, then prints the result and writes into outputs/:
    topdown_seed0.png   map, planned/true/estimated paths, 3-sigma ellipses
    mission_seed0.mp4   third-person | onboard camera | top-down, with phase and NEES

The robot software (core/) only ever sees sensor data. Ground truth is read only in evaluation/
(metrics, figure and video).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core.config import load_config  # noqa: E402
from evaluation.metrics import summarize_run  # noqa: E402  (ground truth: evaluation only)
from evaluation.plots import plot_run_topdown  # noqa: E402
from evaluation.runner import run  # noqa: E402
from evaluation.video import VideoRecorder  # noqa: E402
from sim.scene_builder import build_scene  # noqa: E402

SEED = 0
OUT = HERE / "outputs"

cfg = load_config()
video = VideoRecorder(build_scene(cfg, SEED), f"Nominal mission (seed {SEED})")
out = run(cfg, SEED, on_tick=video.on_tick)
s = summarize_run(out, cfg)
print(f"docked: {s['success']}  (robot believed docked: {s['docked_belief']})")
print(f"final position error: {s['final_pos_err_m'] * 100:.1f} cm, heading error: {s['final_heading_err_deg']:.1f} deg")
print(f"sim time: {s['time_s']:.1f} s, collisions: {s['collisions']}")

plot_run_topdown(out, OUT / f"topdown_seed{SEED}.png", command="python 25879875.py")
video.save(OUT / f"mission_seed{SEED}.mp4")
print(f"wrote {OUT / f'topdown_seed{SEED}.png'} and {OUT / f'mission_seed{SEED}.mp4'}")
