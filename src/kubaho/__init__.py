"""Kubaho: camera-free WiFi-CSI fall and wellness monitoring.

Core ML package shared by the notebook, the training scripts and the API.
"""

__version__ = "0.1.0"

CLASSES = [
    "fall",
    "sit_lie_mat",
    "kneel_pray",
    "squat_stool",
    "walk",
    "bend_sweep",
    "restless_bed",
    "empty",
]

# Everyday floor-level activities that generic detectors confuse with falls.
FLOOR_LEVEL = ["sit_lie_mat", "kneel_pray", "squat_stool", "bend_sweep", "restless_bed"]
