"""Display-only defaults. Never alter scenario start, path or geometry."""
import numpy as np

WINDOW_OPTIONS = '--width=1100 --height=800'


def preview_pose(scenario):
    start = scenario['start']
    return np.array([start.x, start.y-1.2, start.yaw])
