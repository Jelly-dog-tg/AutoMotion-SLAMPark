
"""Convert a probabilistic SLAM map into a planner collision grid."""

import math
import numpy as np

from parking_planner.planning.hybrid_astar import OccupancyGrid


def build_planner_grid(
    mapper,
    planner_resolution=0.20,
    unknown_is_occupied=True,
):
    """
    Convert SLAM occupancy map into Hybrid A* occupancy grid.

    SLAM states:
        -1: unknown
         0: free
         1: occupied

    Conservative conversion:
        A planner cell is blocked if any source cell
        inside it is occupied or unknown.
    """

    if planner_resolution <= 0:
        raise ValueError("Invalid planner resolution")

    source_resolution = float(mapper.resolution)

    ratio = planner_resolution / source_resolution
    factor = int(round(ratio))

    if factor < 1 or not math.isclose(
        ratio, factor, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError(
            "Planner resolution must be an integer "
            "multiple of SLAM resolution"
        )

    states = np.asarray(mapper.states, dtype=np.int8)

    if states.ndim != 2:
        raise ValueError("SLAM map must be 2D")

    target_rows = math.ceil(
        mapper.height / planner_resolution
    )

    target_cols = math.ceil(
        mapper.width / planner_resolution
    )

    # Pad incomplete boundary blocks as unknown.
    padded = np.full(
        (target_rows * factor, target_cols * factor),
        -1,
        dtype=np.int8,
    )

    rows = min(states.shape[0], padded.shape[0])
    cols = min(states.shape[1], padded.shape[1])

    padded[:rows, :cols] = states[:rows, :cols]

    if unknown_is_occupied:
        blocked_source = padded != 0
    else:
        blocked_source = padded == 1

    # Conservative max pooling.
    blocked = blocked_source.reshape(
        target_rows,
        factor,
        target_cols,
        factor,
    ).any(axis=(1, 3))

    grid = OccupancyGrid(
        width=mapper.width,
        height=mapper.height,
        resolution=planner_resolution,
        origin_x=mapper.origin_x,
        origin_y=mapper.origin_y,
    )

    if blocked.shape != grid.occupied.shape:
        raise RuntimeError("Planner map dimensions mismatch")

    grid.occupied[:] = blocked

    return grid
