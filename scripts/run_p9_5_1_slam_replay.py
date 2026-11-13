
"""P9.5.1: Incremental SLAM replay using LiDAR scans and odometry."""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.localization.point_to_line_icp import (
    PointToLineConfig,
    point_to_line_icp,
)
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper
from parking_planner.sensing.lidar_2d import LidarConfig
from parking_planner.vehicle.ackermann import VehicleState


INPUT_PATH = ROOT / "results" / "p9_occupancy_maps.npz"
OUTPUT_DIR = ROOT / "results"

MAP_OUTPUT = OUTPUT_DIR / "p9_5_1_slam_map.npz"
METRICS_OUTPUT = OUTPUT_DIR / "p9_5_1_slam_metrics.json"
FIGURE_OUTPUT = OUTPUT_DIR / "p9_5_1_slam_replay.png"


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_to_matrix(pose):
    x, y, yaw = map(float, pose)
    c, s = math.cos(yaw), math.sin(yaw)

    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0.0, 0.0, 1.0],
    ])


def matrix_to_pose(transform):
    return np.array([
        transform[0, 2],
        transform[1, 2],
        math.atan2(transform[1, 0], transform[0, 0]),
    ])


def relative_pose(previous_pose, current_pose):
    """Current pose expressed in the previous pose frame."""

    return matrix_to_pose(
        np.linalg.inv(pose_to_matrix(previous_pose))
        @ pose_to_matrix(current_pose)
    )


def compose_pose(global_pose, relative_increment):
    """Apply body-frame odometry increment to a global pose."""

    return matrix_to_pose(
        pose_to_matrix(global_pose)
        @ pose_to_matrix(relative_increment)
    )


def scan_to_local_points(angles_rad, ranges_m, valid):
    angles = np.asarray(angles_rad)[valid]
    ranges = np.asarray(ranges_m)[valid]

    return np.column_stack((
        ranges * np.cos(angles),
        ranges * np.sin(angles),
    ))


def nearest_distances(points_a, points_b):
    if len(points_a) == 0:
        return np.empty(0)

    if len(points_b) == 0:
        return np.full(len(points_a), np.inf)

    chunks = []

    for chunk in np.array_split(
        points_a,
        max(1, math.ceil(len(points_a) / 256)),
    ):
        if len(chunk):
            distances_sq = np.sum(
                (chunk[:, None, :] - points_b[None, :, :]) ** 2,
                axis=2,
            )
            chunks.append(
                np.sqrt(np.min(distances_sq, axis=1))
            )

    return np.concatenate(chunks)


def evaluate_map(mapper, true_boundary, visible_boundary):
    states = mapper.states
    rows, cols = np.where(states == 1)

    occupied_points = np.column_stack((
        mapper.origin_x + (cols + 0.5) * mapper.resolution,
        mapper.origin_y + (rows + 0.5) * mapper.resolution,
    ))

    precision = (
        float(np.mean(
            nearest_distances(
                occupied_points, true_boundary
            ) <= 0.15
        ))
        if len(occupied_points) else 0.0
    )

    visible_truth = true_boundary[visible_boundary]

    recall = (
        float(np.mean(
            nearest_distances(
                visible_truth, occupied_points
            ) <= 0.15
        ))
        if len(visible_truth) and len(occupied_points)
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "occupied_cells": int(np.sum(states == 1)),
        "free_cells": int(np.sum(states == 0)),
        "unknown_cells": int(np.sum(states == -1)),
        "boundary_precision": precision,
        "visible_boundary_recall": recall,
        "boundary_f1": f1,
    }


def pose_error_metrics(estimated, reference):
    translation = np.linalg.norm(
        estimated[:, :2] - reference[:, :2],
        axis=1,
    )

    heading = np.degrees(np.array([
        wrap_angle(a - b)
        for a, b in zip(
            estimated[:, 2],
            reference[:, 2],
            strict=True,
        )
    ]))

    return {
        "translation_rmse_m": float(
            np.sqrt(np.mean(translation ** 2))
        ),
        "translation_median_m": float(
            np.median(translation)
        ),
        "final_translation_error_m": float(
            translation[-1]
        ),
        "heading_rmse_deg": float(
            np.sqrt(np.mean(heading ** 2))
        ),
        "heading_median_abs_deg": float(
            np.median(np.abs(heading))
        ),
        "final_heading_error_deg": float(
            abs(heading[-1])
        ),
    }


def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run P9.3 first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load raw LiDAR scans and odometry measurements
    # --------------------------------------------------

    with np.load(INPUT_PATH, allow_pickle=False) as data:

        required = {
            "angles_rad",
            "ranges_m",
            "valid",
            "odometry_poses",
            "reference_poses",
            "true_boundary",
            "visible_boundary",
            "icp_config",
        }

        missing = required - set(data.files)

        if missing:
            raise KeyError(
                f"Missing archive fields: {sorted(missing)}"
            )

        angles_rad = data["angles_rad"].copy()
        ranges_m = data["ranges_m"].copy()
        valid = data["valid"].astype(bool, copy=True)

        odometry_poses = data["odometry_poses"].copy()
        reference_poses = data["reference_poses"].copy()

        true_boundary = data["true_boundary"].copy()
        visible_boundary = data["visible_boundary"].astype(
            bool, copy=True
        )

        mapper_config = json.loads(
            str(data["icp_config"].item())
        )

    scan_count = len(ranges_m)

    if ranges_m.ndim != 2 or valid.shape != ranges_m.shape:
        raise ValueError("Invalid LiDAR array shapes.")

    if ranges_m.shape[1] != len(angles_rad):
        raise ValueError("LiDAR angle/range mismatch.")

    if odometry_poses.shape != (scan_count, 3):
        raise ValueError("Invalid odometry pose array.")

    if reference_poses.shape != (scan_count, 3):
        raise ValueError("Invalid reference pose array.")

    # --------------------------------------------------
    # 2. Initialize SLAM
    # --------------------------------------------------

    scenario = build_perpendicular_parking_scenario()

    # Initial pose is known.
    # Ground truth trajectory is NOT used for localization.
    initial_pose = np.array([
        scenario["start"].x,
        scenario["start"].y,
        scenario["start"].yaw,
    ], dtype=float)

    if not np.allclose(
        initial_pose,
        odometry_poses[0],
        atol=1e-6,
    ):
        raise ValueError(
            "Initial pose does not match the P9.3 odometry frame."
        )

    mapper = OccupancyGridMapper(**mapper_config)

    lidar_config = LidarConfig()
    icp_config = PointToLineConfig()

    estimated_poses = [initial_pose.copy()]
    pair_records = []

    # First scan initializes the map.
    first_scan = {
        "angles_rad": angles_rad,
        "ranges_m": ranges_m[0],
        "valid": valid[0],
    }

    mapper.update(
        VehicleState(*initial_pose, 0.0),
        first_scan,
        lidar_config,
    )

    print("=" * 60)
    print("P9.5.1 INCREMENTAL SLAM REPLAY")
    print("=" * 60)
    print(f"Scans: {scan_count}")
    print(f"Beams per scan: {ranges_m.shape[1]}")
    print("Localization: Point-to-Line ICP")
    print("Fallback: Odometry prediction")

    # --------------------------------------------------
    # 3. Sequential localization and mapping
    # --------------------------------------------------

    for scan_index in range(1, scan_count):

        # 3.1 Obtain relative odometry measurement.
        odom_increment = relative_pose(
            odometry_poses[scan_index - 1],
            odometry_poses[scan_index],
        )

        # 3.2 Predict current global pose.
        predicted_pose = compose_pose(
            estimated_poses[-1],
            odom_increment,
        )

        # 3.3 Convert current LiDAR scan into local points.
        local_scan = scan_to_local_points(
            angles_rad,
            ranges_m[scan_index],
            valid[scan_index],
        )

        # 3.4 Extract the accumulated map points.
        rows, cols = np.where(mapper.states == 1)

        map_points = np.column_stack((
            mapper.origin_x
            + (cols + 0.5) * mapper.resolution,

            mapper.origin_y
            + (rows + 0.5) * mapper.resolution,
        ))

        result = None
        failure_reason = None

        # 3.5 Scan-to-Map ICP.
        try:
            result = point_to_line_icp(
                scan=local_scan,
                initial_pose=predicted_pose,
                map_points=map_points,
                config=icp_config,
            )

            accepted = bool(result["success"])

            if accepted:
                final_pose = result["pose"].copy()
            else:
                final_pose = predicted_pose.copy()
                failure_reason = result["rejection_reason"]

        except (ValueError, np.linalg.LinAlgError) as error:

            accepted = False
            final_pose = predicted_pose.copy()
            failure_reason = str(error)

        # 3.6 Save estimated pose.
        estimated_poses.append(final_pose.copy())

        # 3.7 Update occupancy grid with selected pose.
        current_scan = {
            "angles_rad": angles_rad,
            "ranges_m": ranges_m[scan_index],
            "valid": valid[scan_index],
        }

        mapper.update(
            VehicleState(*final_pose, 0.0),
            current_scan,
            lidar_config,
        )

        record = {
            "scan_index": scan_index,
            "accepted": accepted,
            "failure_reason": failure_reason,
            "map_points_before_update": int(len(map_points)),
            "scan_points": int(len(local_scan)),
            "correspondences": (
                int(result["correspondences"])
                if result is not None else 0
            ),
            "fitness": (
                float(result["fitness"])
                if result is not None else 0.0
            ),
            "line_rmse_m": (
                result["line_rmse_m"]
                if result is not None else None
            ),
            "information_eigen_ratio": (
                float(result["information_eigen_ratio"])
                if result is not None else None
            ),
            "iterations": (
                int(result["iterations"])
                if result is not None else 0
            ),
            "predicted_pose": predicted_pose.tolist(),
            "estimated_pose": final_pose.tolist(),
        }

        pair_records.append(record)

        status = "ACCEPT" if accepted else "ODOM FALLBACK"

        print(
            f"Scan {scan_index + 1:02d}/{scan_count}: "
            f"{status} | map={len(map_points)} "
            f"| correspondences={record['correspondences']}"
        )

    estimated_poses = np.asarray(
        estimated_poses, dtype=float
    )

    # --------------------------------------------------
    # 4. Build odometry-only baseline
    # --------------------------------------------------

    odometry_baseline = np.zeros(
        (scan_count, 3), dtype=float
    )

    odometry_baseline[0] = initial_pose

    for i in range(1, scan_count):

        delta = relative_pose(
            odometry_poses[i - 1],
            odometry_poses[i],
        )

        odometry_baseline[i] = compose_pose(
            odometry_baseline[i - 1],
            delta,
        )

    # --------------------------------------------------
    # 5. Calculate metrics
    # --------------------------------------------------

    accepted_count = sum(
        item["accepted"] for item in pair_records
    )

    rejected_count = len(pair_records) - accepted_count

    rejection_reasons = Counter(
        item["failure_reason"]
        for item in pair_records
        if not item["accepted"]
    )

    metrics = {
        "experiment": "P9.5.1 Incremental SLAM replay",
        "scan_count": int(scan_count),

        "lidar": {
            "beams": int(ranges_m.shape[1]),
            "fov_deg": lidar_config.fov_deg,
            "max_range_m": lidar_config.max_range_m,
            "range_noise_std_m": lidar_config.range_std_m,
        },

        "localization": {
            "algorithm": (
                "Point-to-Line ICP with Huber loss "
                "and odometry prior"
            ),
            "accepted_updates": int(accepted_count),
            "rejected_updates": int(rejected_count),
            "acceptance_rate": float(
                accepted_count / max(1, len(pair_records))
            ),
            "rejection_reasons": dict(rejection_reasons),
            "fallback": (
                "Use odometry-predicted pose "
                "when ICP rejects an update."
            ),
        },

        "pose_errors": {
            "odometry": pose_error_metrics(
                odometry_baseline,
                reference_poses,
            ),
            "incremental_scan_to_map": pose_error_metrics(
                estimated_poses,
                reference_poses,
            ),
        },

        "map": evaluate_map(
            mapper,
            true_boundary,
            visible_boundary,
        ),

        "data_integrity": {
            "ground_truth_used_for_localization": False,
            "ground_truth_used_for_final_evaluation_only": True,
            "odometry_input": (
                "Relative increments derived from P9.3 odometry_poses"
            ),
            "scan_input": (
                "P9.3 raw ranges_m, valid and angles_rad"
            ),
        },

        "pairs": pair_records,
    }

    # --------------------------------------------------
    # 6. Save SLAM map and trajectory
    # --------------------------------------------------

    archive = mapper.arrays()

    archive.update({
        "estimated_poses": estimated_poses,
        "odometry_poses": odometry_baseline,
        "reference_poses": reference_poses,
        "angles_rad": angles_rad,
        "ranges_m": ranges_m,
        "valid": valid,
    })

    np.savez_compressed(
        MAP_OUTPUT,
        **archive,
    )

    METRICS_OUTPUT.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 7. Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 11),
    )

    # Occupancy map.
    ax = axes[0, 0]

    ax.imshow(
        mapper.states + 1,
        origin="lower",
        extent=[
            mapper.origin_x,
            mapper.origin_x + mapper.width,
            mapper.origin_y,
            mapper.origin_y + mapper.height,
        ],
        cmap=ListedColormap([
            "#999999",
            "white",
            "black",
        ]),
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )

    ax.plot(
        reference_poses[:, 0],
        reference_poses[:, 1],
        "--",
        label="Reference",
    )

    ax.plot(
        estimated_poses[:, 0],
        estimated_poses[:, 1],
        label="Incremental SLAM",
    )

    ax.scatter(
        initial_pose[0],
        initial_pose[1],
        marker="o",
        s=35,
        label="Initial pose",
    )

    ax.set_title("Incremental Occupancy Grid Map")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Trajectory comparison.
    ax = axes[0, 1]

    ax.plot(
        reference_poses[:, 0],
        reference_poses[:, 1],
        "--",
        label="Reference",
    )

    ax.plot(
        odometry_baseline[:, 0],
        odometry_baseline[:, 1],
        label="Odometry",
    )

    ax.plot(
        estimated_poses[:, 0],
        estimated_poses[:, 1],
        label="Scan-to-Map ICP",
    )

    ax.set_title("Trajectory Comparison")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.3)

    # Translation errors.
    translation_errors = np.linalg.norm(
        estimated_poses[:, :2] - reference_poses[:, :2],
        axis=1,
    )

    odometry_translation_errors = np.linalg.norm(
        odometry_baseline[:, :2] - reference_poses[:, :2],
        axis=1,
    )

    ax = axes[1, 0]

    ax.plot(
        translation_errors,
        label="Scan-to-Map ICP",
    )

    ax.plot(
        odometry_translation_errors,
        label="Odometry",
    )

    ax.set_title("Translation Error")
    ax.set_xlabel("Scan index")
    ax.set_ylabel("Error (m)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Heading errors.
    heading_errors = np.degrees(np.array([
        wrap_angle(a - b)
        for a, b in zip(
            estimated_poses[:, 2],
            reference_poses[:, 2],
            strict=True,
        )
    ]))

    odometry_heading_errors = np.degrees(np.array([
        wrap_angle(a - b)
        for a, b in zip(
            odometry_baseline[:, 2],
            reference_poses[:, 2],
            strict=True,
        )
    ]))

    ax = axes[1, 1]

    ax.plot(
        np.abs(heading_errors),
        label="Scan-to-Map ICP",
    )

    ax.plot(
        np.abs(odometry_heading_errors),
        label="Odometry",
    )

    ax.set_title("Absolute Heading Error")
    ax.set_xlabel("Scan index")
    ax.set_ylabel("Error (deg)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        "P9.5.1 Incremental SLAM Replay",
        fontsize=15,
    )

    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)

    # --------------------------------------------------
    # 8. Print summary
    # --------------------------------------------------

    print("\n========== P9.5.1 RESULTS ==========")

    print(json.dumps({
        "accepted_updates": accepted_count,
        "rejected_updates": rejected_count,
        "odometry_pose_errors": metrics["pose_errors"]["odometry"],
        "slam_pose_errors": metrics["pose_errors"]["incremental_scan_to_map"],
        "map": metrics["map"],
        "saved_map": str(MAP_OUTPUT),
        "saved_metrics": str(METRICS_OUTPUT),
        "saved_figure": str(FIGURE_OUTPUT),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
