
"""P9.2: ICP scan matching evaluation."""

import json
import math
import sys

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)

from parking_planner.planning.hybrid_astar import (
    HybridAStarPlanner,
)

from parking_planner.planning.path_utils import (
    resample_path,
)

from parking_planner.sensing.lidar_2d import (
    Lidar2D,
    LidarConfig,
)

from parking_planner.localization import icp_2d


OUTPUT_DIR = ROOT / "results"


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def relative_current_to_previous(previous, current):
    """
    Ground-truth transformation from current LiDAR frame
    into previous LiDAR frame.
    """

    dx = current.x - previous.x
    dy = current.y - previous.y

    c = math.cos(previous.yaw)
    s = math.sin(previous.yaw)

    translation = np.array([
        c * dx + s * dy,
        -s * dx + c * dy,
    ])

    yaw = normalize_angle(
        current.yaw - previous.yaw
    )

    return yaw, translation


def transform_points(points, yaw, translation):

    c = math.cos(yaw)
    s = math.sin(yaw)

    R = np.array([
        [c, -s],
        [s, c],
    ])

    return points @ R.T + translation


def finite_value(value):
    value = float(value)

    if math.isfinite(value):
        return round(value, 6)

    return None


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(2026)

    scenario = build_perpendicular_parking_scenario()

    # Generate Hybrid A* reference trajectory.
    planner = HybridAStarPlanner(
        grid=scenario["grid"],
        geometry=scenario["geometry"],
        config=scenario["planner_config"],
    )

    raw_path = planner.plan(
        scenario["start"],
        scenario["goal"],
    )

    if raw_path is None:
        raise RuntimeError("Hybrid A* planning failed")

    dense_path = resample_path(
        raw_path,
        spacing=0.1,
    )

    # Sample one LiDAR scan approximately every 0.3 m.
    poses = dense_path[::3]

    if poses[-1] != dense_path[-1]:
        poses.append(dense_path[-1])

    if len(poses) < 2:
        raise RuntimeError("Not enough scan poses")

    # Initialize LiDAR.
    lidar = Lidar2D(
        LidarConfig(
            num_beams=720,
            fov_deg=270.0,
            min_range_m=0.12,
            max_range_m=12.0,
            range_std_m=0.015,
        )
    )

    records = []
    visual_data = None

    for index in range(1, len(poses)):

        previous = poses[index - 1]
        current = poses[index]

        # Simulate consecutive scans.
        previous_scan = lidar.scan(
            previous,
            scenario["parked_vehicles"],
            rng=rng,
        )

        current_scan = lidar.scan(
            current,
            scenario["parked_vehicles"],
            rng=rng,
        )

        target = previous_scan["points_local"]
        source = current_scan["points_local"]

        # Ground-truth relative pose.
        gt_yaw, gt_translation = (
            relative_current_to_previous(
                previous,
                current,
            )
        )

        # Simulated noisy odometry initial guess.
        odom_yaw = normalize_angle(
            gt_yaw + rng.normal(
                0.0,
                math.radians(0.8),
            )
        )

        odom_translation = (
            gt_translation
            + rng.normal(
                0.0,
                0.04,
                size=2,
            )
        )

        initial_translation_error = float(
            np.linalg.norm(
                odom_translation - gt_translation
            )
        )

        initial_yaw_error = abs(
            normalize_angle(
                odom_yaw - gt_yaw
            )
        )

        if len(source) < 30 or len(target) < 30:
            continue

        # ICP refinement.
        result = icp_2d(
            source_points=source,
            target_points=target,
            initial_yaw=odom_yaw,
            initial_translation=odom_translation,
            max_iterations=35,
            max_correspondence_distance=0.8,
            min_correspondences=30,
            trim_fraction=0.85,
            max_points=220,
        )

        estimated_translation_error = float(
            np.linalg.norm(
                result["translation"] - gt_translation
            )
        )

        estimated_yaw_error = abs(
            normalize_angle(
                result["yaw_rad"] - gt_yaw
            )
        )

        rmse = result["rmse"]

        passed = (
            result["correspondences"] >= 30
            and result["fitness"] >= 0.20
            and estimated_translation_error <= 0.15
            and math.degrees(estimated_yaw_error) <= 2.0
            and math.isfinite(rmse)
            and rmse <= 0.18
        )

        record = {
            "pair_index": index,
            "source_points": len(source),
            "target_points": len(target),

            "initial_translation_error_m":
                finite_value(initial_translation_error),

            "initial_yaw_error_deg":
                finite_value(math.degrees(initial_yaw_error)),

            "icp_translation_error_m":
                finite_value(estimated_translation_error),

            "icp_yaw_error_deg":
                finite_value(math.degrees(estimated_yaw_error)),

            "icp_rmse_m":
                finite_value(rmse),

            "fitness":
                finite_value(result["fitness"]),

            "correspondences":
                result["correspondences"],

            "iterations":
                result["iterations"],

            "converged":
                result["converged"],

            "passed": bool(passed),
        }

        records.append(record)

        # Keep one pair for visualization.
        if visual_data is None:
            visual_data = {
                "target": target.copy(),
                "source_initial": transform_points(
                    source,
                    odom_yaw,
                    odom_translation,
                ),
                "source_icp": transform_points(
                    source,
                    result["yaw_rad"],
                    result["translation"],
                ),
            }

    if not records:
        raise RuntimeError("No valid ICP scan pairs")

    successful = [
        item for item in records
        if item["passed"]
    ]

    def mean_metric(key):
        values = [
            item[key]
            for item in records
            if item[key] is not None
        ]

        return round(
            sum(values) / len(values),
            6,
        ) if values else None

    pass_rate = (
        len(successful) / len(records) * 100.0
    )

    summary = {
        "status": "PASS" if (
            pass_rate >= 85.0
            and (mean_metric("icp_translation_error_m") or 0) <= 0.12
            and (mean_metric("icp_yaw_error_deg") or 0) <= 1.5
        ) else "REVIEW",

        "scan_pairs": len(records),

        "passed_pairs": len(successful),

        "pass_rate_percent": round(pass_rate, 2),

        "mean_initial_translation_error_m":
            mean_metric("initial_translation_error_m"),

        "mean_icp_translation_error_m":
            mean_metric("icp_translation_error_m"),

        "mean_initial_yaw_error_deg":
            mean_metric("initial_yaw_error_deg"),

        "mean_icp_yaw_error_deg":
            mean_metric("icp_yaw_error_deg"),

        "mean_icp_rmse_m":
            mean_metric("icp_rmse_m"),

        "mean_fitness":
            mean_metric("fitness"),

        "sensor_beams": 720,

        "reference_path_points": len(poses),
    }

    metrics = {
        "summary": summary,
        "pairs": records,
    }

    metrics_path = (
        OUTPUT_DIR / "p9_icp_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Visualization.
    fig, axes = plt.subplots(
        1, 3,
        figsize=(17, 5),
    )

    if visual_data is not None:

        axes[0].scatter(
            visual_data["target"][:, 0],
            visual_data["target"][:, 1],
            s=7,
            label="Previous scan",
        )

        axes[0].scatter(
            visual_data["source_initial"][:, 0],
            visual_data["source_initial"][:, 1],
            s=7,
            label="Odometry initial",
        )

        axes[0].scatter(
            visual_data["source_icp"][:, 0],
            visual_data["source_icp"][:, 1],
            s=7,
            label="After ICP",
        )

        axes[0].set_title(
            "Point Cloud Alignment"
        )

        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Y (m)")
        axes[0].axis("equal")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

    pair_indices = [
        item["pair_index"] for item in records
    ]

    initial_translation_errors = [
        item["initial_translation_error_m"]
        for item in records
    ]

    icp_translation_errors = [
        item["icp_translation_error_m"]
        for item in records
    ]

    axes[1].plot(
        pair_indices,
        initial_translation_errors,
        label="Odometry",
    )

    axes[1].plot(
        pair_indices,
        icp_translation_errors,
        label="ICP",
    )

    axes[1].set_title(
        "Translation Error"
    )

    axes[1].set_xlabel("Scan Pair")
    axes[1].set_ylabel("Error (m)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    initial_yaw_errors = [
        item["initial_yaw_error_deg"]
        for item in records
    ]

    icp_yaw_errors = [
        item["icp_yaw_error_deg"]
        for item in records
    ]

    axes[2].plot(
        pair_indices,
        initial_yaw_errors,
        label="Odometry",
    )

    axes[2].plot(
        pair_indices,
        icp_yaw_errors,
        label="ICP",
    )

    axes[2].set_title(
        "Heading Error"
    )

    axes[2].set_xlabel("Scan Pair")
    axes[2].set_ylabel("Error (deg)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.suptitle(
        "P9.2 ICP Scan Matching Evaluation"
    )

    fig.tight_layout()

    figure_path = (
        OUTPUT_DIR / "p9_icp_scan_matching.png"
    )

    fig.savefig(
        figure_path,
        dpi=180,
    )

    plt.show()
    plt.close(fig)

    print("\n========== P9.2 ICP RESULTS ==========")

    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ))

    print(f"\nMetrics: {metrics_path}")
    print(f"Visualization: {figure_path}")


if __name__ == "__main__":
    main()
