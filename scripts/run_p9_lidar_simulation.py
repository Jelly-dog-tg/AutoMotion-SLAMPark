
"""P9.1: 2D LiDAR simulation and visualization."""

import json
import math
import sys

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Polygon


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)

from parking_planner.vehicle.ackermann import VehicleState

from parking_planner.sensing.lidar_2d import (
    Lidar2D,
    LidarConfig,
)


OUTPUT_DIR = ROOT / "results"


def vehicle_corners(state, geometry):

    local = [
        (-geometry.rear_overhang, -geometry.width / 2),
        (geometry.front_extent, -geometry.width / 2),
        (geometry.front_extent, geometry.width / 2),
        (-geometry.rear_overhang, geometry.width / 2),
    ]

    c = math.cos(state.yaw)
    s = math.sin(state.yaw)

    return [
        (
            state.x + c * x - s * y,
            state.y + s * x + c * y,
        )
        for x, y in local
    ]


def build_demo_poses(scenario):

    return [
        ("start", scenario["start"]),

        (
            "approach",
            VehicleState(
                x=11.8,
                y=8.3,
                yaw=math.radians(70),
                v=0.0,
            ),
        ),

        ("inside_bay", scenario["goal"]),
    ]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenario = build_perpendicular_parking_scenario()

    lidar = Lidar2D(
        LidarConfig(
            num_beams=720,
            fov_deg=270.0,
            min_range_m=0.12,
            max_range_m=12.0,
            range_std_m=0.015,
        )
    )

    rng = np.random.default_rng(42)

    poses = build_demo_poses(scenario)

    scan_results = []

    for name, pose in poses:

        scan = lidar.scan(
            pose,
            scenario["parked_vehicles"],
            rng=rng,
        )

        if scan["ranges_m"].shape != (720,):
            raise RuntimeError("Invalid LiDAR scan shape")

        if not np.all(np.isfinite(scan["ranges_m"])):
            raise RuntimeError("Non-finite LiDAR measurements")

        if np.any(scan["ranges_m"] < lidar.config.min_range_m):
            raise RuntimeError("Range below minimum")

        if np.any(scan["ranges_m"] > lidar.config.max_range_m):
            raise RuntimeError("Range exceeds maximum")

        valid_count = int(np.sum(scan["valid"]))

        if valid_count < 10:
            raise RuntimeError(
                f"{name}: too few valid LiDAR returns"
            )

        metrics = {
            "name": name,
            "beam_count": lidar.config.num_beams,
            "valid_returns": valid_count,
            "valid_ratio": round(
                valid_count / lidar.config.num_beams,
                4,
            ),
            "min_distance_m": round(
                float(np.min(scan["ranges_m"][scan["valid"]])),
                4,
            ),
            "max_distance_m": round(
                float(np.max(scan["ranges_m"][scan["valid"]])),
                4,
            ),
        }

        scan_results.append(
            (name, pose, scan, metrics)
        )

        print(f"\n{name}:")
        print(json.dumps(metrics, indent=2))

    # Visualization.
    fig, axes = plt.subplots(
        2, 3,
        figsize=(17, 10),
    )

    for col, (name, pose, scan, metrics) in enumerate(scan_results):

        world_ax = axes[0, col]
        local_ax = axes[1, col]

        # Draw obstacles.
        for x0, y0, x1, y1 in scenario["parked_vehicles"]:

            world_ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    color="gray",
                    alpha=0.65,
                )
            )

        # Draw parking bay.
        sx0, sy0, sx1, sy1 = scenario["parking_slot"]

        world_ax.add_patch(
            Rectangle(
                (sx0, sy0),
                sx1 - sx0,
                sy1 - sy0,
                fill=False,
                edgecolor="green",
                linestyle="--",
                linewidth=2,
            )
        )

        # Draw vehicle.
        world_ax.add_patch(
            Polygon(
                vehicle_corners(pose, scenario["geometry"]),
                closed=True,
                fill=False,
                edgecolor="blue",
                linewidth=1.5,
            )
        )

        # Draw selected LiDAR beams.
        points_world = scan["points_world"]

        for i in range(0, len(scan["valid"]), 24):

            if not scan["valid"][i]:
                continue

            point_index = np.sum(scan["valid"][:i])

            px, py = points_world[point_index]

            world_ax.plot(
                [pose.x, px],
                [pose.y, py],
                color="gold",
                alpha=0.25,
                linewidth=0.6,
            )

        # World point cloud.
        if len(points_world):

            world_ax.scatter(
                points_world[:, 0],
                points_world[:, 1],
                s=5,
                color="darkorange",
            )

        world_ax.scatter(
            pose.x,
            pose.y,
            color="blue",
            s=35,
            label="LiDAR position",
        )

        world_ax.set_title(
            f"{name}: World Scan"
        )

        world_ax.set_xlabel("X (m)")
        world_ax.set_ylabel("Y (m)")
        world_ax.set_xlim(4.5, 16.5)
        world_ax.set_ylim(7.0, 18.5)
        world_ax.set_aspect("equal")
        world_ax.grid(alpha=0.25)

        # Sensor local point cloud.
        points_local = scan["points_local"]

        if len(points_local):

            local_ax.scatter(
                points_local[:, 0],
                points_local[:, 1],
                s=5,
                color="darkorange",
            )

        local_ax.scatter(
            0,
            0,
            color="blue",
            s=35,
        )

        local_ax.set_title(
            f"{name}: Local Point Cloud"
        )

        local_ax.set_xlabel("X local (m)")
        local_ax.set_ylabel("Y local (m)")
        local_ax.set_xlim(-12, 12)
        local_ax.set_ylim(-12, 12)
        local_ax.set_aspect("equal")
        local_ax.grid(alpha=0.25)

    fig.suptitle(
        "P9.1 2D LiDAR Ray Casting Simulation",
        fontsize=16,
    )

    fig.tight_layout()

    figure_path = OUTPUT_DIR / "p9_lidar_simulation.png"

    fig.savefig(
        figure_path,
        dpi=180,
    )

    plt.show()
    plt.close(fig)

    # Save scan data.
    arrays = {
        "pose_names": np.asarray(
            [item[0] for item in scan_results],
            dtype="U32",
        )
    }

    for i, (_, _, scan, _) in enumerate(scan_results):

        arrays[f"scan_{i}_angles_rad"] = scan["angles_rad"]
        arrays[f"scan_{i}_ranges_m"] = scan["ranges_m"]
        arrays[f"scan_{i}_valid"] = scan["valid"]
        arrays[f"scan_{i}_points_local"] = scan["points_local"]
        arrays[f"scan_{i}_points_world"] = scan["points_world"]

    data_path = OUTPUT_DIR / "p9_lidar_scan_data.npz"

    np.savez_compressed(
        data_path,
        **arrays,
    )

    summary = {
        "status": "PASS",
        "sensor": {
            "beams": lidar.config.num_beams,
            "fov_deg": lidar.config.fov_deg,
            "max_range_m": lidar.config.max_range_m,
            "range_noise_std_m": lidar.config.range_std_m,
        },
        "scans": [
            item[3] for item in scan_results
        ],
    }

    metrics_path = OUTPUT_DIR / "p9_lidar_metrics.json"

    metrics_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== P9.1 LIDAR RESULTS ==========")

    print("Status: PASS")

    for item in summary["scans"]:
        print(
            f'{item["name"]}: '
            f'{item["valid_returns"]}/'
            f'{item["beam_count"]} valid returns'
        )

    print(f"Metrics: {metrics_path}")
    print(f"Scan data: {data_path}")
    print(f"Visualization: {figure_path}")


if __name__ == "__main__":
    main()
