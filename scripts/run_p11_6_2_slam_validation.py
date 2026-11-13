
"""P11.6.2: Physical LiDAR -> existing OnlineSLAM2D.

Replays scans captured by P11.6.1.
Uses wheel speed and steering to estimate odometry.
Ground truth is only used for initialization and evaluation.

This stage validates the SLAM interface.
It does not yet perform live driving or path planning.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parking_planner.sensing.lidar_2d import LidarConfig
from parking_planner.slam.online_slam import OnlineSLAM2D, wrap_angle

SRC = ROOT / "results/p11_6_1_physical_lidar_scans.npz"
OUT = ROOT / "results"


def relative_pose(a, b):
    d = b[:2] - a[:2]
    c, s = math.cos(a[2]), math.sin(a[2])

    return np.array([
        c * d[0] + s * d[1],
        -s * d[0] + c * d[1],
        wrap_angle(b[2] - a[2]),
    ])


def angle_errors(a, b):
    delta = a[:, 2] - b[:, 2]
    return np.degrees(
        np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
    )


def main():

    if not SRC.is_file():
        raise FileNotFoundError(
            f"Run P11.6.1 first: {SRC}"
        )

    # 1. Load physical LiDAR observations.
    with np.load(SRC, allow_pickle=False) as d:
        angles = d["angles_rad"].copy()
        ranges = d["ranges_m"].copy()
        hits = d["valid"].copy()
        times = d["scan_timestamps_s"].copy()
        truth = d["true_rear_axle_poses"].copy()
        raw = d["physical_trace"].copy()
        columns = [
            str(c) for c in d["trace_columns"]
        ]

    n, beams = ranges.shape

    assert hits.shape == (n, beams)
    assert angles.shape == (beams,)
    assert times.shape == (n,)
    assert truth.shape == (n, 3)
    assert np.all(np.diff(times) > 0)

    required = ("time_s", "speed", "steering")

    assert all(k in columns for k in required)
    assert all(
        np.isfinite(a).all()
        for a in (angles, ranges, times, truth, raw)
    )

    # 2. Reuse the existing SLAM implementation.
    config = LidarConfig(
        num_beams=beams,
        fov_deg=270,
        min_range_m=0.12,
        max_range_m=12.0,
        range_std_m=0.015,
    )

    # Initial pose is the known global starting anchor.
    # Future true poses are not supplied to SLAM.
    slam = OnlineSLAM2D(
        initial_pose=truth[0],
        lidar_config=config,
    )

    odom = truth[0].copy()
    prev_scan_odom = None

    rng = np.random.default_rng(20261112)

    t_idx, v_idx, steer_idx = (
        columns.index(k) for k in required
    )

    trace_i = 0
    last_time = float(times[0])

    odom_hist = []
    slam_hist = []
    statuses = []
    reasons = []

    # 3. Process scans sequentially.
    for i, stamp in enumerate(times):

        # Integrate recorded wheel speed and steering.
        # Add simulated measurement errors.
        while (
            trace_i < len(raw)
            and raw[trace_i, t_idx] <= stamp + 1e-7
        ):
            record = raw[trace_i]

            dt = max(
                0.0,
                float(record[t_idx] - last_time),
            )

            if dt:
                speed = (
                    float(record[v_idx]) * 1.02
                    + rng.normal(0, 0.008)
                )

                steering = (
                    float(record[steer_idx])
                    + math.radians(0.3)
                    + rng.normal(
                        0,
                        math.radians(0.2),
                    )
                )

                yaw_delta = (
                    speed / 2.7
                    * math.tan(steering)
                    * dt
                )

                heading_mid = (
                    odom[2] + 0.5 * yaw_delta
                )

                odom[0] += (
                    speed * dt
                    * math.cos(heading_mid)
                )

                odom[1] += (
                    speed * dt
                    * math.sin(heading_mid)
                )

                odom[2] = wrap_angle(
                    odom[2] + yaw_delta
                )

            last_time = max(
                last_time,
                float(record[t_idx]),
            )

            trace_i += 1

        # Relative wheel-odometry increment.
        delta = (
            None
            if prev_scan_odom is None
            else relative_pose(
                prev_scan_odom,
                odom,
            )
        )

        scan = {
            "angles_rad": angles,
            "ranges_m": ranges[i],
            "valid": hits[i],
        }

        # Existing ICP + Occupancy Grid pipeline.
        info = slam.step(scan, delta)

        prev_scan_odom = odom.copy()

        odom_hist.append(odom.copy())
        slam_hist.append(slam.pose.copy())

        statuses.append(info["status"])
        reasons.append(
            info.get("reason")
            or "accepted_or_bootstrap"
        )

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i+1}/{n} scans",
                flush=True,
            )

    odom_hist = np.array(odom_hist)
    slam_hist = np.array(slam_hist)

    # 4. Evaluate the results.
    ed = np.linalg.norm(
        odom_hist[:, :2] - truth[:, :2],
        axis=1,
    )

    es = np.linalg.norm(
        slam_hist[:, :2] - truth[:, :2],
        axis=1,
    )

    yd = angle_errors(odom_hist, truth)
    ys = angle_errors(slam_hist, truth)

    states = slam.mapper.states

    summary = {
        "stage": (
            "P11.6.2 physical-LiDAR-to-SLAM "
            "offline integration"
        ),
        "status": "DATA_FLOW_OK_LOCALIZATION_REVIEW",
        "scan_count": int(n),
        "hit_count": int(hits.sum()),
        "status_counts": dict(Counter(statuses)),
        "rejection_reasons": dict(Counter(reasons)),
        "scans_with_less_than_30_hits": int(
            (hits.sum(axis=1) < 30).sum()
        ),
        "occupied_cells": int(
            (states == 1).sum()
        ),
        "free_cells": int(
            (states == 0).sum()
        ),
        "unknown_cells": int(
            (states == -1).sum()
        ),
        "odometry_translation_rmse_m": float(
            np.sqrt(np.mean(ed ** 2))
        ),
        "slam_translation_rmse_m": float(
            np.sqrt(np.mean(es ** 2))
        ),
        "odometry_heading_rmse_deg": float(
            np.sqrt(np.mean(yd ** 2))
        ),
        "slam_heading_rmse_deg": float(
            np.sqrt(np.mean(ys ** 2))
        ),
        "final_slam_position_error_m": float(
            es[-1]
        ),
        "final_slam_heading_error_deg": float(
            ys[-1]
        ),
        "note": (
            "Only the initial reference pose anchors SLAM; "
            "later truth is evaluation-only."
        ),
    }

    # 5. Save the new SLAM map and pose history.
    OUT.mkdir(exist_ok=True, parents=True)

    np.savez_compressed(
        OUT / "p11_6_2_slam_map.npz",
        **slam.mapper.arrays(),
        estimated_poses=slam_hist,
        odometry_poses=odom_hist,
        reference_poses=truth,
        scan_timestamps_s=times,
    )

    (
        OUT / "p11_6_2_slam_metrics.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )

    # 6. Plot occupancy grid and localization errors.
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import (
        ListedColormap,
        BoundaryNorm,
    )

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        layout="constrained",
    )

    colors = ListedColormap([
        "#999999",
        "#ffffff",
        "#101010",
    ])

    norm = BoundaryNorm(
        [-1.5, -0.5, 0.5, 1.5],
        3,
    )

    # Occupancy grid map.
    axs[0].imshow(
        states,
        origin="lower",
        cmap=colors,
        norm=norm,
        extent=[0, 22, 0, 22],
        interpolation="nearest",
    )

    axs[0].plot(
        truth[:, 0],
        truth[:, 1],
        "b--",
        label="Physics truth (evaluation)",
    )

    axs[0].plot(
        slam_hist[:, 0],
        slam_hist[:, 1],
        color="darkorange",
        label="SLAM",
    )

    axs[0].set(
        title="Occupancy grid map",
        xlabel="X (m)",
        ylabel="Y (m)",
        xlim=(0, 22),
        ylim=(0, 22),
    )

    axs[0].set_aspect("equal")
    axs[0].legend(fontsize=7)

    # Trajectory comparison.
    axs[1].plot(
        truth[:, 0],
        truth[:, 1],
        "b--",
        label="Physics truth",
    )

    axs[1].plot(
        odom_hist[:, 0],
        odom_hist[:, 1],
        color="gray",
        label="Wheel odometry",
    )

    axs[1].plot(
        slam_hist[:, 0],
        slam_hist[:, 1],
        color="darkorange",
        label="SLAM",
    )

    axs[1].set(
        title="Trajectories",
        xlabel="X (m)",
        ylabel="Y (m)",
    )

    axs[1].set_aspect("equal")
    axs[1].legend(fontsize=7)

    # Localization error.
    axs[2].plot(
        times,
        ed,
        color="gray",
        label="Odometry",
    )

    axs[2].plot(
        times,
        es,
        color="darkorange",
        label="SLAM",
    )

    axs[2].set(
        title="Localization error",
        xlabel="Simulation time (s)",
        ylabel="Position error (m)",
    )

    axs[2].legend()

    for ax in axs:
        ax.grid(alpha=0.15)

    fig.savefig(
        OUT / "p11_6_2_slam_evaluation.png",
        dpi=150,
    )

    plt.close(fig)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
