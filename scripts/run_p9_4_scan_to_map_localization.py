
"""P9.4: Scan-to-Map ICP localization using the P9.3 ICP occupancy map."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAP_FILE = RESULTS / "p9_occupancy_maps.npz"

OUT_JSON = RESULTS / "p9_4_scan_to_map_metrics.json"
OUT_CSV = RESULTS / "p9_4_scan_to_map_results.csv"
OUT_PNG = RESULTS / "p9_4_scan_to_map_localization.png"

RAY_COUNT = 720
MAX_RANGE = 8.0

MAX_CORRESPONDENCE = 0.35
MIN_CORRESPONDENCES = 30
MAX_ITERATIONS = 60
TRIM_FRACTION = 0.80
TRANS_EPS = 1e-4
ROT_EPS = 1e-4

SEED = 2026
POSITION_NOISE_STD = 0.08
HEADING_NOISE_STD_DEG = 1.5

TEST_POSES = np.array([
    [6.85, 9.75, 0.05],
    [7.45, 9.45, 0.15],
    [8.55, 8.90, 0.45],
    [9.75, 8.45, 1.00],
    [10.60, 9.00, 1.45],
    [10.55, 10.85, 1.57],
    [10.55, 13.05, 1.57],
    [10.55, 14.85, 1.57],
], dtype=float)


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_map():
    if not MAP_FILE.exists():
        raise FileNotFoundError(
            f"Missing {MAP_FILE}. Put p9_occupancy_maps.npz in results."
        )

    data = np.load(MAP_FILE, allow_pickle=False)
    required = {"icp_states", "icp_config", "parked_vehicles"}
    missing = required - set(data.files)

    if missing:
        raise KeyError(f"Missing fields: {sorted(missing)}")

    cfg = json.loads(str(data["icp_config"].item()))
    resolution = float(cfg["resolution"])

    rows, cols = np.where(data["icp_states"] == 1)

    map_points = np.column_stack((
        float(cfg["origin_x"]) + (cols + 0.5) * resolution,
        float(cfg["origin_y"]) + (rows + 0.5) * resolution,
    ))

    obstacles = np.asarray(data["parked_vehicles"], dtype=float)

    if len(map_points) < MIN_CORRESPONDENCES:
        raise ValueError("Not enough occupied map cells.")

    if obstacles.ndim != 2 or obstacles.shape[1] != 4:
        raise ValueError("Invalid parked_vehicles format.")

    return map_points, obstacles


def ray_box_hits(origin, direction, box):
    xmin, ymin, xmax, ymax = box
    x, y = origin
    dx, dy = direction

    hits = []
    eps = 1e-12

    if abs(dx) > eps:
        for xe in (xmin, xmax):
            t = (xe - x) / dx
            yh = y + t * dy

            if t > 0 and ymin - 1e-9 <= yh <= ymax + 1e-9:
                hits.append(t)

    if abs(dy) > eps:
        for ye in (ymin, ymax):
            t = (ye - y) / dy
            xh = x + t * dx

            if t > 0 and xmin - 1e-9 <= xh <= xmax + 1e-9:
                hits.append(t)

    return hits


def simulate_scan(pose, obstacles):
    """Generate a new LiDAR scan in the sensor coordinate frame."""

    x, y, yaw = pose
    origin = np.array([x, y])
    points = []

    for angle in np.linspace(
        -math.pi, math.pi, RAY_COUNT, endpoint=False
    ):
        world_angle = yaw + angle

        direction = np.array([
            math.cos(world_angle),
            math.sin(world_angle)
        ])

        distance = MAX_RANGE

        for box in obstacles:
            hits = ray_box_hits(origin, direction, box)

            if hits:
                distance = min(distance, min(hits))

        if distance < MAX_RANGE:
            points.append([
                distance * math.cos(angle),
                distance * math.sin(angle)
            ])

    points = np.asarray(points, dtype=float)

    if len(points) < MIN_CORRESPONDENCES:
        raise RuntimeError(
            f"Only {len(points)} LiDAR returns at pose {pose}."
        )

    return points


def transform(points, pose):
    """Transform local scan points into global coordinates."""

    c = math.cos(pose[2])
    s = math.sin(pose[2])

    rotation = np.array([
        [c, -s],
        [s, c]
    ])

    return points @ rotation.T + pose[:2]


def nearest_points(source, target_map):
    """Find nearest map point for every transformed scan point."""

    delta = source[:, None, :] - target_map[None, :, :]
    d2 = np.sum(delta * delta, axis=2)

    indices = np.argmin(d2, axis=1)

    distances = np.sqrt(
        d2[np.arange(len(source)), indices]
    )

    return distances, target_map[indices]


def rigid_transform(source, target):
    """Kabsch/SVD 2D rigid transformation."""

    cs = source.mean(axis=0)
    ct = target.mean(axis=0)

    u, _, vt = np.linalg.svd(
        (source - cs).T @ (target - ct)
    )

    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T

    translation = ct - rotation @ cs

    return rotation, translation


def scan_to_map_icp(local_scan, initial_pose, target_map):
    """Align a LiDAR scan with the fixed global occupancy map."""

    pose = initial_pose.copy()
    converged = False

    for iteration in range(1, MAX_ITERATIONS + 1):

        world_scan = transform(local_scan, pose)

        distances, matched = nearest_points(
            world_scan,
            target_map
        )

        inliers = distances <= MAX_CORRESPONDENCE

        if inliers.sum() < MIN_CORRESPONDENCES:
            break

        src = world_scan[inliers]
        dst = matched[inliers]

        inlier_distances = distances[inliers]

        # Remove the worst 20% correspondences.
        trim_limit = np.quantile(
            inlier_distances,
            TRIM_FRACTION
        )

        keep = inlier_distances <= trim_limit

        src = src[keep]
        dst = dst[keep]

        if len(src) < MIN_CORRESPONDENCES:
            break

        rotation, translation = rigid_transform(src, dst)

        delta_yaw = math.atan2(
            rotation[1, 0],
            rotation[0, 0]
        )

        pose[:2] = rotation @ pose[:2] + translation

        pose[2] = wrap_angle(
            pose[2] + delta_yaw
        )

        if (
            np.linalg.norm(translation) < TRANS_EPS
            and abs(delta_yaw) < ROT_EPS
        ):
            converged = True
            break

    final_scan = transform(local_scan, pose)

    final_distances, _ = nearest_points(
        final_scan,
        target_map
    )

    final_inliers = final_distances <= MAX_CORRESPONDENCE
    count = int(final_inliers.sum())

    success = count >= MIN_CORRESPONDENCES

    final_rmse = (
        float(np.sqrt(np.mean(final_distances[final_inliers] ** 2)))
        if success else None
    )

    return {
        "pose": pose,
        "success": success,
        "converged": converged,
        "iterations": iteration,
        "correspondences": count,
        "fitness": float(count / len(local_scan)),
        "rmse_m": final_rmse,
    }


def pose_errors(estimate, truth):
    translation = float(
        np.linalg.norm(estimate[:2] - truth[:2])
    )

    heading = abs(
        wrap_angle(float(estimate[2] - truth[2]))
    )

    return translation, math.degrees(heading)


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def main():

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("P9.4 Scan-to-Map ICP Localization")
    print("=" * 60)

    map_points, obstacles = load_map()

    rng = np.random.default_rng(SEED)

    position_noise = rng.normal(
        0,
        POSITION_NOISE_STD,
        (len(TEST_POSES), 2)
    )

    heading_noise = np.deg2rad(
        rng.normal(
            0,
            HEADING_NOISE_STD_DEG,
            len(TEST_POSES)
        )
    )

    initial_poses = TEST_POSES.copy()

    initial_poses[:, :2] += position_noise

    initial_poses[:, 2] = [
        wrap_angle(yaw + noise)
        for yaw, noise in zip(
            TEST_POSES[:, 2],
            heading_noise
        )
    ]

    scans = []
    estimates = []
    icp_results = []
    rows = []

    for i, (truth, initial) in enumerate(
        zip(TEST_POSES, initial_poses),
        start=1
    ):

        scan = simulate_scan(truth, obstacles)

        result = scan_to_map_icp(
            scan,
            initial,
            map_points
        )

        estimate = result["pose"]

        before_t, before_h = pose_errors(initial, truth)
        after_t, after_h = pose_errors(estimate, truth)

        scans.append(scan)
        estimates.append(estimate)
        icp_results.append(result)

        rows.append({
            "scan_id": i,
            "scan_points": len(scan),
            "success": result["success"],
            "converged": result["converged"],
            "iterations": result["iterations"],
            "correspondences": result["correspondences"],
            "fitness": result["fitness"],
            "icp_rmse_m": result["rmse_m"],
            "ground_truth_x_m": truth[0],
            "ground_truth_y_m": truth[1],
            "ground_truth_yaw_deg": math.degrees(truth[2]),
            "initial_x_m": initial[0],
            "initial_y_m": initial[1],
            "initial_yaw_deg": math.degrees(initial[2]),
            "estimated_x_m": estimate[0],
            "estimated_y_m": estimate[1],
            "estimated_yaw_deg": math.degrees(estimate[2]),
            "initial_translation_error_m": before_t,
            "final_translation_error_m": after_t,
            "initial_heading_error_deg": before_h,
            "final_heading_error_deg": after_h,
        })

        print(
            f"Scan {i}: "
            f"{'SUCCESS' if result['success'] else 'FAILED'} "
            f"| corr={result['correspondences']} "
            f"| iter={result['iterations']} "
            f"| position {before_t:.3f}->{after_t:.3f} m "
            f"| heading {before_h:.3f}->{after_h:.3f} deg"
        )

    # Calculate overall evaluation metrics.

    t_before = [r["initial_translation_error_m"] for r in rows]
    t_after = [r["final_translation_error_m"] for r in rows]

    h_before = [r["initial_heading_error_deg"] for r in rows]
    h_after = [r["final_heading_error_deg"] for r in rows]

    success_count = sum(r["success"] for r in icp_results)

    all_converged = all(
        r["converged"] for r in icp_results
    )

    t_rmse_before = rmse(t_before)
    t_rmse_after = rmse(t_after)

    h_rmse_before = rmse(h_before)
    h_rmse_after = rmse(h_after)

    summary = {
        "experiment": "P9.4 Scan-to-Map ICP Localization",

        "map_source": "results/p9_occupancy_maps.npz: icp_states",

        "test_scan_source": (
            "New ray-cast scans from a separate test trajectory; "
            "P9.3 mapping scans are not reused."
        ),

        "random_seed": SEED,

        "initialization_perturbation": {
            "translation_std_m": POSITION_NOISE_STD,
            "heading_std_deg": HEADING_NOISE_STD_DEG,
            "note": "Deliberate stress-test perturbation."
        },

        "lidar": {
            "ray_count": RAY_COUNT,
            "max_range_m": MAX_RANGE
        },

        "icp": {
            "max_correspondence_distance_m": MAX_CORRESPONDENCE,
            "minimum_correspondences": MIN_CORRESPONDENCES,
            "max_iterations": MAX_ITERATIONS,
            "trim_fraction": TRIM_FRACTION
        },

        "scan_count": len(rows),
        "successful_scans": success_count,
        "success_rate": success_count / len(rows),
        "all_icp_converged": all_converged,

        "translation_rmse_before_m": t_rmse_before,
        "translation_rmse_after_m": t_rmse_after,

        "translation_rmse_improvement_percent": (
            (t_rmse_before - t_rmse_after)
            / t_rmse_before * 100
        ),

        "heading_rmse_before_deg": h_rmse_before,
        "heading_rmse_after_deg": h_rmse_after,

        "heading_rmse_improvement_percent": (
            (h_rmse_before - h_rmse_after)
            / h_rmse_before * 100
        ),

        "mean_icp_alignment_rmse_m": float(
            np.mean([
                r["rmse_m"]
                for r in icp_results
                if r["rmse_m"] is not None
            ])
        ),

        "mean_icp_iterations": float(
            np.mean([r["iterations"] for r in icp_results])
        ),

        "acceptance": {
            "all_scans_matched": success_count == len(rows),
            "all_icp_converged": all_converged,
            "translation_rmse_improved": t_rmse_after < t_rmse_before,
            "heading_rmse_improved": h_rmse_after < h_rmse_before,
        },

        "per_scan_results": rows
    }

    OUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    # Visualization.

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]

    ax.scatter(
        map_points[:, 0],
        map_points[:, 1],
        s=8,
        c="black",
        alpha=0.5,
        label="P9.3 ICP map"
    )

    ax.plot(
        TEST_POSES[:, 0],
        TEST_POSES[:, 1],
        "k--",
        label="Test trajectory (GT)"
    )

    ax.scatter(
        TEST_POSES[:, 0],
        TEST_POSES[:, 1],
        marker="*",
        s=80,
        c="green",
        label="Ground truth"
    )

    ax.scatter(
        initial_poses[:, 0],
        initial_poses[:, 1],
        marker="x",
        s=55,
        label="Initial odometry"
    )

    est = np.asarray(estimates)

    ax.scatter(
        est[:, 0],
        est[:, 1],
        marker="+",
        s=65,
        label="After Scan-to-Map ICP"
    )

    ax.set(
        title="Localization Poses",
        xlabel="X (m)",
        ylabel="Y (m)",
        aspect="equal"
    )

    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ids = np.arange(1, len(rows) + 1)

    ax = axes[0, 1]

    ax.plot(ids, t_before, "o--", label="Initial odometry")
    ax.plot(ids, t_after, "s-", label="After Scan-to-Map ICP")

    ax.set(
        title="Translation Error",
        xlabel="Test scan",
        ylabel="Position error (m)",
        xticks=ids
    )

    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]

    ax.plot(ids, h_before, "o--", label="Initial odometry")
    ax.plot(ids, h_after, "s-", label="After Scan-to-Map ICP")

    ax.set(
        title="Heading Error",
        xlabel="Test scan",
        ylabel="Absolute heading error (deg)",
        xticks=ids
    )

    ax.grid(alpha=0.25)
    ax.legend()

    # Representative point-cloud alignment.

    k = 4

    initial_scan = transform(
        scans[k],
        initial_poses[k]
    )

    final_scan = transform(
        scans[k],
        est[k]
    )

    ax = axes[1, 1]

    ax.scatter(
        map_points[:, 0],
        map_points[:, 1],
        s=9,
        c="black",
        alpha=0.3,
        label="ICP map"
    )

    ax.scatter(
        initial_scan[:, 0],
        initial_scan[:, 1],
        s=9,
        label="Initial scan"
    )

    ax.scatter(
        final_scan[:, 0],
        final_scan[:, 1],
        s=9,
        label="After ICP"
    )

    ax.set(
        title=f"Point-cloud Alignment (Scan {k + 1})",
        xlabel="X (m)",
        ylabel="Y (m)",
        aspect="equal"
    )

    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    fig.suptitle(
        "P9.4 Scan-to-Map ICP Localization Evaluation",
        fontsize=15
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    print("\n========== P9.4 SUMMARY ==========")

    print(f"Successful scans: {success_count}/{len(rows)}")
    print(f"ICP converged: {all_converged}")

    print(
        f"Translation RMSE: "
        f"{t_rmse_before:.4f} -> {t_rmse_after:.4f} m"
    )

    print(
        f"Heading RMSE: "
        f"{h_rmse_before:.4f} -> {h_rmse_after:.4f} deg"
    )

    print("Metrics:", OUT_JSON)
    print("CSV:", OUT_CSV)
    print("Visualization:", OUT_PNG)

    if all(summary["acceptance"].values()):
        print("\nP9.4 SCAN-TO-MAP LOCALIZATION PASSED")
    else:
        print("\nP9.4 COMPLETED WITH ITEMS TO REVIEW")


if __name__ == "__main__":
    main()
