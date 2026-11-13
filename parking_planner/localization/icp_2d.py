
"""2D point-to-point ICP scan matching."""

import math
import numpy as np


def rotation_matrix(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)

    return np.array([
        [c, -s],
        [s, c],
    ])


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def estimate_rigid_transform(source, target):
    """Estimate R, t satisfying target ~= R @ source + t."""

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    H = source_centered.T @ target_centered

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = target_mean - R @ source_mean

    return R, t


def downsample(points, max_points):
    if len(points) <= max_points:
        return points

    indices = np.linspace(
        0,
        len(points) - 1,
        max_points,
        dtype=int,
    )

    return points[indices]


def icp_2d(
    source_points,
    target_points,
    initial_yaw=0.0,
    initial_translation=(0.0, 0.0),
    max_iterations=35,
    max_correspondence_distance=0.8,
    min_correspondences=30,
    trim_fraction=0.85,
    tolerance=1e-4,
    max_points=220,
):
    """
    Align current scan (source) to previous scan (target).

    Transform convention:
        target_point ~= R @ source_point + t
    """

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)

    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source_points must have shape (N, 2)")

    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target_points must have shape (N, 2)")

    if not np.all(np.isfinite(source)):
        raise ValueError("source contains invalid values")

    if not np.all(np.isfinite(target)):
        raise ValueError("target contains invalid values")

    if len(source) < min_correspondences:
        raise ValueError("Not enough source points")

    if len(target) < min_correspondences:
        raise ValueError("Not enough target points")

    if not 0 < trim_fraction <= 1:
        raise ValueError("Invalid trim_fraction")

    source = downsample(source, max_points)
    target = downsample(target, max_points)

    R = rotation_matrix(initial_yaw)

    t = np.asarray(
        initial_translation,
        dtype=float,
    ).reshape(2)

    max_dist_sq = max_correspondence_distance ** 2

    converged = False
    iterations = 0

    for iteration in range(max_iterations):

        # Transform current scan into previous scan frame.
        transformed = source @ R.T + t

        # Brute-force nearest-neighbor correspondence.
        differences = (
            transformed[:, None, :]
            - target[None, :, :]
        )

        distances_sq = np.sum(
            differences ** 2,
            axis=2,
        )

        nearest_indices = np.argmin(
            distances_sq,
            axis=1,
        )

        nearest_distances_sq = distances_sq[
            np.arange(len(source)),
            nearest_indices,
        ]

        valid_indices = np.where(
            nearest_distances_sq <= max_dist_sq
        )[0]

        if len(valid_indices) < min_correspondences:
            break

        # Trim distant correspondences to reduce outlier influence.
        order = valid_indices[
            np.argsort(nearest_distances_sq[valid_indices])
        ]

        keep_count = max(
            min_correspondences,
            int(math.ceil(len(order) * trim_fraction)),
        )

        selected = order[:keep_count]

        matched_source = transformed[selected]

        matched_target = target[
            nearest_indices[selected]
        ]

        # Estimate incremental rigid transformation.
        R_delta, t_delta = estimate_rigid_transform(
            matched_source,
            matched_target,
        )

        # Compose transformations.
        R = R_delta @ R
        t = R_delta @ t + t_delta

        delta_yaw = math.atan2(
            R_delta[1, 0],
            R_delta[0, 0],
        )

        iterations = iteration + 1

        if (
            np.linalg.norm(t_delta) < tolerance
            and abs(delta_yaw) < tolerance
        ):
            converged = True
            break

    # Evaluate final alignment.
    transformed = source @ R.T + t

    differences = (
        transformed[:, None, :]
        - target[None, :, :]
    )

    distances_sq = np.sum(
        differences ** 2,
        axis=2,
    )

    nearest_distances_sq = np.min(
        distances_sq,
        axis=1,
    )

    valid = nearest_distances_sq <= max_dist_sq

    correspondences = int(np.sum(valid))

    if correspondences > 0:
        rmse = math.sqrt(
            float(np.mean(nearest_distances_sq[valid]))
        )
    else:
        rmse = float("inf")

    fitness = correspondences / max(1, len(source))

    yaw = math.atan2(
        R[1, 0],
        R[0, 0],
    )

    return {
        "rotation": R,
        "translation": t,
        "yaw_rad": normalize_angle(yaw),
        "rmse": rmse,
        "fitness": fitness,
        "correspondences": correspondences,
        "iterations": iterations,
        "converged": converged,
    }
