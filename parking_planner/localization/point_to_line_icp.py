"""Robust 2D point-to-line scan-to-map ICP; no ground truth is used here."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class PointToLineConfig:
    normal_radius_m: float = .25
    normal_min_neighbors: int = 5
    normal_max_eigen_ratio: float = .12
    max_correspondence_m: float = .35
    min_correspondences: int = 30
    min_match_ratio: float = .30
    huber_delta_m: float = .04
    # 5 cm cell quantization (sigma=0.05/sqrt(12)) plus 1.5 cm mapping
    # range noise gives about 2 cm; fixed for all scans, not truth calibrated.
    surface_sigma_m: float = .02
    prior_translation_std_m: float = .08
    prior_heading_std_rad: float = np.deg2rad(1.5)
    min_information_eigen_ratio: float = .005
    max_line_rmse_m: float = .08
    max_prior_mahalanobis: float = 3.
    max_iterations: int = 60
    translation_tolerance_m: float = 1e-4
    heading_tolerance_rad: float = 1e-4


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def transform(points, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + pose[:2]


def estimate_normals(points, config=None):
    """PCA on bounded local neighborhoods, rejecting corners and short support.

    Return fitted local-line centers as well as normals to reduce grid jitter.
    Normal signs are arbitrary; squared/Huber residuals are sign invariant.
    """
    cfg = config or PointToLineConfig()
    points = np.asarray(points, dtype=float)
    normals, centers = np.zeros_like(points), points.copy()
    valid = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        neighbors = points[np.sum((points-point)**2, axis=1) <= cfg.normal_radius_m**2]
        if len(neighbors) < cfg.normal_min_neighbors:
            continue
        center = neighbors.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh((neighbors-center).T @ (neighbors-center)/len(neighbors))
        if eigenvalues[1] < .0025 or eigenvalues[0]/eigenvalues[1] > cfg.normal_max_eigen_ratio:
            continue
        centers[i], normals[i], valid[i] = center, eigenvectors[:, 0], True
    return centers, normals, valid


def point_to_line_icp(scan, initial_pose, map_points, config=None, surface=None):
    """Optimize global (x,y,yaw) with Huber IRLS and an odometry Gaussian prior.

    Backtracking holds correspondences fixed during each Gauss-Newton step;
    rematching occurs on the next outer iteration.

    Data objective is a MEAN over correspondences: densely sampled beams are
    correlated, so they must not arbitrarily overwhelm the single pose prior.
    Geometry is checked on scaled, data-only J'WJ, before adding the prior.
    Any rejection returns the original pose; candidate_pose retains diagnostics.
    """
    cfg = config or PointToLineConfig()
    scan, points = np.asarray(scan, float), np.asarray(map_points, float)
    initial = np.asarray(initial_pose, float).copy()
    if scan.ndim != 2 or scan.shape[1] != 2 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Point arrays must have shape (N,2)')
    if initial.shape != (3,) or not all(np.all(np.isfinite(a)) for a in (scan, points, initial)):
        raise ValueError('Nonfinite points or invalid pose')
    centers, normals, valid = surface if surface is not None else estimate_normals(points, cfg)
    target, centers, normals = points[valid], centers[valid], normals[valid]
    pose = initial.copy()
    prior_std = np.array([cfg.prior_translation_std_m]*2+[cfg.prior_heading_std_rad])
    prior_information = np.diag(1/prior_std**2)
    converged, degenerate, reason, iterations = False, False, None, 0
    eigenvalues, ratio = np.zeros(3), 0.

    def linearize(p):
        rotated = transform(scan, p)-p[:2]
        world = rotated+p[:2]
        if not len(target) or not len(scan):
            return np.empty(0), np.empty((0,3)), np.empty(0), None
        d2 = np.sum((world[:,None]-target[None])**2, axis=2)
        indices = np.argmin(d2, axis=1)
        keep = d2[np.arange(len(scan)), indices] <= cfg.max_correspondence_m**2
        n = normals[indices[keep]]
        residual = np.sum(n*(world[keep]-centers[indices[keep]]), axis=1)
        angular = np.column_stack((-rotated[keep,1], rotated[keep,0]))
        jacobian = np.column_stack((n, np.sum(n*angular, axis=1)))
        return residual, jacobian, np.sqrt(d2[np.arange(len(scan)),indices][keep]), (keep, indices[keep])

    def cost(p, matches):
        keep, indices = matches
        residual = np.sum(normals[indices]*(transform(scan[keep], p)-centers[indices]), axis=1)
        if len(residual) < cfg.min_correspondences:
            return np.inf
        a = np.abs(residual)
        huber = np.where(a <= cfg.huber_delta_m, .5*a*a, cfg.huber_delta_m*(a-.5*cfg.huber_delta_m))
        delta = p-initial
        delta[2] = wrap(delta[2])
        return np.mean(huber)/cfg.surface_sigma_m**2 + .5*np.sum((delta/prior_std)**2)

    def geometry(residual, jacobian):
        weights = np.minimum(1., cfg.huber_delta_m/np.maximum(np.abs(residual),1e-12))
        # Scale yaw by scan lever arm to compare translation and rotation units.
        lever = max(float(np.sqrt(np.mean(np.sum(scan**2,axis=1)))), .1)
        scaled = jacobian @ np.diag([1.,1.,1./lever])
        eig = np.linalg.eigvalsh((scaled.T*weights) @ scaled/max(len(residual),1))
        return weights, eig, float(eig[0]/max(eig[-1],1e-12))

    for iterations in range(1,cfg.max_iterations+1):
        residual, jacobian, _, matches = linearize(pose)
        if len(residual) < cfg.min_correspondences or len(residual)/max(len(scan),1) < cfg.min_match_ratio:
            reason = 'insufficient_correspondences'
            break
        weights, eigenvalues, ratio = geometry(residual,jacobian)
        if ratio < cfg.min_information_eigen_ratio:
            degenerate, reason = True, 'geometric_degeneracy'
            break
        w = weights/(len(residual)*cfg.surface_sigma_m**2)
        delta = pose-initial
        delta[2] = wrap(delta[2])
        hessian = (jacobian.T*w) @ jacobian+prior_information
        gradient = jacobian.T @ (w*residual)+prior_information @ delta
        step = -np.linalg.solve(hessian, gradient)
        objective = cost(pose, matches)
        accepted = False
        for scale in (1., .5, .25, .125, .0625, .03125):
            candidate = pose+scale*step
            candidate[2] = wrap(candidate[2])
            if cost(candidate, matches) <= objective+1e-12:
                pose, accepted = candidate, True
                break
        if not accepted:
            reason = 'line_search_stalled'
            break
        if np.linalg.norm(scale*step[:2]) < cfg.translation_tolerance_m and abs(scale*step[2]) < cfg.heading_tolerance_rad:
            converged = True
            break
    residual, jacobian, distances, _ = linearize(pose)
    if len(residual) >= cfg.min_correspondences:
        _, eigenvalues, ratio = geometry(residual,jacobian)
        if ratio < cfg.min_information_eigen_ratio:
            degenerate, reason = True, 'geometric_degeneracy'
    line_rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
    correction = pose-initial
    correction[2] = wrap(correction[2])
    mahalanobis = float(np.linalg.norm(correction/prior_std))
    success = bool(converged and not degenerate and len(residual) >= cfg.min_correspondences and
                   len(residual)/max(len(scan),1) >= cfg.min_match_ratio and line_rmse is not None and
                   line_rmse <= cfg.max_line_rmse_m and mahalanobis <= cfg.max_prior_mahalanobis)
    if not success and reason is None:
        reason = 'quality_gate_or_iteration_limit'
    return dict(pose=pose.copy() if success else initial.copy(), candidate_pose=pose,
                success=success, converged=converged, degenerate=degenerate, rejection_reason=reason,
                iterations=iterations, correspondences=len(residual), fitness=len(residual)/max(len(scan),1),
                line_rmse_m=line_rmse, rmse_m=float(np.sqrt(np.mean(distances**2))) if len(distances) else None,
                information_eigenvalues=eigenvalues.tolist(), information_eigen_ratio=ratio,
                prior_mahalanobis=mahalanobis, config=asdict(cfg))
