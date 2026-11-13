"""P9.3: cached scans, sequential SE(2) localization and occupancy mapping."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import HybridAStarPlanner, PathPoint
from parking_planner.vehicle.ackermann import VehicleState
from parking_planner.sensing.lidar_2d import Lidar2D, LidarConfig
from parking_planner.localization import icp_2d
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def matrix(pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([[c, -s, pose[0]], [s, c, pose[1]], [0., 0., 1.]])


def vector(transform):
    return np.array([transform[0, 2], transform[1, 2], math.atan2(transform[1, 0], transform[0, 0])])


def sample_path(path, spacing=0.3):
    """Interpolate at cumulative arc length, retaining the final pose."""
    xyz = np.array([[p.x, p.y, p.yaw] for p in path])
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
    distances = np.r_[np.arange(0., lengths[-1], spacing), lengths[-1]]
    poses = []
    for distance in distances:
        i = min(np.searchsorted(lengths, distance, side='right'), len(path)-1)
        j = max(0, i-1)
        ratio = (distance-lengths[j])/max(lengths[i]-lengths[j], 1e-12)
        x, y = xyz[j, :2]+ratio*(xyz[i, :2]-xyz[j, :2])
        yaw = float(wrap(xyz[j, 2]+ratio*wrap(xyz[i, 2]-xyz[j, 2])))
        poses.append(PathPoint(float(x), float(y), yaw, path[i].direction, path[i].steering))
    return poses


def boundary_samples(obstacles, step):
    points = []
    for x0, y0, x1, y1 in obstacles:
        xs = np.linspace(x0, x1, math.ceil((x1-x0)/step)+1)
        ys = np.linspace(y0, y1, math.ceil((y1-y0)/step)+1)
        points.extend([(x, y) for x in xs for y in (y0, y1)])
        points.extend([(x, y) for y in ys[1:-1] for x in (x0, x1)])
    return np.array(points)


def visible_boundary(points, poses, obstacles, cfg):
    """Analytical visibility of true surfaces; never use estimated map as truth."""
    visible = np.zeros(len(points), dtype=bool)
    for pose in poses:
        delta = points-np.array([pose.x, pose.y])
        distance = np.linalg.norm(delta, axis=1)
        directions = delta/np.maximum(distance[:, None], 1e-12)
        nearest = np.full(len(points), np.inf)
        for x0, y0, x1, y1 in obstacles:
            tx0, tx1 = Lidar2D._axis_interval(pose.x, directions[:, 0], x0, x1)
            ty0, ty1 = Lidar2D._axis_interval(pose.y, directions[:, 1], y0, y1)
            enter, leave = np.maximum(tx0, ty0), np.minimum(tx1, ty1)
            valid = (leave >= np.maximum(enter, 0)) & (enter >= cfg.min_range_m)
            nearest = np.minimum(nearest, np.where(valid, enter, np.inf))
        angles = wrap(np.arctan2(delta[:, 1], delta[:, 0])-pose.yaw)
        visible |= ((abs(angles) <= math.radians(cfg.fov_deg)/2) &
                    (distance <= cfg.max_range_m) & (distance >= cfg.min_range_m) &
                    (abs(nearest-distance) < 1e-6))
    return visible


def nearest_distances(a, b):
    if len(b) == 0:
        return np.full(len(a), np.inf)
    return np.concatenate([np.sqrt(np.min(np.sum((chunk[:, None]-b[None])**2, axis=2), axis=1))
                           for chunk in np.array_split(a, max(1, math.ceil(len(a)/256))) if len(chunk)]) if len(a) else np.empty(0)


def evaluate(mapper, all_boundary, visible):
    states = mapper.states
    rows, cols = np.where(states == 1)
    occupied = np.column_stack((mapper.origin_x+(cols+.5)*mapper.resolution,
                                mapper.origin_y+(rows+.5)*mapper.resolution))
    # Precision compares predictions to physical boundaries; recall only visible truth.
    precision = float(np.mean(nearest_distances(occupied, all_boundary) <= .15)) if len(occupied) else 0.
    recall = float(np.mean(nearest_distances(all_boundary[visible], occupied) <= .15)) if np.any(visible) else 0.
    return dict(occupied_cells=int(np.sum(states == 1)), free_cells=int(np.sum(states == 0)),
                unknown_cells=int(np.sum(states == -1)), boundary_precision=precision,
                boundary_recall=recall, boundary_f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def pose_errors(estimated, truth):
    translation = np.linalg.norm(estimated[:, :2]-truth[:, :2], axis=1)
    yaw = np.degrees(wrap(estimated[:, 2]-truth[:, 2]))
    return dict(translation_rmse_m=float(np.sqrt(np.mean(translation**2))),
                heading_rmse_deg=float(np.sqrt(np.mean(yaw**2))),
                final_translation_error_m=float(translation[-1]), final_heading_error_deg=float(abs(yaw[-1])))


def main():
    scenario = build_perpendicular_parking_scenario()
    planner = HybridAStarPlanner(grid=scenario['grid'], geometry=scenario['geometry'], config=scenario['planner_config'])
    path = planner.plan(scenario['start'], scenario['goal'])
    if not path:
        raise RuntimeError('Hybrid A* failed')
    poses = sample_path(path)
    cfg = LidarConfig()
    lidar = Lidar2D(cfg)
    scan_rng, odom_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(2026).spawn(2)]
    scans = [lidar.scan(p, scenario['parked_vehicles'], rng=scan_rng) for p in poses]
    truth = np.array([[p.x, p.y, p.yaw] for p in poses])
    odometry, icp_poses = [truth[0]], [truth[0]]
    odom_transform = icp_transform = matrix(truth[0])
    records = []
    for i in range(1, len(poses)):
        relative = np.linalg.inv(matrix(truth[i-1])) @ matrix(truth[i])
        noisy = vector(relative)
        noisy[:2] += odom_rng.normal(0., .04, 2)
        noisy[2] = wrap(noisy[2]+odom_rng.normal(0., math.radians(.8)))
        odom_relative = matrix(noisy)
        corrected = odom_relative
        record = dict(pair_index=i, success=False, failure_reason=None)
        try:
            result = icp_2d(scans[i]['points_local'], scans[i-1]['points_local'],
                            initial_yaw=noisy[2], initial_translation=noisy[:2])
            success = bool(result['converged'] and result['correspondences'] >= 30 and
                           result['fitness'] >= .20 and np.isfinite(result['rmse']) and result['rmse'] <= .18)
            record.update(success=success, converged=result['converged'], fitness=result['fitness'],
                          rmse_m=float(result['rmse']) if np.isfinite(result['rmse']) else None,
                          correspondences=result['correspondences'], iterations=result['iterations'])
            if success:
                corrected = np.eye(3)
                corrected[:2, :2], corrected[:2, 2] = result['rotation'], result['translation']
            else:
                record['failure_reason'] = 'ICP convergence/correspondence/fitness/residual gate failed'
        except (ValueError, np.linalg.LinAlgError) as error:
            record['failure_reason'] = str(error)
        # Current source -> previous target; compose without inverting ICP output.
        odom_transform = odom_transform @ odom_relative
        icp_transform = icp_transform @ corrected
        odometry.append(vector(odom_transform))
        icp_poses.append(vector(icp_transform))
        records.append(record)
    odometry, icp_poses = np.array(odometry), np.array(icp_poses)
    grid = scenario['grid']
    maps = {}
    for name, trajectory in [('ground_truth', truth), ('odometry', odometry), ('icp', icp_poses)]:
        mapper = OccupancyGridMapper(grid.width, grid.height, .05, grid.origin_x, grid.origin_y)
        for pose, scan in zip(trajectory, scans, strict=True):
            mapper.update(VehicleState(*pose), scan, cfg)
        maps[name] = mapper
    boundary = boundary_samples(scenario['parked_vehicles'], .025)
    visible = visible_boundary(boundary, poses, scenario['parked_vehicles'], cfg)
    metrics = dict(seed=2026, scan_count=len(scans), scan_calls=len(scans), scan_pairs=len(records),
                   resolution_m=.05, boundary_tolerance_m=.15,
                   evaluation='Precision: all physical rectangle boundaries; recall: analytically visible boundary samples; interiors excluded.',
                   visible_boundary_samples=int(visible.sum()), total_boundary_samples=len(boundary),
                   visible_boundary_fraction=float(visible.mean()),
                   odometry_noise=dict(translation_std_m=.04, heading_std_deg=.8),
                   icp_failure_gate=dict(require_converged=True, min_correspondences=30, min_fitness=.20, max_rmse_m=.18),
                   icp_failures=sum(not r['success'] for r in records),
                   maps={name: evaluate(m, boundary, visible) for name, m in maps.items()},
                   poses=dict(odometry=pose_errors(odometry, truth), icp=pose_errors(icp_poses, truth)), pairs=records)
    metrics['acceptance'] = dict(functional_complete=True,
        icp_improves_translation=metrics['poses']['icp']['translation_rmse_m'] < metrics['poses']['odometry']['translation_rmse_m'],
        icp_improves_heading=metrics['poses']['icp']['heading_rmse_deg'] < metrics['poses']['odometry']['heading_rmse_deg'],
        icp_improves_map_f1=metrics['maps']['icp']['boundary_f1'] > metrics['maps']['odometry']['boundary_f1'],
        note='No numerical P9.3 acceptance thresholds were supplied; report measurements without inventing PASS thresholds.')
    output = ROOT/'results'
    output.mkdir(exist_ok=True)
    archive = dict(reference_poses=truth, odometry_poses=odometry, icp_poses=icp_poses,
                   angles_rad=scans[0]['angles_rad'], ranges_m=np.array([s['ranges_m'] for s in scans]),
                   valid=np.array([s['valid'] for s in scans]), true_boundary=boundary, visible_boundary=visible,
                   parked_vehicles=np.array(scenario['parked_vehicles']))
    for name, mapper in maps.items():
        archive.update(mapper.arrays(name+'_'))
    archive_path = output/'p9_occupancy_maps.npz'
    np.savez_compressed(archive_path, **archive)
    # Verify public load API on every persisted map, including classification.
    for name, mapper in maps.items():
        restored = OccupancyGridMapper.load(archive_path, name+'_')
        assert np.array_equal(restored.states, mapper.states)
        assert np.array_equal(restored.log_odds, mapper.log_odds)
    metrics['persistence_roundtrip_verified'] = True
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    cmap = ListedColormap(['#999999', 'white', 'black'])
    for ax, (name, mapper), estimate, title in zip(axes, maps.items(), [truth, odometry, icp_poses],
                                                 ['Ground Truth Map', 'Odometry Map', 'ICP Map']):
        ax.imshow(mapper.states+1, origin='lower', extent=[grid.origin_x, grid.origin_x+grid.width,
                  grid.origin_y, grid.origin_y+grid.height], cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        ax.plot(truth[:, 0], truth[:, 1], color='tab:blue', linewidth=1.5, label='Reference')
        ax.plot(estimate[:, 0], estimate[:, 1], '--', color='tab:orange', linewidth=1.2, label='Map poses')
        score = metrics['maps'][name]
        ax.set_title(f"{title}\nPrecision {score['boundary_precision']:.3f} | Recall {score['boundary_recall']:.3f} | F1 {score['boundary_f1']:.3f}")
        ax.set(xlabel='X (m)', xlim=(grid.origin_x, grid.origin_x+grid.width), ylim=(grid.origin_y, grid.origin_y+grid.height))
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=8)
    axes[0].set_ylabel('Y (m)')
    fig.legend(handles=[Patch(facecolor='black', label='Occupied'), Patch(facecolor='white', edgecolor='black', label='Free'),
                        Patch(facecolor='#999999', label='Unknown / uncertain')], loc='lower center', ncol=3)
    fig.suptitle('P9.3 Occupancy Grid Mapping | 0.05 m | shared cached scans')
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output/'p9_occupancy_grid_mapping.png', dpi=180)
    plt.close(fig)
    (output/'p9_mapping_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
