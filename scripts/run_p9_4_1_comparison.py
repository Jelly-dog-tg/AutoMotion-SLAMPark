"""P9.4.1 paired comparison: exact legacy inputs, unchanged legacy algorithm."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import run_p9_4_scan_to_map_localization as legacy
from parking_planner.localization.point_to_line_icp import PointToLineConfig, estimate_normals, point_to_line_icp, transform


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def synthetic_checks():
    """Independent observability tests; not mixed into the 8-scan statistics."""
    x = np.linspace(-3.,3.,121)
    parallel = np.vstack((np.column_stack((x,np.full_like(x,-1.))), np.column_stack((x,np.full_like(x,1.)))))
    corner = np.vstack((np.column_stack((x,np.full_like(x,-1.))), np.column_stack((np.full_like(x,-3.),x))))
    initial = np.array([.03,.04,.01])
    results = {}
    for name, points in [('parallel_walls',parallel),('perpendicular_surfaces',corner)]:
        result = point_to_line_icp(points,np.array(initial),points)
        results[name] = {k:result[k] for k in ('success','converged','degenerate','rejection_reason','information_eigen_ratio')}
        results[name]['pose_unchanged'] = bool(np.array_equal(result['pose'],initial))
    assert results['parallel_walls']['degenerate'] and results['parallel_walls']['pose_unchanged']
    assert results['perpendicular_surfaces']['success'] and not results['perpendicular_surfaces']['degenerate']
    return results


def main():
    protected = [ROOT/'scripts/run_p9_occupancy_mapping.py',legacy.MAP_FILE,
                 ROOT/'scripts/run_p9_4_scan_to_map_localization.py']
    protected += sorted((ROOT/'results').glob('p9_4_scan_to_map*'))
    protected += [ROOT/'results/p9_mapping_metrics.json',ROOT/'results/p9_occupancy_grid_mapping.png']
    original_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    map_points, obstacles = legacy.load_map()
    truth = legacy.TEST_POSES.copy()
    rng = np.random.default_rng(legacy.SEED)
    initial = truth.copy()
    initial[:,:2] += rng.normal(0,legacy.POSITION_NOISE_STD,(len(truth),2))
    heading_noise = np.deg2rad(rng.normal(0,legacy.HEADING_NOISE_STD_DEG,len(truth)))
    initial[:,2] = [legacy.wrap_angle(y+n) for y,n in zip(truth[:,2],heading_noise)]
    scans = [legacy.simulate_scan(p,obstacles) for p in truth]
    cfg = PointToLineConfig()
    surface = estimate_normals(map_points,cfg)
    rows, details, estimates = [], [], {'initial':initial,'point_to_point':[], 'point_to_line':[]}
    history = json.loads(legacy.OUT_JSON.read_text(encoding='utf-8'))
    reproduction = True
    for i,(gt,init,scan) in enumerate(zip(truth,initial,scans)):
        # Both algorithms receive precisely the same arrays, generated once.
        before = digest(scan,init,map_points)
        old = legacy.scan_to_map_icp(scan,init,map_points)
        new = point_to_line_icp(scan,init,map_points,cfg,surface)
        assert before == digest(scan,init,map_points)
        reproduction &= bool(np.allclose(old['pose'][:2], [history['per_scan_results'][i]['estimated_x_m'],history['per_scan_results'][i]['estimated_y_m']], atol=1e-10,rtol=0))
        reproduction &= abs(legacy.pose_errors(old['pose'],gt)[1]-history['per_scan_results'][i]['final_heading_error_deg']) < 1e-8
        row = dict(scan_id=i+1,scan_points=len(scan),input_sha256=before)
        for name,p in [('ground_truth',gt),('initial',init),('point_to_point',old['pose']),('point_to_line',new['pose'])]:
            row.update({name+'_x_m':float(p[0]),name+'_y_m':float(p[1]),name+'_yaw_rad':float(p[2])})
            t,h = legacy.pose_errors(p,gt)
            row.update({name+'_translation_error_m':t,name+'_heading_error_deg':h})
        for name,result in [('point_to_point',old),('point_to_line',new)]:
            estimates[name].append(result['pose'])
            for key in ('success','converged','iterations','correspondences','fitness','rmse_m'):
                row[name+'_'+key] = result[key]
            for metric in ('translation_error_m','heading_error_deg'):
                delta = row[name+'_'+metric]-row['initial_'+metric]
                row[name+'_'+metric+'_change'] = delta
                row[name+'_'+metric+'_improved'] = bool(delta < -1e-12)
        for key in ('degenerate','rejection_reason','line_rmse_m','information_eigen_ratio','prior_mahalanobis'):
            row['point_to_line_'+key] = new[key]
        # Report line residual for baseline too, using identical PCA and gating.
        centers,normals,valid = surface
        world = transform(scan,old['pose'])
        d2 = np.sum((world[:,None]-map_points[valid][None])**2,axis=2)
        idx = np.argmin(d2,axis=1)
        keep = d2[np.arange(len(scan)),idx] <= cfg.max_correspondence_m**2
        residual = np.sum(normals[valid][idx[keep]]*(world[keep]-centers[valid][idx[keep]]),axis=1)
        row['point_to_point_line_rmse_m'] = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
        details.append(dict(scan_id=i+1,candidate_pose=new['candidate_pose'].tolist(),information_eigenvalues=new['information_eigenvalues']))
        rows.append(row)
    summaries = {}
    for name in estimates:
        t = np.array([r[name+'_translation_error_m'] for r in rows])
        h = np.array([r[name+'_heading_error_deg'] for r in rows])
        summary = dict(translation_rmse_m=float(np.sqrt(np.mean(t*t))), translation_median_m=float(np.median(t)),
                       heading_rmse_deg=float(np.sqrt(np.mean(h*h))),heading_median_deg=float(np.median(h)))
        if name != 'initial':
            summary.update(translation_improved_count=sum(r[name+'_translation_error_m_improved'] for r in rows),
                           heading_improved_count=sum(r[name+'_heading_error_deg_improved'] for r in rows),
                           converged_count=sum(r[name+'_converged'] for r in rows),accepted_count=sum(r[name+'_success'] for r in rows),
                           mean_point_rmse_m=float(np.mean([r[name+'_rmse_m'] for r in rows if r[name+'_rmse_m'] is not None])),
                           mean_line_rmse_m=float(np.mean([r[name+'_line_rmse_m'] for r in rows if r[name+'_line_rmse_m'] is not None])))
        if name != 'initial':
            summary.update(translation_worsened_count=sum(r[name+'_translation_error_m_change'] > 1e-12 for r in rows),
                           heading_worsened_count=sum(r[name+'_heading_error_deg_change'] > 1e-12 for r in rows),
                           translation_unchanged_count=sum(abs(r[name+'_translation_error_m_change']) <= 1e-12 for r in rows),
                           heading_unchanged_count=sum(abs(r[name+'_heading_error_deg_change']) <= 1e-12 for r in rows),
                           mean_correspondences=float(np.mean([r[name+'_correspondences'] for r in rows])),
                           mean_match_ratio=float(np.mean([r[name+'_fitness'] for r in rows])))
        summaries[name] = summary
    checks = synthetic_checks()
    new,old,init = summaries['point_to_line'],summaries['point_to_point'],summaries['initial']
    acceptance = dict(identical_inputs=True,legacy_results_reproduced=reproduction,
        both_rmse_better_than_baseline=all(new[k] < old[k] for k in ('translation_rmse_m','heading_rmse_deg')),
        both_rmse_better_than_initial=all(new[k] < init[k] for k in ('translation_rmse_m','heading_rmse_deg')),
        both_medians_no_worse_than_baseline=all(new[k] <= old[k] for k in ('translation_median_m','heading_median_deg')),
        both_medians_no_worse_than_initial=all(new[k] <= init[k] for k in ('translation_median_m','heading_median_deg')),
        majority_scans_improve_both_metrics=new['translation_improved_count'] > 4 and new['heading_improved_count'] > 4,
        parallel_geometry_rejected=checks['parallel_walls']['degenerate'])
    for relative,h in original_hashes.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == h
    metrics = dict(experiment='P9.4.1 paired Point-to-Line comparison', seed=2026,scan_count=8,
        input_sha256=digest(map_points,truth,initial,*scans),protected_file_sha256=original_hashes,
        protected_files_unchanged=True, lidar=dict(beams=720,fov_deg=360,max_range_m=8,range_noise_std_m=0,
        note='Exact P9.4 simulate_scan reused, intentionally not the different P9.1 sensor configuration.'),
        normal_valid_points=int(surface[2].sum()),normal_total_points=len(map_points),config=new_config(cfg),
        summaries=summaries,geometric_degeneracy_count=sum(r['point_to_line_degenerate'] for r in rows),
        synthetic_geometry_checks=checks,acceptance=acceptance,
        status='PASS' if all(acceptance.values()) else 'REVIEW',per_scan_results=rows,candidate_diagnostics=details,
        analysis=dict(
            translation_worsened_scan_ids=[r['scan_id'] for r in rows if r['point_to_line_translation_error_m_change'] > 1e-12],
            heading_worsened_scan_ids=[r['scan_id'] for r in rows if r['point_to_line_heading_error_deg_change'] > 1e-12],
            rejected_scan_ids=[r['scan_id'] for r in rows if not r['point_to_line_success']],
            interpretation='Point-to-line removes the tangential attraction to discretized map cells. Huber reduces large line residuals; the odometry prior limits correction. Neither removes fixed map bias. Near-parallel normals produce an unobservable along-wall translation in scans 7 and 8; both updates are rejected. Small initial errors in scans 3 and 5 are still worsened slightly by map bias. Heading median is worse than the legacy baseline, so full acceptance is not achieved.'),
        limitations=['Eight fixed poses are a paired regression experiment, not an independent generalization benchmark.',
                    'The fixed ICP map has discretization and accumulated mapping-pose errors; geometric fit does not establish absolute accuracy.',
                    'Rejected scans retain odometry; no scan is removed from statistics.',
                    'PCA/Huber/data-only observability and prior settings are fixed across scans and do not use ground truth.'])
    out = ROOT/'results'
    (out/'p9_4_1_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'p9_4_1_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig,axes = plt.subplots(2,2,figsize=(14,10))
    colors = {'initial':'tab:orange','point_to_point':'tab:red','point_to_line':'tab:blue'}
    axes[0,0].scatter(map_points[:,0],map_points[:,1],s=4,c='black',label='Fixed ICP map')
    axes[0,0].plot(truth[:,0],truth[:,1],'g*--',label='Ground truth')
    for name,trajectory in estimates.items():
        trajectory = np.array(trajectory)
        axes[0,0].scatter(trajectory[:,0],trajectory[:,1],s=35,c=colors[name],marker='x',label=name)
        for ax,metric in [(axes[0,1],'translation_error_m'),(axes[1,0],'heading_error_deg')]:
            ax.plot(range(1,9),[r[name+'_'+metric] for r in rows],'o-',color=colors[name],label=name)
    axes[0,0].set(title='Poses on unchanged ICP map',xlabel='X (m)',ylabel='Y (m)',aspect='equal')
    axes[0,1].set(title='Per-scan translation errors (all 8 scans)',xlabel='Scan',ylabel='Error (m)',xticks=range(1,9))
    axes[1,0].set(title='Per-scan heading errors (all 8 scans)',xlabel='Scan',ylabel='Error (deg)',xticks=range(1,9))
    for name in ('point_to_point','point_to_line'):
        axes[1,1].plot(range(1,9),[r[name+'_line_rmse_m'] for r in rows],'o-',label=name,color=colors[name])
    rejected = [r['scan_id'] for r in rows if not r['point_to_line_success']]
    for i in rejected:
        axes[1,1].axvline(i,color='gray',alpha=.3)
    axes[1,1].set(title=f'Comparable point-to-line residuals; rejected={rejected}',xlabel='Scan',ylabel='Line RMSE (m)',xticks=range(1,9))
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('P9.4.1 | Identical legacy inputs | PCA + Huber + odometry prior + observability gate')
    fig.tight_layout()
    fig.savefig(out/'p9_4_1_comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps({k:metrics[k] for k in ('summaries','geometric_degeneracy_count','synthetic_geometry_checks','acceptance','status')},indent=2))
    for r in rows:
        print(r['scan_id'], 't:',*[round(r[n+'_translation_error_m'],5) for n in estimates],
              'h:',*[round(r[n+'_heading_error_deg'],5) for n in estimates],
              r['point_to_line_rejection_reason'],round(r['point_to_line_information_eigen_ratio'],6))


def new_config(cfg):
    from dataclasses import asdict
    return asdict(cfg)


if __name__ == '__main__':
    main()
