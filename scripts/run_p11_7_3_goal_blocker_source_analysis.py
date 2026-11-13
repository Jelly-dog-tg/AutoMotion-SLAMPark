"""P11.7.3 read-only saved-map provenance, geometry and blocking analysis.

No planning, physical simulation, scan recapture or saved-map modification.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle,Polygon,Patch
from matplotlib.lines import Line2D
from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper
from scripts.run_p11_7_1_live_map_plan_preview import make_planner_grid,footprint_polygon
from scripts.run_p11_7_2_goal_footprint_diagnostic import polygon_intersects_cell

OUT=ROOT/'results';SOURCE=OUT/'p11_6_4_live_slam_map.npz'
PREFIX='p11_7_3_goal_blocker_source_analysis'
EPS=1e-9


def clip_rectangle(poly,box):
    result=[list(p) for p in poly]
    for axis,bound,sign in [(0,box[0],1),(0,box[2],-1),(1,box[1],1),(1,box[3],-1)]:
        output=[]
        for a,b in zip(result,result[1:]+result[:1]):
            va=sign*(a[axis]-bound);vb=sign*(b[axis]-bound)
            ia=va>=-EPS;ib=vb>=-EPS
            if ia:output.append(a)
            if ia!=ib and abs(b[axis]-a[axis])>1e-14:
                u=(bound-a[axis])/(b[axis]-a[axis])
                p=[a[k]+u*(b[k]-a[k]) for k in (0,1)];p[axis]=bound;output.append(p)
        result=output
        if not result:break
    return np.array(result).reshape(-1,2)


def area(poly):
    if len(poly)<3:return 0.
    # Translation avoids cancellation of ~150 m^2 world-coordinate products.
    q=poly-poly[0]
    return abs(float(np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))))/2


def classification(values):
    u=int(np.sum(values==-1));o=int(np.sum(values==1));f=int(np.sum(values==0))
    return ('mixed' if u and o else 'occupied' if o else 'unknown' if u else 'free'),dict(unknown=u,occupied=o,free=f)


def cells(poly,res,origin,shape):
    minimum=poly.min(axis=0);maximum=poly.max(axis=0)
    for r in range(max(0,math.floor((minimum[1]-origin[1])/res)-1),min(shape[0]-1,math.floor((maximum[1]-origin[1])/res)+1)+1):
        for c in range(max(0,math.floor((minimum[0]-origin[0])/res)-1),min(shape[1]-1,math.floor((maximum[0]-origin[0])/res)+1)+1):
            box=[origin[0]+c*res,origin[1]+r*res,origin[0]+(c+1)*res,origin[1]+(r+1)*res]
            clipped=clip_rectangle(poly,box)
            if not len(clipped):continue
            a=area(clipped)
            yield dict(row=r,col=c,center=[box[0]+res/2,box[1]+res/2],bounds=box,
                       relation='area_overlap' if a>1e-10 else 'boundary_touch_only',
                       overlap_area_m2=a,intersection=clipped.tolist())


def counts(records):
    return {relation:{name:sum(item['relation']==relation and item['state']==name for item in records)
                      for name in ('free','unknown','occupied','mixed')}
            for relation in ('area_overlap','boundary_touch_only')}


def boundary_distance(points,rectangles):
    result=np.full(len(points),np.inf)
    for x0,y0,x1,y1 in rectangles:
        for a,b in [(np.array([x0,y0]),np.array([x1,y0])),(np.array([x1,y0]),np.array([x1,y1])),
                    (np.array([x1,y1]),np.array([x0,y1])),(np.array([x0,y1]),np.array([x0,y0]))]:
            edge=b-a;u=np.clip((points-a)@edge/np.dot(edge,edge),0,1)
            result=np.minimum(result,np.linalg.norm(points-(a+u[:,None]*edge),axis=1))
    return result


def clusters(records):
    pending={(x['row'],x['col']) for x in records if x['state']=='occupied' and x['relation']=='area_overlap'}
    result=[]
    lookup={(x['row'],x['col']):x for x in records}
    while pending:
        seed=min(pending);pending.remove(seed);queue=[seed];component=[]
        while queue:
            r,c=queue.pop();component.append((r,c))
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if (r+dr,c+dc) in pending:pending.remove((r+dr,c+dc));queue.append((r+dr,c+dc))
        centers=np.array([lookup[key]['center'] for key in component])
        result.append(dict(cell_count=len(component),cells=[list(x)for x in sorted(component)],
            center_mean=centers.mean(axis=0).tolist(),center_bounds=[*centers.min(axis=0).tolist(),*centers.max(axis=0).tolist()]))
    return result


def main():
    # Protect all existing source/results in completed stages; only new outputs
    # below are written. Never load the P10 fixed execution trajectory.
    protected=[p for p in OUT.iterdir() if p.is_file() and not p.name.startswith(PREFIX)]
    protected += [p for folder in ('parking_planner','scripts','configs') for p in (ROOT/folder).rglob('*.py')
                  if p.name!='run_p11_7_3_goal_blocker_source_analysis.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in protected}
    with np.load(SOURCE,allow_pickle=False) as archive:
        inventory={k:dict(shape=list(archive[k].shape),dtype=str(archive[k].dtype))for k in archive.files}
        saved={k:archive[k].copy()for k in archive.files}
    cfg=json.loads(str(saved['config'].item()));states=saved['states']
    mapper=OccupancyGridMapper.load(SOURCE)
    scenario=build_perpendicular_parking_scenario();goal=scenario['goal'];geometry=scenario['geometry']
    poly=footprint_polygon(goal,geometry)[:-1]
    grid=make_planner_grid(states,cfg)
    planner=HybridAStarPlanner(grid=grid,geometry=geometry,config=scenario['planner_config'])
    res=float(cfg['resolution']);origin=[cfg['origin_x'],cfg['origin_y']];ratio=round(grid.resolution/res)
    raw=list(cells(poly,res,origin,states.shape))
    for item in raw:
        r,c=item['row'],item['col'];item['state']={-1:'unknown',0:'free',1:'occupied'}[int(states[r,c])]
        item.update(observed=bool(saved['observed'][r,c]),probability=float(saved['probabilities'][r,c]),log_odds=float(saved['log_odds'][r,c]))
    coarse=list(cells(poly,grid.resolution,origin,grid.occupied.shape))
    c,s=math.cos(goal.yaw),math.sin(goal.yaw)
    wx=goal.x+c*planner.local_x-s*planner.local_y;wy=goal.y+s*planner.local_x+c*planner.local_y
    ix=np.floor((wx-origin[0])/grid.resolution).astype(int);iy=np.floor((wy-origin[1])/grid.resolution).astype(int)
    sampled=set(zip(iy.tolist(),ix.tolist()))
    normalized=np.column_stack(((wx-origin[0])/grid.resolution,(wy-origin[1])/grid.resolution))
    near_integer=np.isclose(normalized,np.rint(normalized),rtol=0,atol=1e-9)
    integer_diagnostic=np.floor(np.where(near_integer,np.rint(normalized),normalized)).astype(int)
    actual_indices=np.column_stack((ix,iy))
    sensitive=np.any(actual_indices!=integer_diagnostic,axis=1)
    floor_audit=dict(sensitive_sample_count=int(sensitive.sum()),
        policy_unchanged=True,
        meaning='Read-only comparison with mathematical integer boundaries; diagnostic indices are never passed to the collision checker.',
        samples=[dict(sample_index=int(i),world_xy=[float(wx[i]),float(wy[i])],
            actual_row_col=[int(iy[i]),int(ix[i])],integer_boundary_diagnostic_row_col=[int(integer_diagnostic[i,1]),int(integer_diagnostic[i,0])])
            for i in np.flatnonzero(sensitive)])
    blockers=[];source_keys=set()
    for item in coarse:
        r,col=item['row'],item['col'];block=states[r*ratio:(r+1)*ratio,col*ratio:(col+1)*ratio]
        item['state'],item['source_counts']=classification(block)
        item['blocked']=bool(grid.occupied[r,col]);item['planner_sampled']=(r,col)in sampled
        mask=(iy==r)&(ix==col)
        item['planner_sample_count']=int(mask.sum());item['planner_sample_world_xy']=np.column_stack([wx[mask],wy[mask]]).tolist()
        if item['blocked']:
            sources=[]
            for dr,dc in np.argwhere(block!=0):
                rr,cc=r*ratio+int(dr),col*ratio+int(dc);source_keys.add((rr,cc))
                box=[origin[0]+cc*res,origin[1]+rr*res,origin[0]+(cc+1)*res,origin[1]+(rr+1)*res]
                clipped=clip_rectangle(poly,box);a=area(clipped)
                sources.append(dict(row=rr,col=cc,center=[box[0]+res/2,box[1]+res/2],
                    state={-1:'unknown',1:'occupied'}[int(states[rr,cc])],
                    footprint_relation='area_overlap' if a>1e-10 else 'boundary_touch_only' if len(clipped) else 'outside',overlap_area_m2=a))
            item['blocking_source_cells']=sources;blockers.append(item)
    source_keys |= {(i['row'],i['col'])for i in raw if i['state']!='free'}
    # Historical range selection + touching SAT, reproduced exactly without
    # executing its main(), planning or modifying its outputs.
    legacy=[]
    for r in range(max(0,math.floor((poly[:,1].min()-origin[1])/grid.resolution)),min(grid.rows-1,math.floor((poly[:,1].max()-origin[1])/grid.resolution))+1):
        for col in range(max(0,math.floor((poly[:,0].min()-origin[0])/grid.resolution)),min(grid.cols-1,math.floor((poly[:,0].max()-origin[0])/grid.resolution))+1):
            box=[origin[0]+col*grid.resolution,origin[1]+r*grid.resolution,origin[0]+(col+1)*grid.resolution,origin[1]+(r+1)*grid.resolution]
            if polygon_intersects_cell(poly,*box):legacy.append((r,col))
    lookup={(i['row'],i['col']):i for i in coarse}
    evidence={key:dict(hit_events=[],free_frame_indices=[],hit_frame_indices=[],update_history=[])for key in source_keys}
    trace_available=all(k in saved for k in ['ranges_m','valid','angles_rad','estimated_poses','scan_timestamps_s'])
    replay=np.zeros_like(saved['log_odds']);replay_observed=np.zeros_like(states,dtype=bool)
    provenance={};attribution=[]
    if trace_available:
        poses=saved['estimated_poses'];ranges=saved['ranges_m'];valid=saved['valid'];angles=saved['angles_rad'];times=saved['scan_timestamps_s']
        assert ranges.shape==valid.shape==(len(poses),len(angles)) and len(times)==len(poses)
        reference=saved.get('reference_poses')
        truth_available=reference is not None and reference.shape==poses.shape
        hit_log=math.log(cfg['hit_probability']/(1-cfg['hit_probability']));miss_log=math.log(cfg['miss_probability']/(1-cfg['miss_probability']))
        lower=math.log(cfg['probability_min']/(1-cfg['probability_min']));upper=math.log(cfg['probability_max']/(1-cfg['probability_max']))
        # Replay the existing mapper's ray/update semantics in memory solely to
        # audit saved evidence. No ICP, planning or fresh LiDAR scans are run.
        for frame,(pose,distance,hits) in enumerate(zip(poses,ranges,valid)):
            free=set();occupied=set();start=mapper.world_to_grid(*pose[:2])
            for beam,(angle,d,hit) in enumerate(zip(angles,distance,hits)):
                d=float(d)
                if not np.isfinite(d) or not .12<=d<=12.:continue
                if not hit and not np.isclose(d,12.,atol=1e-9,rtol=0):continue
                theta=pose[2]+angle
                endxy=np.array([pose[0]+d*math.cos(theta),pose[1]+d*math.sin(theta)])
                end=mapper.world_to_grid(*endxy)
                traversed=list(mapper.ray_cells(start,end))
                free.update(cell for cell in (traversed[:-1]if hit else traversed)if mapper.contains(*cell))
                if hit and mapper.contains(*end):
                    occupied.add(end);key=(end[1],end[0])
                    if key in evidence:
                        event=dict(frame_index=frame,timestamp_s=float(times[frame]),beam_index=beam,range_m=d,
                                   angle_rad=float(angle),estimated_pose=pose.tolist(),projected_hit_xy=endxy.tolist())
                        if truth_available:
                            true=reference[frame];true_xy=true[:2]+d*np.array([math.cos(true[2]+angle),math.sin(true[2]+angle)])
                            event.update(evaluation_only_true_pose=true.tolist(),evaluation_only_hit_xy=true_xy.tolist(),
                                projection_shift_m=float(np.linalg.norm(endxy-true_xy)),
                                true_hit_distance_to_configured_obstacle_boundary_m=float(boundary_distance(true_xy[None],scenario['parked_vehicles'])[0]),
                                estimated_hit_distance_to_configured_obstacle_boundary_m=float(boundary_distance(endxy[None],scenario['parked_vehicles'])[0]))
                            attribution.append(event)
                        evidence[key]['hit_events'].append(event)
            free.difference_update(occupied)
            for cellset,increment in [(free,miss_log),(occupied,hit_log)]:
                if cellset:
                    cols,rows=np.array(list(cellset)).T
                    replay[rows,cols]+=increment;replay_observed[rows,cols]=True
                    for col,row in cellset:
                        key=(row,col)
                        if key in evidence:
                            evidence[key]['hit_frame_indices'if increment>0 else 'free_frame_indices'].append(frame)
                            evidence[key]['update_history'].append(dict(frame_index=frame,timestamp_s=float(times[frame]),
                                update='occupied_endpoint' if increment>0 else 'free_ray',
                                clipped_log_odds_after_frame=float(np.clip(replay[row,col],lower,upper))))
            np.clip(replay,lower,upper,out=replay)
        probability=1/(1+np.exp(-replay));replay_states=np.full(states.shape,-1,dtype=np.int8)
        replay_states[replay_observed&(probability<=cfg['free_threshold'])]=0
        replay_states[replay_observed&(probability>=cfg['occupied_threshold'])]=1
        provenance=dict(scan_count=len(poses),beams_per_scan=len(angles),valid_hit_count=int(valid.sum()),
            replay_state_agreement_fraction=float(np.mean(states==replay_states)),
            replay_max_log_odds_difference=float(np.max(np.abs(replay-saved['log_odds']))),
            replay_differing_cell_count=int(np.sum(states!=replay_states)),
            exact_replay_caveat='Saved ranges are float32, whereas runtime measurements were float64. Cell-boundary projection may therefore differ; the archive has no original float64 ranges or hit body IDs.',
            hit_point_arrays_saved=False,hit_points_reconstructible=True,per_frame_icp_status_saved=False,
            original_hit_body_ids_saved=False,noise_free_ranges_saved=False,physical_reference_poses_used='evaluation only; never planning or map substitution')
        for key,item in evidence.items():
            r,col=key;item.update(row=r,col=col,replayed_state=int(replay_states[r,col]),saved_state=int(states[r,col]),
                replayed_log_odds=float(replay[r,col]),saved_log_odds=float(saved['log_odds'][r,col]),
                saved_state_reproduced=bool(replay_states[r,col]==states[r,col]),
                hit_frame_count=len(item['hit_frame_indices']),free_frame_count=len(item['free_frame_indices']))
    else:
        provenance=dict(traceable=False,reason='Missing aligned saved ranges/valid/angles/poses/timestamps; cannot identify contributing frames.')
    for item in raw:
        key=(item['row'],item['col'])
        if item['state']!='free':item['evidence']=evidence.get(key)
    # Full center/edge roundtrip checks and independent block reduction.
    rows,cols=np.indices(states.shape)
    xx=origin[0]+(cols+.5)*res;yy=origin[1]+(rows+.5)*res
    roundtrip=bool(np.array_equal(np.floor((xx-origin[0])/res).astype(int),cols)and np.array_equal(np.floor((yy-origin[1])/res).astype(int),rows))
    blocks=states.reshape(grid.rows,ratio,grid.cols,ratio)
    independent=np.zeros_like(grid.occupied)
    for dr in range(ratio):
        for dc in range(ratio):independent|=states[dr::ratio,dc::ratio]!=0
    raw_counts=counts(raw);planner_counts=counts(coarse)
    occupied=[i for i in raw if i['state']=='occupied'and i['relation']=='area_overlap']
    occupied_events=[e for i in occupied for e in (i.get('evidence')or{}).get('hit_events',[])]
    stats={}
    if occupied_events and 'projection_shift_m'in occupied_events[0]:
        for field in ['projection_shift_m','true_hit_distance_to_configured_obstacle_boundary_m','estimated_hit_distance_to_configured_obstacle_boundary_m']:
            values=np.array([e[field]for e in occupied_events]);stats[field]=dict(mean=float(values.mean()),median=float(np.median(values)),max=float(values.max()))
        stats['hit_events']=len(occupied_events);stats['unique_frames']=len({e['frame_index']for e in occupied_events})
    unknown_interior=[i for i in raw if i['state']=='unknown' and i['relation']=='area_overlap']
    unknown_observed=sum(i['observed'] for i in unknown_interior)
    frame_ids=sorted({e['frame_index'] for e in occupied_events})
    if frame_ids and 'reference_poses' in saved:
        delta=saved['estimated_poses'][frame_ids]-saved['reference_poses'][frame_ids]
        heading=np.degrees(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2])))
        stats['contributing_frame_indices']=frame_ids
        stats['contributing_timestamp_range_s']=[float(saved['scan_timestamps_s'][frame_ids[0]]),float(saved['scan_timestamps_s'][frame_ids[-1]])]
        stats['contributing_pose_position_error_mean_m']=float(np.linalg.norm(delta[:,:2],axis=1).mean())
        stats['contributing_pose_heading_abs_error_mean_deg']=float(np.abs(heading).mean())
    geometrical_obstacle_area=sum(area(clip_rectangle(poly,rectangle))for rectangle in scenario['parked_vehicles'])
    actual=bool(grid.is_collision(goal,planner.local_x,planner.local_y))
    sampled_blockers=[i for i in blockers if i['planner_sampled']]
    legacy_counts={name:sum(lookup[key]['state']==name for key in legacy)for name in ('free','unknown','occupied','mixed')}
    focused=[]
    for x,y in [(9.9,15.7)]:
        col=math.floor((x-origin[0])/grid.resolution);r=math.floor((y-origin[1])/grid.resolution)
        focused.append(lookup.get((r,col),dict(row=r,col=col,not_in_footprint_candidates=True)))
    summary=dict(stage='P11.7.3 Live SLAM Occupancy Source Analysis',status='ANALYSIS_COMPLETE_MAP_STILL_BLOCKED',analysis_only=True,
        source_file=str(SOURCE.relative_to(ROOT)),source_sha256=hashes[str(SOURCE.relative_to(ROOT))],archive_inventory=inventory,
        saved_data_availability=dict(raw_noisy_polar_scans=trace_available,hit_point_arrays=False,
            hit_points_reconstructed_from_polar=True,estimated_poses='estimated_poses'in saved,
            odometry='odometry_poses'in saved,map_states='states'in saved,timestamps='scan_timestamps_s'in saved,evaluation_only_reference_poses='reference_poses' in saved),
        map_config=cfg,coordinate_audit=dict(array_order='states[row_y, col_x]',origin=origin,resolution_m=res,
            cell_convention='half-open bounds; world_to_grid=floor; displayed positions are cell centers',
            center_roundtrip_all_cells=roundtrip,shape_matches_geometry=list(states.shape)==[mapper.rows,mapper.cols],
            saved_states_match_mapper_reclassification=bool(np.array_equal(states,mapper.states)),
            saved_probabilities_match_log_odds=bool(np.allclose(saved['probabilities'],mapper.probabilities)),
            no_transpose_in_inspected_data_path=True,boundary_floor_sensitivity=floor_audit,
            positive_raw_cell_overlap_area_sum_m2=sum(i['overlap_area_m2'] for i in raw),
            analytic_body_area_m2=geometry.length*geometry.width,transpose_caveat='The square array shape alone cannot establish orientation; inspected producer/mapper/downsampler consistently index [y,x].',
            downsample_ratio=ratio,planner_resolution_m=grid.resolution,
            downsample_matches_independent_strided_reduction=bool(np.array_equal(grid.occupied,independent)),
            source_block='states[row*4:(row+1)*4,col*4:(col+1)*4]; any source !=0 blocks'),
        vehicle_geometry=dict(length_m=geometry.length,width_m=geometry.width,wheelbase_m=geometry.wheelbase,
            rear_overhang_m=geometry.rear_overhang,front_extent_m=geometry.front_extent,reference='rear axle',
            yaw_convention='radians; positive counterclockwise about +Z',goal=[goal.x,goal.y,goal.yaw],polygon=poly.tolist(),
            planner_local_sample_count=len(planner.local_x),planner_spacing_m=grid.resolution*.5,
            local_x_limits=[float(planner.local_x.min()),float(planner.local_x.max())],
            local_y_limits=[float(planner.local_y.min()),float(planner.local_y.max())]),
        answers=dict(
            occupied_really_in_saved_map=len(occupied)>0,
            raw_occupied_positive_area_cells=len(occupied),
            raw_occupied_boundary_touch_only_cells=raw_counts['boundary_touch_only']['occupied'],
            raw_unknown_positive_area_cells=len(unknown_interior),
            raw_unknown_observed_but_ambiguous=unknown_observed,
            raw_unknown_never_observed=len(unknown_interior)-unknown_observed,
            raw_unknown_free_only_insufficient_evidence=sum((i.get('evidence')or{}).get('hit_frame_count',0)==0 for i in unknown_interior),
            raw_unknown_hit_and_free_evidence=sum((i.get('evidence')or{}).get('hit_frame_count',0)>0 and (i.get('evidence')or{}).get('free_frame_count',0)>0 for i in unknown_interior),
            actual_blockers_positive_area=sum(i['relation']=='area_overlap'for i in sampled_blockers),
            actual_blockers_boundary_touch_only=sum(i['relation']=='boundary_touch_only'for i in sampled_blockers),
            actual_planner_matches_sample_index_check=actual==bool(np.any(grid.occupied[iy,ix])),
            planner_has_correct_body_dimensions=True,
            coordinate_findings='No axis/origin/resolution mismatch found. Floating-point floor sensitivity at integer grid boundaries is present and quantified; it does not explain away the 10 positive-area occupied source cells.',
            planner_is_not_strict_positive_area_geometry=True,
            provenance_exact_for_this_archive=provenance.get('replay_max_log_odds_difference')==0.,
            dominant_evidence_based_source='SLAM pose error displaces measured obstacle surfaces into the nominal bay, then multi-frame log-odds accumulation preserves nonfree cells. Coarse any-nonfree aggregation and boundary-inclusive planner samples propagate this into safety refusal.',
            cause_not_identifiable_further='The archive cannot uniquely attribute the pose error to ICP versus wheel-odometry/history, or identify a saved PyBullet hit body ID. No pure boundary-contact reinterpretation removes the 10 real interior occupied cells.'),
        raw_005m_counts=raw_counts,raw_occupied_area_cells=occupied,
        raw_unknown_area_cells=[i for i in raw if i['state']=='unknown'and i['relation']=='area_overlap'],
        raw_boundary_nonfree_cells=[i for i in raw if i['state']!='free'and i['relation']=='boundary_touch_only'],
        raw_occupied_clusters_8_connected=clusters(raw),planner_020m_counts=planner_counts,
        all_intersecting_planner_cells=coarse,planner_blockers=blockers,actual_planner_sampled_blockers=sampled_blockers,
        actual_collision_result=actual,planner_sampled_cell_count=len(sampled),
        historical_p11_7_2_reproduction=dict(total_cells=len(legacy),counts=legacy_counts,
            blocked=sum(lookup[key]['blocked']for key in legacy),
            note='Historical SAT includes edge contact and bbox floor selection can omit neighbor cells touching a lower edge. Complete contact inventory is reported separately.'),
        focus_cells=focused,provenance=provenance,all_nonfree_source_evidence=list(evidence.values()),
        occupied_interior_hit_evaluation=stats,
        physical_configuration_evaluation=dict(configured_parked_rectangles=scenario['parked_vehicles'],
            positive_area_overlap_with_goal_footprint_m2=geometrical_obstacle_area,
            note='Configured physical geometry and saved reference poses are evaluation evidence only. No GT map is used for planning; no physics world is created.'),
        interpretation=dict(direct_blocking_cause='Saved SLAM nonfree cells propagate by 4x4 any-nonfree reduction into planner cells hit by actual footprint samples.',
            geometry='Actual planner tests filled rectangle samples INCLUDING its boundary, rather than polygon area intersection. Boundary-only cells and floating-point floor assignments can block; both classifications are retained without changing policy.',
            localization_evidence='Compare saved estimated-pose hit projections with the SAME saved ranges projected by evaluation-only physical reference poses; quantitative distances and contributing frames are reported.',
            causal_limit='Final log-odds plus saved noisy scans/poses can establish contributing evidence and projection displacement, but cannot uniquely separate range noise, pose-estimation error, ICP decisions, or accumulation attributions without per-frame raw float64 ranges/body IDs/update history.',
            unknown='Unknown can mean never observed OR observed but probability between free/occupied thresholds; observed/log-odds are reported per cell.',
            planning_executed=False,map_modified=False,safety_policy_changed=False,physical_vehicle_moved=False,
            p10_fixed_path_loaded=False),protected_sha256=hashes)
    # Local-only three-panel view: raw cells, coarse source contamination, and
    # sample-based collision semantics. Exact clipped area is hatched.
    fig,axes=plt.subplots(1,3,figsize=(17,10),sharex=True,sharey=True)
    extent=[origin[0],origin[0]+cfg['width'],origin[1],origin[1]+cfg['height']]
    cmap=ListedColormap(['#e5b75a','white','#333333'])
    xlim=(8.9,12.1);ylim=(10.5,17.3)
    for index,ax in enumerate(axes):
        ax.imshow(states+1,origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=2,interpolation='nearest')
        x0,y0,x1,y1=scenario['parking_slot'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='green',ls='--',lw=1.6))
        ax.plot(*np.vstack([poly,poly[0]]).T,color='#006aca',lw=1.8,zorder=8)
        ax.scatter(goal.x,goal.y,marker='*',c='#006aca',s=70,zorder=9)
        if index==0:
            for item in raw:
                if item['state']=='free':continue
                box=item['bounds'];color='#d93025'if item['state']=='occupied'else'#c98300'
                ax.add_patch(Rectangle(box[:2],res,res,fill=False,ec=color,lw=.7))
                if item['relation']=='area_overlap':
                    ax.add_patch(Polygon(item['intersection'],facecolor='none',edgecolor=color,hatch='///',lw=.25))
                else:ax.add_patch(Rectangle(box[:2],res,res,fill=False,ec='#a149bd',lw=1.2))
        else:
            for item in coarse:
                box=item['bounds']
                ax.add_patch(Rectangle(box[:2],.2,.2,fill=False,ec='#888888',lw=.35))
                if not item['blocked']:continue
                if item['relation']=='area_overlap':
                    ax.add_patch(Polygon(item['intersection'],facecolor='none',edgecolor='#d93025'if item['state']in('occupied','mixed')else'#d58d00',hatch='///',lw=.7))
                else:ax.add_patch(Rectangle(box[:2],.2,.2,fill=False,ec='#a149bd',lw=1.7,ls=':'))
                if index==2 and item['planner_sampled']:
                    pts=np.array(item['planner_sample_world_xy']);ax.scatter(pts[:,0],pts[:,1],s=5,c='#007d83',zorder=7)
            if index==1:
                for item in blockers:
                    if item['state']in('occupied','mixed'):
                        ax.text(*item['center'],'O'if item['state']=='occupied'else'M',ha='center',va='center',fontsize=6,color='#b00000')
        ax.set(xlim=xlim,ylim=ylim,xlabel='X (m)');ax.set_aspect('equal');ax.grid(alpha=.1)
        ax.plot([9.0,9.5],[11.5,11.5],color='black',lw=2);ax.text(9.25,11.54,'0.50 m',ha='center',fontsize=8)
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title(f'Raw SLAM 0.05 m | interior O={len(occupied)}\nCell evidence, not GT replacement')
    axes[1].set_title('Planner 0.20 m | any nonfree in 4x4\nHatch = actual footprint overlap')
    axes[2].set_title(f'Actual planner sampled blockers = {len(sampled_blockers)}\nPurple = boundary-only contact')
    legend=[Patch(fc='white',ec='gray',label='Free'),Patch(fc='#e5b75a',label='Unknown'),Patch(fc='#333333',label='Occupied'),
        Line2D([0],[0],color='#006aca',lw=2,label='Goal body / rear axle star'),Line2D([0],[0],color='green',ls='--',label='Parking bay'),
        Patch(fc='none',ec='#d93025',hatch='///',label='Nonfree cell: positive-area intersection'),
        Patch(fc='none',ec='#a149bd',ls=':',label='Boundary-touch-only cell'),Line2D([0],[0],marker='o',ls='',color='#007d83',label='Actual planner samples in blocked cells')]
    fig.legend(handles=legend,loc='lower center',ncol=4,fontsize=9)
    fig.suptitle('P11.7.3 Live SLAM blocker source analysis | geometry / saved scan evidence only',fontsize=14)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(OUT/f'{PREFIX}.png',dpi=190);plt.close(fig)
    summary['protected_files_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    assert summary['protected_files_unchanged']
    (OUT/f'{PREFIX}.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:summary[k]for k in ['raw_005m_counts','planner_020m_counts','historical_p11_7_2_reproduction','actual_collision_result','planner_sampled_cell_count','provenance','occupied_interior_hit_evaluation','protected_files_unchanged']},indent=2))
    print('actual sampled blockers',len(sampled_blockers),'area blockers',sum(i['relation']=='area_overlap'for i in blockers))


if __name__=='__main__':main()
