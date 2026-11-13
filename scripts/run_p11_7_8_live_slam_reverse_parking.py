"""P11.7.8 live-map reverse-parking safety preflight and evidence capture.

Historical data are read-only. Physical truth initializes the sensor scene only;
scan registration receives local returns and an odometry-propagated SLAM prior.
No motion/planning is allowed without a supported localization-error bound.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon,Patch
import run_p11_7_4_5_live_slam_hybrid_astar_physical_validation as live
from run_p11_7_3_goal_blocker_source_analysis import cells,counts,classification,clip_rectangle,area
from run_p11_7_6_robust_hybrid_astar_physical_validation import envelope_outline,clean
from parking_planner.vehicle.ackermann import AckermannModel,VehicleState
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.robust_clearance import CellClearance
from parking_planner.localization.point_to_line_icp import PointToLineConfig,point_to_line_icp,transform
from parking_planner.sensing.lidar_2d import LidarConfig

OUT=ROOT/'results';PREFIX='p11_7_8_parking'
PRIOR=OUT/'p11_7_6_robust_physical_metrics.json'
# Declared before observing new sensor results; no post-experiment relaxation.
POLICY=dict(execution_envelope_m=.13,stop_speed_mps=.01,stop_hold_s=2.,
    final_position_error_m=.10,final_heading_error_deg=2.,
    forward_speed_mps=.28,reverse_speed_mps=.20,physics_hz=240,
    require_entire_body_inside_original_slot=True,require_final_segment_reverse=True,
    require_certified_localization_bound=True,allow_unbounded_icp_covariance_as_bound=False,
    allow_truth_pose_as_planning_input=False,allow_occupied_or_unknown_override=False,
    allow_motion_before_localization_certificate=False)
TRACE_COLUMNS=['time_s','rear_x','rear_y','yaw_rad','actual_speed_mps','actual_steer_rad','gear',
    'target_speed_mps','physical_obstacle_contacts','diagnostic_map_clearance_m']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def historical_inputs():
    required=[OUT/'p11_7_6_clearance_diagnostic.json',OUT/'p11_7_6_robust_hybrid_astar_path.npz',PRIOR,
        OUT/'p11_7_6_robust_physical_trace.npz',OUT/'p11_7_6_robust_physical_report.md',live.MAP]
    for p in required:
        if not p.is_file():raise FileNotFoundError(p)
    previous=json.loads(PRIOR.read_text(encoding='utf-8'))
    diagnostic=json.loads(required[0].read_text(encoding='utf-8'))
    report=required[4].read_text(encoding='utf-8')
    with np.load(required[1],allow_pickle=False)as a:old_path=a['path'].copy()
    with np.load(required[3],allow_pickle=False)as a:trace=a['trace'].copy();columns=list(a['columns'])
    names=['time_s','rear_x','rear_y','yaw_rad','actual_speed','actual_steer','gear']
    indexes=[columns.index(k)for k in names]
    valid=np.flatnonzero(np.isfinite(trace[:,indexes]).all(axis=1))
    if not len(valid):raise ValueError('No valid P11.7.6 physical state')
    last_index=int(valid[-1]);last=trace[last_index];actual={k:float(last[j])for k,j in zip(names,indexes)}
    if actual['gear']not in(-1.,0.,1.):raise ValueError('Invalid recorded gear')
    if previous['map_sha256']!=sha(live.MAP):raise ValueError('Historical map hash mismatch')
    window=trace[(trace[:,0]>=last[0]-POLICY['stop_hold_s'])&(trace[:,0]<=last[0])]
    stopped=bool(len(window)>=480 and np.all(np.abs(window[:,columns.index('actual_speed')])<=POLICY['stop_speed_mps']))
    if not stopped:raise ValueError('Historical terminal vehicle not confirmed stationary for two seconds')
    return previous,diagnostic,report,old_path,trace,columns,actual,dict(last_valid_index=last_index,
        trailing_invalid_rows=len(trace)-last_index-1,stopped_hold_2s=stopped,
        input_sha256={str(p.relative_to(ROOT)):sha(p)for p in required})


def odometry_prediction(saved,trace,columns,geometry):
    # World x/y/yaw in the physical trace are deliberately NOT used here.
    model=AckermannModel(wheelbase=geometry.wheelbase)
    pose=np.asarray(saved['estimated_poses'][-1],float).copy()
    ti,vi,si=[columns.index(k)for k in ('time_s','actual_speed','actual_steer')]
    for a,b in zip(trace[:-1],trace[1:]):
        dt=float(b[ti]-a[ti])
        if dt<=0 or not np.isfinite([dt,a[vi],a[si]]).all():raise ValueError('Invalid odometry interval')
        state=model.step(VehicleState(*pose,float(a[vi])),0.,float(a[si]),dt)
        pose=np.array([state.x,state.y,state.yaw])
    return pose,dict(initial_source='P11.6.4 estimated_poses[-1]',
        propagation_source='P11.7.6 recorded signed speed + actual mean front steer + timestamp only',
        world_pose_columns_used=False,model='existing AckermannModel.step',
        limitation='Recorded body speed is a simulated odometry proxy, not raw wheel encoders. Mean wheel angle is not the equivalent bicycle angle. No encoder/slip/initial-pose covariance or map deformation bound is saved; this prior has no certified error bound.')


def records(poly,saved,cfg,grid):
    raw=[];coarse=[];ratio=round(grid.resolution/cfg['resolution'])
    for item in cells(poly,cfg['resolution'],[cfg['origin_x'],cfg['origin_y']],saved['states'].shape):
        r,c=item['row'],item['col'];v=int(saved['states'][r,c])
        item.update(state={-1:'unknown',0:'free',1:'occupied'}[v],observed=bool(saved['observed'][r,c]),
            log_odds=float(saved['log_odds'][r,c]),probability=float(saved['probabilities'][r,c]))
        raw.append(item)
    for item in cells(poly,grid.resolution,[grid.origin_x,grid.origin_y],grid.occupied.shape):
        r,c=item['row'],item['col'];block=saved['states'][r*ratio:(r+1)*ratio,c*ratio:(c+1)*ratio]
        kind,parts=classification(block);item.update(state=kind,source_counts=parts,planner_blocked=bool(grid.occupied[r,c]))
        # Report blockers due solely to evidence OUTSIDE the actual overlap.
        overlapping=[x for x in raw if x['row']//ratio==r and x['col']//ratio==c and x['relation']=='area_overlap']
        item['positive_area_sources_nonfree']=sum(x['state']!='free'for x in overlapping)
        item['aggregation_only_positive_area_blocker']=bool(item['relation']=='area_overlap'and item['planner_blocked']and overlapping and not item['positive_area_sources_nonfree'])
        coarse.append(item)
    return dict(raw_counts=counts(raw),planner_counts=counts(coarse),raw_nonfree=[x for x in raw if x['state']!='free'],
        planner_nonfree=[x for x in coarse if x['state']!='free'],
        aggregation_only_cells=[x for x in coarse if x['aggregation_only_positive_area_blocker']])


def goal_diagnostic(saved,cfg,grid,scenario):
    g=scenario['geometry'];goal=scenario['goal'];pose=np.array([goal.x,goal.y,goal.yaw])
    index=CellClearance(grid.occupied,grid.resolution,[grid.origin_x,grid.origin_y],g);poly=index.polygon(pose)
    result=records(poly,saved,cfg,grid)
    planner=HybridAStarPlanner(grid=grid,geometry=g,config=scenario['planner_config'])
    result.update(goal_pose=pose.tolist(),footprint=poly.tolist(),nominal_planner_collision=grid.is_collision(goal,planner.local_x,planner.local_y),
        exact_body_collision=index.within(pose,0.),execution_envelope_blocked=index.within(pose,POLICY['execution_envelope_m']),
        nearest_blocked_cell=index.nearest(pose),footprint_inside_slot=area(clip_rectangle(poly,scenario['parking_slot']))>=g.length*g.width-1e-9)
    x0,y0,x1,y1=scenario['parking_slot'];half_cell=cfg['resolution']/2
    result['region_definition']='Slot entrance strip extends from slot lower Y to target body front Y plus one planner cell; boundary strips are one original SLAM cell wide.'
    front_y=float(poly[:,1].min())
    for label,box in dict(entry=[x0,y0,x1,front_y+grid.resolution],slot_interior=list(scenario['parking_slot']),
        left_boundary=[x0-half_cell,y0,x0+half_cell,y1],right_boundary=[x1-half_cell,y0,x1+half_cell,y1]).items():
        x0,y0,x1,y1=box;region=records(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]),saved,cfg,grid)
        result[label]=dict(bounds=box,raw_counts=region['raw_counts'],planner_counts=region['planner_counts'])
    result['execution_envelope_blocked_cells']=[]
    distances=index.exact_distances(pose,index.boxes)
    for j in np.flatnonzero(distances<=POLICY['execution_envelope_m']+1e-12):
        result['execution_envelope_blocked_cells'].append(dict(row=int(index.rows[j]),col=int(index.cols[j]),bounds=index.boxes[j].tolist(),body_distance_m=float(distances[j])))
    result['blocking_classes']=dict(A=dict(saved_occupied_present=result['raw_counts']['area_overlap']['occupied']>0,
        confirmed_current_physical_obstacle=None,interpretation='Saved occupied evidence is real in the archive; its correspondence to a current physical obstacle is unresolved without trustworthy localization. Preserve all occupied labels.'),
        B=dict(unknown_present=result['raw_counts']['area_overlap']['unknown']>0,
            observed_ambiguous=sum(x['state']=='unknown'and x['observed']and x['relation']=='area_overlap'for x in result['raw_nonfree'])),
        C=dict(aggregation_only_positive_area_count=len(result['aggregation_only_cells'])),
        D=dict(array_coordinate_anomaly_detected=False,checks='Existing load_live validation; probabilities/log-odds/observed/states consistent; row=y col=x; origin and 4x4 conservative aggregation unchanged.'))
    return result


def stationary_observation(actual,prior,grid,scenario,gui):
    import pybullet as p
    from parking_planner.simulation.pybullet.scene import ParkingScene
    from parking_planner.simulation.pybullet.physics_vehicle import PhysicsVehicle
    from parking_planner.sensing.pybullet_lidar import PyBulletLidar2D
    from parking_planner.simulation.pybullet.hud import EdgeStatusBars
    from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS
    from scripts.run_p9_5_3_keyboard_exploration import collision
    cid=p.connect(p.GUI if gui else p.DIRECT,options=WINDOW_OPTIONS if gui else '')
    rows=[];forbidden=[];originals={};picture=None
    try:
        dt=1/POLICY['physics_hz'];p.setTimeStep(dt,physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=cid)
        scene=ParkingScene(cid,scenario)
        physical_pose=[actual[k]for k in ('rear_x','rear_y','yaw_rad')]
        car=PhysicsVehicle(cid,scenario['geometry'],physical_pose)
        # Restore only the stopped gearbox's software state; no wheel drive.
        car.direction=int(actual['gear'])
        for name in ('resetBasePositionAndOrientation','resetBaseVelocity','resetJointState','applyExternalForce','applyExternalTorque'):
            fn=getattr(p,name);originals[name]=fn
            def guard(*a,_fn=fn,_name=name,**kw):
                body=a[0]if a else kw.get('bodyUniqueId',kw.get('objectUniqueId'))
                if body==car.body:forbidden.append(_name);raise RuntimeError('Forbidden vehicle motion: '+_name)
                return _fn(*a,**kw)
            setattr(p,name,guard)
        hud=None
        if gui:
            scene.configure_gui();hud=EdgeStatusBars(cid)
        index=CellClearance(grid.occupied,grid.resolution,[grid.origin_x,grid.origin_y],scenario['geometry'])
        # One stationary sensor session, not a parking/control experiment.
        for step in range(481):
            state=car.state();contacts=[c for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid)if c[2]in scene.obstacle_ids and(c[8]<=0 or c[9]>0)]
            diagnostic_clearance=index.nearest([state[k]for k in ('x','y','yaw')])[0]
            rows.append([step*dt,state['x'],state['y'],state['yaw'],state['speed'],state['steering'],state['direction'],0.,len(contacts),diagnostic_clearance])
            if contacts or collision(VehicleState(state['x'],state['y'],state['yaw']),scenario):raise RuntimeError('Stationary observation collision')
            if step<480:car.command(0.,actual['actual_steer'],dt);p.stepSimulation(physicsClientId=cid)
        state=car.state()
        lidar_cfg=LidarConfig(num_beams=180,fov_deg=270.,min_range_m=.12,max_range_m=12.,range_std_m=.015)
        lidar=PyBulletLidar2D(cid,car.body,lidar_cfg,height_m=1.25)
        # Exactly one fresh scan; every matching attempt reuses it.
        scan=lidar.scan(state)
        if gui:
            hud.update('P11.7.8 | LOCALIZATION SAFETY GATE | STATIONARY',
                f'v={state["speed"]:+.5f} m/s | steer={math.degrees(state["steering"]):+.2f} deg | yaw={math.degrees(state["yaw"]):+.2f} deg',
                'No autonomous motion before localization + goal clearance certification','Stationary Ray Casting; original SLAM retained')
        view=p.computeViewMatrixFromYawPitchRoll([10.3,12.2,0],17,35,-62,0,2)
        projection=p.computeProjectionMatrixFOV(55,1.375,.1,100)
        picture=np.array(p.getCameraImage(1100,800,viewMatrix=view,projectionMatrix=projection,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER,physicsClientId=cid)[2]).reshape(800,1100,4)
        metadata=dict(lidar_config=asdict(lidar_cfg),ray_cast_count=1,scan_timestamp_s=2.,
            drive_commands_all_zero=True,vehicle_initialized_only_once=True,forbidden_motion_calls=forbidden,
            restored_software_gear=int(actual['gear']),steering_restoration='Existing steering motors commanded with recorded mean angle; no resetJointState. Trace lacks separate wheel angles.',
            physical_pose_used_only_for_sensor_simulation_and_evaluation=True,points_world_excluded_from_registration=True,
            stationary_speed_max_mps=float(np.max(np.abs(np.array(rows)[240:,4]))),
            stationary_pose_drift_m=float(np.linalg.norm(np.array(rows)[-1,1:3]-physical_pose[:2])))
        return scan,np.array(rows),metadata,picture
    finally:
        for name,fn in originals.items():setattr(p,name,fn)
        if p.isConnected(cid):p.disconnect(cid)


def localization_check(scan,prediction,saved,cfg,scenario):
    rows,cols=np.where(saved['states']==1)
    points=np.column_stack((cfg['origin_x']+(cols+.5)*cfg['resolution'],cfg['origin_y']+(rows+.5)*cfg['resolution']))
    icp_config=PointToLineConfig()
    match=point_to_line_icp(scan['points_local'],prediction,points,icp_config)
    # A converged local optimizer does NOT certify absolute map accuracy.
    result=dict(initial_prediction=prediction.tolist(),map_source='saved Occupied cell centers only',
        local_point_count=len(scan['points_local']),registration=match,
        reference_poses_read=False,physical_xy_yaw_supplied_to_icp=False,
        certified_translation_bound_m=None,certified_heading_bound_deg=None,certified_body_displacement_bound_m=None,
        trusted_pose=None,certificate_passed=False,
        certificate_reasons=['No map-to-physical-world deformation/error bound is stored.',
            'Previous endpoint localization discrepancy is evaluation evidence, not a transferable covariance or a correction.',
            'A single scan-to-map residual/eigenvalue or Gaussian odometry prior cannot bound systematic map errors.',
            'The saved physical trace lacks independent wheel-encoder increments and their uncertainty; propagated mean-steer odometry is diagnostic only.'])
    if not match['success']:result['certificate_reasons'].insert(0,'Current scan-to-map rejected: '+str(match['rejection_reason']))
    goal=scenario['goal'];bearing=math.atan2(goal.y-prediction[1],goal.x-prediction[0])-prediction[2]
    bearing=math.atan2(math.sin(bearing),math.cos(bearing))
    cfg_lidar=LidarConfig(num_beams=len(scan['angles_rad']))
    result['sensor_visibility']=dict(goal_center_relative_bearing_deg=math.degrees(bearing),
        half_fov_deg=cfg_lidar.fov_deg/2,goal_center_outside_fov=abs(math.degrees(bearing))>cfg_lidar.fov_deg/2,
        bearing_pose_source='untrusted odometry-predicted pose; diagnostic only',valid_returns=len(scan['points_local']),
        minimum_correspondences=icp_config.min_correspondences,
        geometric_degeneracy_test_assessable=match['correspondences']>=icp_config.min_correspondences)
    result['nominal_parking_lateral_clearance_m']=(scenario['parking_slot'][2]-scenario['parking_slot'][0]-scenario['geometry'].width)/2
    result['necessary_max_localization_body_bound_m']=result['nominal_parking_lateral_clearance_m']-POLICY['execution_envelope_m']
    result['bound_budget_note']='0.40 m nominal side clearance minus 0.13 m execution envelope leaves at most 0.27 m for localization BEFORE coarse-map obstacles and turn sweep; this is a necessary budget, not an asserted uncertainty bound.'
    return result


def render(result,saved,cfg,scenario,trace,scan,frame):
    g=scenario['geometry'];goal=result['original_goal'];actual=result['actual_terminal_state']
    actual_pose=[actual[k]for k in ('rear_x','rear_y','yaw_rad')]
    index=CellClearance(np.zeros((1,1),bool),cfg['resolution'],[cfg['origin_x'],cfg['origin_y']],g)
    fig,axes=plt.subplots(3,3,figsize=(18,14),layout='constrained')
    extent=[cfg['origin_x'],cfg['origin_x']+cfg['width'],cfg['origin_y'],cfg['origin_y']+cfg['height']]
    for ax in (axes[0,0],axes[0,1]):
        ax.imshow(saved['states']+1,origin='lower',extent=extent,interpolation='nearest',cmap=ListedColormap(['#999999','white','#151515']),vmin=0,vmax=2)
        x0,y0,x1,y1=scenario['parking_slot'];ax.plot([x0,x0,x1,x1],[y0,y1,y1,y0],color='goldenrod',lw=2,label='Original parking slot')
        ax.add_patch(Polygon(index.polygon(goal),fill=False,ec='red',label='Full goal body'))
        ax.plot(*envelope_outline(np.array(goal),g,.13).T,color='purple',lw=1,label='Execution envelope 0.13m')
        ax.plot(*np.vstack([index.polygon(actual_pose),index.polygon(actual_pose)[0]]).T,color='green',lw=1.3,label='Historical actual terminal body (audit)')
        ax.set(xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
    axes[0,0].set_title('Original SLAM; no certified planning path')
    axes[0,0].legend(handles=[Patch(color='#999999',label='Unknown'),Patch(color='white',ec='gray',label='Free'),Patch(color='#151515',label='Occupied')],fontsize=8)
    axes[0,1].set(xlim=(8.7,12.3),ylim=(10.5,17.4),title='Original bay / complete goal footprint');axes[0,1].legend(fontsize=7)
    ax=axes[0,2];ax.imshow(frame);ax.axis('off');ax.set_title('Stationary sensor scene; no parking execution')
    ax=axes[1,0]
    if len(scan['points_local']):ax.scatter(*scan['points_local'].T,s=10,label='One fresh physical LiDAR scan')
    ax.scatter(0,0,marker='^',c='blue');ax.set(title='Sensor-frame returns (180 beams / 270 deg)',xlabel='Local forward (m)',ylabel='Local left (m)',aspect='equal');ax.legend(fontsize=8)
    ax=axes[1,1];candidate=result['localization']['registration']['candidate_pose'];world=transform(scan['points_local'],np.asarray(candidate))
    ax.imshow(saved['states']+1,origin='lower',extent=extent,cmap=ListedColormap(['#999999','white','#151515']),vmin=0,vmax=2)
    if len(world):ax.scatter(*world.T,s=8,c='orangered',label='ICP candidate projection (untrusted)')
    ax.set(title='Registration diagnosis; NOT a certified pose',xlabel='X (m)',ylabel='Y (m)',aspect='equal');ax.legend(fontsize=7)
    ax=axes[1,2];ax.plot(trace[:,0],trace[:,4],label='Actual stationary speed');ax.plot(trace[:,0],trace[:,7],'--',label='Target = 0')
    ax.set(title='Stationary observation only',xlabel='Time (s)',ylabel='Speed (m/s)');ax.legend(fontsize=8)
    ax=axes[2,0];ax.plot(trace[:,0],np.degrees(trace[:,5]));ax.set(title='Recorded steering restored via actuator',xlabel='Time (s)',ylabel='Actual front steer (deg)')
    axes[2,1].axis('off');axes[2,1].text(.05,.85,'Parking tracking errors: NOT AVAILABLE\n\nNo path certified\nNo forward / reverse drive executed\nNo final parking pose exists',va='top',transform=axes[2,1].transAxes)
    ax=axes[2,2];ax.plot(trace[:,0],trace[:,9]);ax.set(title='Stationary true-body / SLAM clearance (audit only)',xlabel='Time (s)',ylabel='Full-body clearance (m)')
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('P11.7.8 | '+result['status']+' | Full parking NOT executed',fontsize=14)
    fig.savefig(OUT/f'{PREFIX}_validation.png',dpi=140);plt.close(fig)


def supplement(result):
    v=result['localization']['sensor_visibility'];e=result['post_gate_evaluation_only']
    return f"""
## 视场与分时定位审计

目标中心相对里程计预测车头方位 {v['goal_center_relative_bearing_deg']:.3f}°，超出 ±{v['half_fov_deg']:.1f}° 水平视场。当前只有 {v['valid_returns']} 个有效回波，少于 ICP 所需 {v['minimum_correspondences']} 个匹配点，即使全部命中匹配也不足。配准拒绝发生在信息几何检测之前，不能把 degenerate=false 理解为场景不退化。

决策后的独立评估：里程计代理预测与记录的实际终点位置差 {e['odometry_prediction_vs_recorded_actual_position_m']:.9f} m、航向差 {e['heading_error_deg']:.9f}°；这不是可信误差上界，未反馈用于 ICP、规划或地图更新。历史 0.994 m 对应原探索末帧，不是当前 P11.7.6 终点；当前身体位姿接近预测也不能排除 SLAM 障碍物位置误差。

最终结果为安全门控拒绝，P11.7.8 的完整倒车泊车目标尚未达成。没有在失败之后扩大匹配阈值、切换 GT、重复物理实验或继续执行。
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gui',action='store_true');args=parser.parse_args()
    protected=[p for p in OUT.iterdir()if p.is_file()and not p.name.startswith('p11_7_8_')]
    protected += [p for folder in ('parking_planner','configs','scripts')for p in (ROOT/folder).rglob('*.py')if p.name!='run_p11_7_8_live_slam_reverse_parking.py']
    baseline={str(p.relative_to(ROOT)):sha(p)for p in protected}
    old,diagnostic,report,old_path,old_trace,columns,actual,input_audit=historical_inputs()
    saved,cfg,grid=live.load_live();scenario=live.build_perpendicular_parking_scenario();g=scenario['geometry']
    prediction,odom_audit=odometry_prediction(saved,old_trace,columns,g)
    blockers=goal_diagnostic(saved,cfg,grid,scenario)
    print('Actual terminal state:',actual,flush=True)
    print('Raw goal area counts:',blockers['raw_counts']['area_overlap'],flush=True)
    print('SLAM/odometry registration prior:',prediction.tolist(),flush=True)
    scan,trace,sensor_audit,frame=stationary_observation(actual,prediction,grid,scenario,args.gui)
    localization=localization_check(scan,prediction,saved,cfg,scenario)
    status='LOCALIZATION_UNCERTIFIED_GOAL_OCCUPIED_UNKNOWN'
    result=dict(stage='P11.7.8',status=status,complete_parking_pass=False,
        lifecycle=dict(APPROACH_COMPLETE=bool(old['status']=='APPROACH_VALIDATED_GOAL_BLOCKED'),
            APPROACH_COMPLETE_source='historical P11.7.6 only; not rerun',PARKING_EXECUTED=False,PARKING_VALIDATED=False,COMPLETE_PARKING_PASS=False),
        policy_declared_before_measurement=POLICY,actual_terminal_state=actual,historical_inputs=input_audit,
        map_file=str(live.MAP.relative_to(ROOT)),map_sha256=sha(live.MAP),original_goal=[scenario['goal'].x,scenario['goal'].y,scenario['goal'].yaw],
        planning_start=None,planning_start_reason='No certified current SLAM localization; actual physical terminal pose is NOT substituted.',
        odometry_prior=odom_audit,goal_blockers=blockers,localization=localization,stationary_sensor_observation=sensor_audit,
        hybrid_astar_executed=False,path_generated=False,continuous_swept_certificate_passed=False,
        physical_parking_execution=False,active_observation_motion_executed=False,map_updates=0,
        lidar_scan_to_map_used=True,occupied_labels_changed=0,unknown_forced_free=0,p10_fixed_trajectory_loaded=False,
        goal_replaced_with_waiting_pose=False,physical_parameters_changed=False,controller_changed=False,
        final_parking_metrics=dict(position_error_m=None,heading_error_deg=None,complete_body_inside_bay=None,
            collision_count=None,slam_violation_count=None,tracking_error_max_m=None,stopped_hold_2s=None),
        active_observation_decision='One stationary scan is safe and was captured; moving to view the bay is forbidden while localization has no credible bound. No scan is inserted into the map at an uncertified pose.',
        safety_certificate=dict(passed=False,execution_margin_m=.13,localization_body_margin_m=None,total_required_margin_m=None,
            reasons=localization['certificate_reasons']+['Original goal includes raw positive-area Occupied and Unknown cells. Aggregation-only reinterpretation cannot clear them.']),
        stopping_reason='Trustworthy localization bound unavailable; goal remains occupied/unknown. No autonomous drive, no path search to an unsafe goal, no map clearing.',
        protected_sha256=baseline)
    result['protected_files_unchanged']=all(sha(ROOT/p)==digest for p,digest in baseline.items());assert result['protected_files_unchanged']
    # Evaluation AFTER all stop decisions; never supplied to registration/map/planning.
    delta=prediction-np.array([actual[k]for k in ('rear_x','rear_y','yaw_rad')])
    result['post_gate_evaluation_only']=dict(odometry_prediction_vs_recorded_actual_position_m=float(np.linalg.norm(delta[:2])),
        heading_error_deg=abs(math.degrees(math.atan2(math.sin(delta[2]),math.cos(delta[2])))),
        historical_exploration_endpoint_discrepancy=old['localization_audit']['position_error_m'],
        interpretation='Agreement with a physical evaluation pose cannot certify map accuracy or justify using truth as localization. Historical 0.994m error refers to a different epoch; it is not assigned to this endpoint.')
    result['physical_observation_collision_samples']=int(np.sum(trace[:,8]>0))
    path=np.empty((0,6),float)
    np.savez_compressed(OUT/f'{PREFIX}_path.npz',path=path,columns=np.array(live.PATH_COLUMNS),
        x=path[:,0],y=path[:,1],yaw=path[:,2],steering=path[:,3],gear=path[:,4].astype(np.int8),curvature=path[:,5],
        goal=np.array(result['original_goal']),planning_start=np.empty(0),odometry_prediction=prediction,
        status=np.array(status),safety_certificate=np.array(json.dumps(clean(result['safety_certificate']))),map_sha256=np.array(result['map_sha256']))
    np.savez_compressed(OUT/f'{PREFIX}_trace.npz',trace=trace,columns=np.array(TRACE_COLUMNS),
        phase=np.full(len(trace),'STATIONARY_LOCALIZATION_OBSERVATION'),parking_trace=np.empty((0,len(TRACE_COLUMNS))),
        angles_rad=scan['angles_rad'],ranges_m=scan['ranges_m'],valid=scan['valid'],points_local=scan['points_local'],
        scan_timestamps_s=np.array([sensor_audit['scan_timestamp_s']]),odometry_prediction=prediction,
        untrusted_icp_candidate=np.asarray(localization['registration']['candidate_pose']),
        trusted_slam_poses=np.empty((0,3)),historical_actual_terminal=np.array([actual[k]for k in ('rear_x','rear_y','yaw_rad','actual_speed','actual_steer','gear')]),
        map_sha256=np.array(result['map_sha256']))
    render(result,saved,cfg,scenario,trace,scan,frame)
    write_json(OUT/f'{PREFIX}_metrics.json',result)
    icp=localization['registration']
    text=f'''# P11.7.8 Live SLAM → Robust Hybrid A* → Reverse Parking\n\n实际状态：**{status}**。没有执行倒车入库，不是 COMPLETE_PARKING_PASS。\n\n## 实际起点与记录语义\n\n读取 P11.7.6 最后有效 Trace 第 {input_audit['last_valid_index']} 行，后轴实际位置 ({actual['rear_x']:.9f}, {actual['rear_y']:.9f}) m，车身 yaw {math.degrees(actual['yaw_rad']):.9f}°，速度 {actual['actual_speed']:.9f} m/s，前轮平均转角 {math.degrees(actual['actual_steer']):.9f}°，档位 {actual['gear']:+.0f}。末两秒静止检查通过。没有将理论等待位姿作为起点。\n\n该物理位姿仅恢复一次独立传感器物理场景。SLAM 预测从原地图 estimated_poses[-1] 出发，仅用旧 Trace 的有符号速度、前轮转角与时间，通过现有 AckermannModel.step 积分；未向配准输入物理世界 x/y/yaw，也未读取 reference_poses。速度记录是模拟里程计代理，平均前轮角不是严格等效自行车角，缺少编码器增量和误差校准，因此这不是有已知上界的定位。\n\n## 定位门控和当前 LiDAR\n\n只采集一次真实 PyBullet Ray Casting 扫描，180 线、270°、12 m、1.5 cm 距离噪声，固定种子沿用现有传感器。扫描命中数 {len(scan['points_local'])}；复用现有 point_to_line_icp 和原 PointToLineConfig，配准接受 {icp['success']}，退化 {icp['degenerate']}，拒绝原因 {icp['rejection_reason']}，匹配点 {icp['correspondences']}，比例 {icp['fitness']:.6f}，残差 {icp['line_rmse_m']} m，信息特征值比 {icp['information_eigen_ratio']}。没有放宽 ICP 阈值。\n\n不能将 ICP 收敛、残差、Gaussian 先验或信息矩阵自动当成绝对定位误差上界。历史约 0.994377 m / 4.391258° 是上一探索终点的评估差异，不是当前车位等待终点的已知误差，更不是可直接用于修正的刚体变换。原地图没有保存局部形变/世界锚定误差证书；当前单次配准无法排除系统地图误差。0.13 m 只覆盖已测物理跟踪误差。定位包络保持 null，禁止把 null 当作 0。\n\n## 原始目标阻塞分类\n\n目标仍为 {result['original_goal']}，车辆为 4.5×1.8 m、轴距 2.7 m，后悬沿用 1.0 m。目标 footprint 的原始 0.05 m 栅格正面积重叠：\n\n```json\n{json.dumps(blockers['raw_counts']['area_overlap'],indent=2)}\n```\n\nA：原始 Occupied 确实存在，但不能仅凭标签断言是真实物理障碍，当前定位不可信时无法确认其物理来源，全部保留。B：Unknown 中包括已观测但概率仍不确定的单元，不能直接视为未扫描。C：统计仅由重叠部分外证据传播的正面积聚合阻塞，数量 {len(blockers['aggregation_only_cells'])}，但不能解释或移除原始正面积 Occupied。D：未发现数组分类、坐标轴、原点、分辨率及聚合一致性异常。每个阻塞单元的坐标、面积、概率与原始证据计数见 JSON。\n\n完整目标车身原规划器碰撞：{blockers['nominal_planner_collision']}；精确矩形碰撞：{blockers['exact_body_collision']}；加 0.13 m 执行包络仍阻塞：{blockers['execution_envelope_blocked']}。车位入口、左右边界及内部区域也分别记录。\n\n## 主动观测与安全停止\n\n已在历史安全终点进行静止观测，全部驱动目标速度为零，未进行前进或倒车。没有通过 reset 位姿或外力制造运动。局部扫描未经可信定位，不能调用 SLAM 更新将证据投射进地图，因此地图更新次数 0。进一步改变观测位姿需要先通过定位及鲁棒路径门控；本次没有证书，禁止为了看清车位而盲目移动。\n\n根据 2.6 m 车位宽和 1.8 m 车宽，名义左右余量 0.40 m；扣除 0.13 m 跟踪包络，留给定位的必要预算至多为 0.27 m，且尚未考虑占据栅格和转弯扫掠。当前没有可信误差上界，不得声称满足此预算。\n\n## 完整入库验收\n\n未通过：定位安全证书失败，同时原目标仍被原始 Occupied/Unknown 阻塞。Hybrid A* 未执行，路径为空，连续扫掠证书未通过，物理泊车未执行。APPROACH_COMPLETE 只引用历史 P11.7.6；PARKING_EXECUTED、PARKING_VALIDATED、COMPLETE_PARKING_PASS 均为 false。没有以等待位置替代停车目标。\n\n停车精度在本次观测前声明为 0.10 m / 2°，完整车身必须位于原车位内、末段必须倒车、零碰撞、零地图违规、定位加跟踪包络全程有效，并以不超过 0.01 m/s 保持 2 秒。由于泊车未执行，最终停车误差、碰撞次数、跟踪误差保持 null，不将静止观测的零碰撞伪装为泊车通过。\n\nNPZ trace 是 2 秒静止传感器会话；parking_trace 为零行。图片清楚标明未执行部分，不伪造规划线或跟踪误差曲线。地图、旧路径及旧物理执行全部只读，历史文件与旧源码 SHA256 保护检查：{result['protected_files_unchanged']}。\n\n## 后续所需证据（本轮不擅自修改）\n\n需要可验证的当前定位/地图几何误差界，以及可信定位下对原始阻塞单元的传感器证据；当前数据无法支持安全运动。不能通过修改概率阈值、强制 Unknown→Free、删除 Occupied、GT 纠偏或重放固定路径解决。\n\n运行：\n```powershell\nconda run --no-capture-output -n parkingplanner python scripts/run_p11_7_8_live_slam_reverse_parking.py --gui\n```\n此命令重新执行本阶段必要的静止观测和安全门控，不重跑历史实验。\n'''
    text+=supplement(result)
    (OUT/f'{PREFIX}_report.md').write_text(text,encoding='utf-8')
    print(json.dumps(clean(dict(status=status,registration=icp,physical_parking_execution=False,historical_unchanged=result['protected_files_unchanged'])),indent=2),flush=True)


if __name__=='__main__':main()
