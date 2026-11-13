"""P11.7.8 final dual-LiDAR localization / gated live-map parking pipeline.

No historical result is overwritten. Simulator truth is acquisition/evaluation
only. An unsupported systematic-error bound cannot be replaced with ICP RMSE.
"""
import argparse
from dataclasses import asdict
import json,math,sys
from pathlib import Path
from statistics import NormalDist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import run_p11_7_8_live_slam_reverse_parking as prior
import run_p11_7_6_robust_hybrid_astar_physical_validation as robust
from parking_planner.sensing.dual_lidar import DualLidar360
from parking_planner.sensing.encoder_imu import EncoderIMU
from parking_planner.localization.correlative_scan_matcher import correlative_match
from parking_planner.localization.point_to_line_icp import PointToLineConfig,point_to_line_icp,estimate_normals,transform
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper
from parking_planner.planning.robust_clearance import CellClearance,CertifiedRobustGrid
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import resample_path
from parking_planner.vehicle.ackermann import VehicleState

OUT=ROOT/'results';PREFIX='p11_7_8_final'
LIMITS=dict(position_m=.10,heading_deg=2.,stop_speed_mps=.01,hold_s=2.,execution_margin_m=.13,
    reverse_speed_ceiling_mps=.12,forward_speed_ceiling_mps=.28,confidence_failure_probability=1e-6,
    require_independent_map_and_pose_bounds=True,require_actual_last_gear_reverse=True)
COLUMNS=['time_s','rear_x_eval','rear_y_eval','yaw_eval','speed_eval','steer_eval','gear','encoder_speed','imu_heading','imu_gyro','command_speed','contacts_eval']


def acquire(actual,scenario,gui):
    import pybullet as p
    from parking_planner.simulation.pybullet.physics_vehicle import PhysicsVehicle
    from parking_planner.simulation.pybullet.scene import ParkingScene
    from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS
    cid=p.connect(p.GUI if gui else p.DIRECT,options=WINDOW_OPTIONS if gui else '')
    originals={};forbidden=[];packets=[];rows=[];sensors=[];frames=[];eval_scan_poses=[]
    try:
        p.setTimeStep(1/240,physicsClientId=cid);p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=cid)
        scene=ParkingScene(cid,scenario)
        car=PhysicsVehicle(cid,scenario['geometry'],[actual[k]for k in ('rear_x','rear_y','yaw_rad')]);car.direction=int(actual['gear'])
        for name in ('resetBasePositionAndOrientation','resetBaseVelocity','resetJointState','applyExternalForce','applyExternalTorque'):
            fn=getattr(p,name);originals[name]=fn
            def guard(*a,_fn=fn,_name=name,**kw):
                body=a[0]if a else kw.get('bodyUniqueId',kw.get('objectUniqueId'))
                if body==car.body:forbidden.append(_name);raise RuntimeError('Forbidden vehicle motion')
                return _fn(*a,**kw)
            setattr(p,name,guard)
        if gui:scene.configure_gui()
        dual=DualLidar360(cid,car.body);encoder=EncoderIMU(car)
        for step in range(673):
            measurement=encoder.sample(1/240)
            # Isolated acquisition and evaluation: these values never enter matching.
            actual_state=car.state()
            contacts=[c for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid)if c[2]in scene.obstacle_ids and(c[8]<=0 or c[9]>0)]
            rows.append([step/240,*[actual_state[k]for k in ('x','y','yaw','speed','steering','direction')],measurement['speed_mps'],measurement['heading_rad'],measurement['gyro_rad_s'],0.,len(contacts)])
            sensors.append([measurement['forward_increment_m'],*measurement['wheel_angles_rad'],*measurement['wheel_speed_mps'],measurement['heading_rad'],measurement['gyro_rad_s'],measurement['steering_rad']])
            if contacts:raise RuntimeError('Contact during stationary sensing')
            if step in(480,576,672):
                packets.append(dual.capture(actual_state));eval_scan_poses.append([actual_state[k]for k in ('x','y','yaw')])
            if step<672:car.command(0.,actual['actual_steer'],1/240);p.stepSimulation(physicsClientId=cid)
        view=p.computeViewMatrixFromYawPitchRoll([10.3,12.0,0],17,35,-62,0,2);proj=p.computeProjectionMatrixFOV(55,1.375,.1,100)
        image=np.asarray(p.getCameraImage(1100,800,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER,physicsClientId=cid)[2]).reshape(800,1100,4)
        info=dict(sensor_mounts_rear_axle_frame=dual.mounts.tolist(),sensor_config=asdict(dual.config),
            encoder_ticks_per_revolution=4096,encoder_distance_quantization_m=encoder.encoder_tick_rad*car.config.wheel_radius_m,
            imu_heading_noise_bound_deg=.1,imu_gyro_noise_bound_deg_s=.02,
            wheel_slip_bound_m=None,initial_absolute_position_bound_m=None,
            self_hits_removed=[packet['self_hits']for packet in packets],post_initialization_pose_reset_count=len(forbidden),
            sensor_sessions=1,dual_scan_frames=len(packets),drive_commands_all_zero=True,
            information_boundary='World pose is confined to simulated ray acquisition and offline evaluation. Localization receives local points, encoder increments, IMU, original estimated poses and occupied cells only.',
            repeated_time_samples_are_not_independent_spatial_views=True,two_sensor_baselines_m=.6)
        return packets,np.array(rows),np.array(sensors),info,image,np.array(eval_scan_poses)
    finally:
        for name,fn in originals.items():setattr(p,name,fn)
        if p.isConnected(cid):p.disconnect(cid)


def register(packets,prediction,saved,cfg,imu_heading):
    rc=np.column_stack(np.where(saved['states']==1));map_points=np.column_stack((cfg['origin_x']+(rc[:,1]+.5)*cfg['resolution'],cfg['origin_y']+(rc[:,0]+.5)*cfg['resolution']))
    config=PointToLineConfig();surface=estimate_normals(map_points,config);results=[]
    for i,packet in enumerate(packets):
        coarse=correlative_match(packet['points_local'],prediction,saved['states'],cfg)
        seed=np.asarray(coarse.get('pose',prediction),float)
        match=point_to_line_icp(packet['points_local'],seed,map_points,config,surface=surface)
        gap=abs(math.degrees(math.atan2(math.sin(match['candidate_pose'][2]-imu_heading[i]),math.cos(match['candidate_pose'][2]-imu_heading[i]))))
        entry=dict(scan_index=i,forward_hits=int(packet['raw'][0]['valid'].sum()),rear_hits=int(packet['raw'][1]['valid'].sum()),
            fused_hits=len(packet['points_local']),coarse=coarse,fine=match,imu_heading_rad=float(imu_heading[i]),
            registration_imu_yaw_disagreement_deg=gap)
        results.append(entry)
        print(f'Dual frame {i}: forward={entry["forward_hits"]} rear={entry["rear_hits"]}, ICP={match["success"]}, reason={match["rejection_reason"]}, matches={match["correspondences"]}, IMU gap={gap:.3f}deg',flush=True)
    return results,config


def error_certificate(registrations,saved,cfg,sensor_info):
    # Measurable sensor terms are distinct from unsupported initial/slip/map errors.
    n=3*2*180;alpha=LIMITS['confidence_failure_probability'];z=NormalDist().inv_cdf(1-alpha/(2*n))
    range_envelope=z*.015;quantization=cfg['resolution']/math.sqrt(2)
    reasons=[]
    if not all(r['fine']['success']for r in registrations):reasons.append('EXISTING_POINT_TO_LINE_ICP_GATE_REJECTED')
    if any(r['registration_imu_yaw_disagreement_deg']>sensor_info['imu_heading_noise_bound_deg']for r in registrations):
        reasons.append('MAP_ALIGNMENT_AND_ABSOLUTE_IMU_HEADING_INCONSISTENT_WITH_SENSOR_NOISE_ONLY')
    reasons.extend(['INITIAL_ABSOLUTE_POSITION_ERROR_BOUND_NOT_AVAILABLE','WHEEL_SLIP_ERROR_BOUND_NOT_AVAILABLE',
        'HISTORICAL_MAP_SYSTEMATIC_DEFORMATION_BOUND_NOT_AVAILABLE'])
    # These are not zero, not ICP covariance, and not estimated from truth.
    return dict(passed=not reasons,rejection_reasons=reasons,
        pose_translation_bound_m=None,pose_heading_bound_deg=None,map_systematic_error_bound_m=None,
        localization_body_displacement_bound_m=None,map_geometry_bound_m=None,
        bounded_encoder_quantization_m=sensor_info['encoder_distance_quantization_m'],
        bounded_imu_heading_noise_deg=.1,bounded_imu_gyro_noise_deg_s=.02,
        gaussian_range_envelope_m=range_envelope,range_envelope_confidence=1-alpha,
        range_envelope_definition=f'Union bound over {n} Gaussian range errors, z={z}; probabilistic, NOT a deterministic hard bound.',
        cell_center_quantization_radius_m=quantization,
        execution_envelope_m=.13,total_planning_margin_m=None,
        explanation='Independent pose and map uncertainties must both be supported. Small residual, optimizer convergence, temporal repeatability and sensor-noise bounds alone do not bound shared map deformation or global translation. No truth-derived bound is inserted.')


def plan_certified(mapper,pose,scenario,certificate):
    """Existing Hybrid A*, stricter collision layer, continuous sweep certificate."""
    if not certificate['passed']:return np.empty((0,6)),dict(executed=False,reason='LOCALIZATION_OR_MAP_CERTIFICATE_FAILED')
    cfg=mapper.config;states=mapper.states;grid=prior.live.make_planner_grid(states,cfg)
    margin=.13+certificate['localization_body_displacement_bound_m']+certificate['map_geometry_bound_m']
    collision=CertifiedRobustGrid(grid,scenario['geometry'],margin,scenario['planner_config'],cfg['resolution'])
    planner=HybridAStarPlanner(grid=collision,geometry=scenario['geometry'],config=scenario['planner_config'])
    goal=scenario['goal'];start=VehicleState(*pose)
    if collision.is_collision(goal,planner.local_x,planner.local_y):return np.empty((0,6)),dict(executed=False,reason='ORIGINAL_GOAL_ROBUST_FOOTPRINT_BLOCKED')
    raw=planner.plan(start,goal)
    if not raw:return np.empty((0,6)),dict(executed=True,success=False,reason='NO_ROBUST_PATH')
    dense=resample_path(raw,spacing=.025)
    path=np.array([[p.x,p.y,p.yaw,p.steering,p.direction,math.tan(p.steering)/scenario['geometry'].wheelbase]for p in dense])
    lookup=robust.indexes(dict(states=states),cfg,grid,scenario['geometry'])
    sweep=robust.path_clearance(path,lookup,margin)
    valid=sweep['all_continuous_segments_robust']and path[-1,4]==-1
    return path,dict(executed=True,success=bool(valid),continuous_safety=sweep,margin_m=margin,
        final_segment_reverse=bool(path[-1,4]==-1),reason=None if valid else'FINAL_REVERSE_OR_CONTINUOUS_SAFETY_FAILED')


def plot(result,saved,cfg,scenario,packets,trace,frame):
    fig,axes=plt.subplots(3,3,figsize=(18,14),layout='constrained');extent=[0,22,0,22]
    for ax in (axes[0,0],axes[0,1],axes[1,1]):
        ax.imshow(saved['states']+1,origin='lower',extent=extent,interpolation='nearest',cmap=ListedColormap(['#999999','white','#111111']),vmin=0,vmax=2)
        ax.set(xlabel='X (m)',ylabel='Y (m)',aspect='equal')
    goal=np.array(result['goal']);g=scenario['geometry'];index=CellClearance(np.zeros((1,1),bool),.05,[0,0],g)
    for ax in axes[0,:2]:
        ax.plot(*np.vstack([index.polygon(goal),index.polygon(goal)[0]]).T,c='red',label='Full goal body')
        ax.plot(*robust.envelope_outline(goal,g,.13).T,c='purple',label='0.13 m execution envelope')
        x0,y0,x1,y1=scenario['parking_slot'];ax.plot([x0,x0,x1,x1],[y0,y1,y1,y0],c='goldenrod')
    axes[0,0].set_title('Original SLAM retained; no certified parking path')
    axes[0,1].set(xlim=(8.5,12.5),ylim=(10.5,17.5),title='Original bay: unresolved occupied / unknown')
    axes[0,2].imshow(frame);axes[0,2].axis('off');axes[0,2].set_title('Current physical terminal / stationary sensors')
    ax=axes[1,0]
    for i,color in [(0,'royalblue'),(1,'darkorange')]:
        p=packets[0];pts=p['ray_endpoints_local'][p['valid']&(p['sensor_index']==i)]
        if len(pts):ax.scatter(*pts.T,c=color,s=12,label='Front'if i==0 else'Rear')
    ax.set(title='Front + rear returns in ONE vehicle frame',xlabel='Forward (m)',ylabel='Left (m)',aspect='equal');ax.legend()
    reg=result['registration'][0];world=transform(packets[0]['points_local'],np.asarray(reg['fine']['candidate_pose']))
    axes[1,1].scatter(*world.T,s=6,c='red');axes[1,1].set_title('Coarse → fine map alignment (not certified)')
    axes[1,2].plot(trace[:,0],trace[:,4],label='Physical speed (evaluation)');axes[1,2].plot(trace[:,0],trace[:,10],'--',label='Command = 0')
    axes[1,2].set(xlabel='Time (s)',ylabel='m/s',title='No parking drive executed');axes[1,2].legend(fontsize=8)
    axes[2,0].plot(trace[:,0],np.degrees(trace[:,5]));axes[2,0].set(xlabel='Time (s)',ylabel='Steer (deg)',title='Real steering actuator / stationary capture')
    axes[2,1].plot(trace[:,0],np.degrees(trace[:,8]),label='IMU heading');axes[2,1].set(xlabel='Time (s)',ylabel='IMU heading (deg)',title='IMU independent of map registration')
    axes[2,2].axis('off');axes[2,2].text(.02,.9,'Parking NOT executed\n\nNo certified pose/map error bound\nNo certified full path\nNo parking tracking errors or final pose\n\nFailed samples retained; no PASS claim',va='top',transform=axes[2,2].transAxes)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('P11.7.8 FINAL | '+result['status']);fig.savefig(OUT/f'{PREFIX}_parking_validation.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gui',action='store_true');args=parser.parse_args()
    prefix_files=list(OUT.glob(PREFIX+'*'))
    if prefix_files:raise FileExistsError('Final experiment outputs already exist; refusing to overwrite: '+str(prefix_files[0]))
    protected=[p for p in OUT.iterdir()if p.is_file()]
    protected += [p for folder in ('parking_planner','configs','scripts')for p in (ROOT/folder).rglob('*.py')if p.name not in('run_p11_7_8_final_parking.py','dual_lidar.py','encoder_imu.py','correlative_scan_matcher.py')]
    hashes={str(p.relative_to(ROOT)):prior.sha(p)for p in protected}
    old,diag,report,old_path,old_trace,columns,actual,input_audit=prior.historical_inputs()
    # Read the previous failed implementation and report, not rerun them.
    failed_report=(OUT/'p11_7_8_parking_report.md').read_text(encoding='utf-8')
    saved,cfg,grid=prior.live.load_live();scenario=prior.live.build_perpendicular_parking_scenario()
    prediction,odom=prior.odometry_prediction(saved,old_trace,columns,scenario['geometry'])
    original=prior.goal_diagnostic(saved,cfg,grid,scenario)
    packets,trace,sensors,sensor_info,frame,eval_poses=acquire(actual,scenario,args.gui)
    registrations,icp_cfg=register(packets,prediction,saved,cfg,trace[[480,576,672],8])
    certificate=error_certificate(registrations,saved,cfg,sensor_info)
    mapper=OccupancyGridMapper(**cfg);mapper.log_odds[:]=saved['log_odds'];mapper.observed[:]=saved['observed']
    # No uncertain projection is allowed to modify even the separate working copy.
    map_updates=0;planning_pose=None
    if certificate['passed']:
        planning_pose=np.asarray(registrations[-1]['fine']['pose'])
        raise RuntimeError('This archive has no independent map-error certificate; unexpected gate transition')
    path,planning=plan_certified(mapper,planning_pose,scenario,certificate)
    status='LOCALIZATION_MAP_UNCERTIFIED_GOAL_BLOCKED'
    # Ground-truth comparisons BEGIN HERE, after all online decisions are frozen.
    evaluation=[]
    for reg,truth in zip(registrations,eval_poses):
        candidate=np.asarray(reg['fine']['candidate_pose']);d=candidate-truth
        evaluation.append(dict(position_error_m=float(np.linalg.norm(d[:2])),heading_error_deg=abs(math.degrees(math.atan2(math.sin(d[2]),math.cos(d[2]))))))
    rc=np.column_stack(np.where(saved['states']==1));points=np.column_stack((cfg['origin_x']+(rc[:,1]+.5)*cfg['resolution'],cfg['origin_y']+(rc[:,0]+.5)*cfg['resolution']))
    from run_p11_7_3_goal_blocker_source_analysis import boundary_distance
    surface_error=boundary_distance(points,scenario['parked_vehicles'])
    localized=dict(status=status,prediction=prediction,odometry_proxy=odom,registration=registrations,icp_configuration=asdict(icp_cfg),
        error_certificate=certificate,sensor_info=sensor_info,offline_pose_error_evaluation=evaluation,
        offline_map_error_evaluation=dict(definition='Original Occupied cell-center distance to physical obstacle boundaries; EVALUATION ONLY; not inserted into online certificate.',
            mean_m=float(surface_error.mean()),p95_m=float(np.quantile(surface_error,.95)),maximum_m=float(surface_error.max())),
        truth_used_in_localization=False,truth_used_in_map_update=False,truth_used_in_planning=False)
    result=dict(stage='P11.7.8 final',status=status,COMPLETE_PARKING_PASS=False,thresholds_predeclared=LIMITS,
        actual_terminal_state=actual,goal=[scenario['goal'].x,scenario['goal'].y,scenario['goal'].yaw],planning_start=None,
        registration=registrations,error_certificate=certificate,goal_before=original,goal_after=original,
        sensor_info=sensor_info,map_updates=map_updates,map_cells_changed=0,
        planning=planning,physical_parking_executed=False,physical_parking_collision_count=None,
        observation_physical_contact_samples=int(trace[:,11].sum()),final_position_error_m=None,final_heading_error_deg=None,
        complete_body_inside_slot=None,final_reverse_gear=None,parking_stopped_hold_2s=None,
        encoder_imu_control_interface_implemented=True,actual_pose_supplied_to_controller=False,
        historical_inputs=input_audit,protected_sha256=hashes,
        independent_error_evaluation=localized['offline_map_error_evaluation'],
        stopping_reason='Dual perception and coarse-to-fine registration attempted. Independent global pose/map-error bounds unavailable; IMU/map consistency and original goal clearance not certified. No motion beyond stationary sensor acquisition.')
    result['protected_files_unchanged']=all(prior.sha(ROOT/p)==digest for p,digest in hashes.items());assert result['protected_files_unchanged']
    prior.write_json(OUT/f'{PREFIX}_localization.json',localized)
    np.savez_compressed(OUT/f'{PREFIX}_dual_lidar.npz',
        raw_angles_rad=np.array([[p['raw'][i]['angles_rad']for i in range(2)]for p in packets]),
        raw_ranges_m=np.array([[p['raw'][i]['ranges_m']for i in range(2)]for p in packets]),
        raw_valid=np.array([[p['raw'][i]['valid']for i in range(2)]for p in packets]),
        ray_origins_local=np.array([p['ray_origins_local']for p in packets]),ray_directions_local=np.array([p['ray_directions_local']for p in packets]),
        ray_endpoints_local=np.array([p['ray_endpoints_local']for p in packets]),valid=np.array([p['valid']for p in packets]),
        fused_angles_rad=np.array([p['angles_rad']for p in packets]),raw_self_hits=np.array([[p['raw'][i]['self_hits']for i in range(2)]for p in packets]),sensor_index=packets[0]['sensor_index'],mounts=packets[0]['mounts'],sensor_names=np.array(['front','rear']),timestamps_s=np.array([2.,2.4,2.8]),
        largest_angular_gap_deg=np.array([p['largest_angular_gap_deg']for p in packets]),scene_rgba=frame)
    np.savez_compressed(OUT/f'{PREFIX}_live_map.npz',config=np.array(json.dumps(cfg)),states=mapper.states,log_odds=mapper.log_odds,observed=mapper.observed,probabilities=mapper.probabilities,
        source_sha256=np.array(prior.sha(prior.live.MAP)),updated=np.array(False),update_count=np.array(map_updates),status=np.array(status))
    np.savez_compressed(OUT/f'{PREFIX}_hybrid_astar_path.npz',path=path,columns=np.array(prior.live.PATH_COLUMNS),goal=np.array(result['goal']),planning_start=np.empty(0),certificate=np.array(json.dumps(robust.clean(certificate))),status=np.array(status))
    np.savez_compressed(OUT/f'{PREFIX}_physical_trace.npz',trace=trace,columns=np.array(COLUMNS),phase=np.full(len(trace),'STATIONARY_DUAL_SENSOR_CAPTURE'),
        parking_trace=np.empty((0,len(COLUMNS))),encoder_imu=sensors,encoder_imu_columns=np.array(['forward_increment','left_angle','right_angle','left_speed','right_speed','imu_heading','imu_gyro','steer']),
        actual_scan_poses_evaluation_only=eval_poses,planned_path=path,localization_candidates=np.array([r['fine']['candidate_pose']for r in registrations]))
    prior.write_json(OUT/f'{PREFIX}_parking_metrics.json',result);plot(result,saved,cfg,scenario,packets,trace,frame)
    report_text=f'''# P11.7.8 最终自主泊车闭环\n\n状态：**{status}**。本次完整倒车入库验收未通过，未输出 COMPLETE_PARKING_PASS。\n\n## 已实现并实际运行\n\n复用原前向 LiDAR，新增后向 180 线、270°、12 m、0.015 m 噪声 LiDAR。安装位置为后轴坐标 (0,0,0°) 与 (-0.6,0,180°)，高度 1.25 m；点云转入统一后轴车体坐标，原始前后扫描分别完整保留。融合保留每条射线独立起点，禁止把后传感器距离错误当成后轴极坐标做清空；地图更新适配器调用原 inverse sensor model 两次。原传感器去掉自身物体命中；若有自命中导致无回波歧义，该传感器所有无效束不用于清空。\n\n新增真实关节轮速/4096 tick 编码器、模拟 IMU 航向和角速度接口，定位接口不暴露世界 XY。IMU 噪声有明确界 ±0.1°、角速度 ±0.02°/s；编码器量化有界，但轮胎打滑误差和历史初始绝对位置误差没有独立界。后者不能用量化噪声掩盖。\n\n只做一次静止采集会话，3 帧前后扫描；前后安装基线 0.6 m 是两个传感器视角，三次同位置扫描不是三个独立空间视角。对每帧实际执行原占据图上的 Correlative Scan Matching（±1.5 m / ±10°，粗 0.1 m / 1°，细 0.025 m / 0.25°），再用既有 PointToLine ICP 精配准；min_correspondences=30、ratio=0.3、max correspondence=0.35 m、line RMSE=0.08 m、eigen ratio=0.005 完全未改。\n\n## 实际定位结果\n\n| 帧 | 前/后回波 | ICP 接受 | 匹配数 | 线残差 m | 与 IMU 航向差 ° |\n|---|---|---|---:|---:|---:|\n'''
    for r in registrations:
        q=r['fine'];report_text+=f"| {r['scan_index']} | {r['forward_hits']}/{r['rear_hits']} | {q['success']} | {q['correspondences']} | {q['line_rmse_m']} | {r['registration_imu_yaw_disagreement_deg']:.6f} |\n"
    report_text+=f'''\n独立定位与地图证书：\n```json\n{json.dumps(robust.clean(certificate),indent=2)}\n```\n\nGaussian 距离噪声仅能给指定置信度下的联合范围，不是确定硬界；栅格中心量化半径单独计算。没有把 RMSE、优化协方差或重复性当作全局定位/系统地图误差界。不存在可信的世界锚定/旧地图局部形变界，不能安全执行地图修复与泊车。\n\n## 地图更新前后\n\n原目标 footprint 有正面积 Occupied={original['raw_counts']['area_overlap']['occupied']}、Unknown={original['raw_counts']['area_overlap']['unknown']}；保留完整 4.5×1.8 m 车身、2.7 m 轴距及原目标。新地图单独保存但因定位门控失败，实际更新 0 次，改变 0 格。没有删除 Occupied、强制 Unknown→Free、改阈值、使用物理真值投影修改地图或读取 P10 固定路径。也不能将一个地图坐标中的占据标签直接断言为真实物理障碍。\n\n多视角运动必须先有可信定位、地图和连续扫掠证书，当前均不足，因此未为了补观测而盲目前进。原地图的坐标原点/轴向/分类一致性检查仍通过；粗精匹配与 IMU 结果详见独立 localization JSON。\n\n## 路径、物理执行与验收\n\nHybrid A* 安全适配器复用原算法及完整 footprint，安全裕度应为 0.13 m + 定位包络 + 地图误差包络；任何未知项不得按 0 处理。实现了已认证输入下的原目标安全门控、重规划和连续扫掠检查入口，但当前无合法规划输入，未执行搜索。**物理泊车控制、前进/倒车段和换挡过程没有运行；本轮没有验证完整闭环功能。** 不将静止采样当作物理泊车。\n\n位置 ≤0.10 m、航向 ≤2°、零碰撞、零地图违规、完整车身在位、末段倒挡、速度 ≤0.01 m/s 并保持 2 秒的标准执行前已声明。未执行项目均为 null/false，不计 PASS。图中包含真实静止速度、实际转向和 IMU；停车跟踪误差、路径与最终泊车误差不存在，因此不伪造曲线。\n\n## 冻结决策后的独立评估\n\n定位真值误差：{json.dumps(evaluation)}\n\n原 Occupied 地图点到真实物理障碍边界的评估：{json.dumps(localized['offline_map_error_evaluation'])}。这不是定位/地图更新/规划输入，也不能事后作为在线证书。本次未发现并证明一个阻止入库的真实物理障碍，而是定位和地图证书不足，不能声称车位真实不可用。\n\n历史及旧源码 SHA256 检查：{result['protected_files_unchanged']}。所有历史结果只读；本轮只有新的静止双传感器采集，没有重跑旧实验或失败后反复物理重试。输出 NPZ 使用非 object 数组，可 allow_pickle=False 读取。\n\n运行入口 `scripts/run_p11_7_8_final_parking.py --gui`；已存在本阶段结果时主动拒绝覆盖。最终状态为拒绝自主运动，完整自主泊车目标尚未实现。\n'''
    (OUT/f'{PREFIX}_report.md').write_text(report_text,encoding='utf-8')
    print('Final status:',status,'; historical files unchanged:',result['protected_files_unchanged'],flush=True)


if __name__=='__main__':main()
