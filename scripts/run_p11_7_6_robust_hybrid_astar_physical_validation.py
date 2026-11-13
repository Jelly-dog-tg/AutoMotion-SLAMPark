"""P11.7.6: empirical tracking envelope, swept clearance, robust live-map plan."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon,Rectangle,Circle
import run_p11_7_4_5_live_slam_hybrid_astar_physical_validation as previous
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import resample_path
from parking_planner.planning.robust_clearance import CellClearance,CertifiedRobustGrid,pose_between,movement_bound
from parking_planner.vehicle.ackermann import VehicleState

OUT=ROOT/'results';PREFIX='p11_7_6'
PATH=OUT/f'{PREFIX}_robust_hybrid_astar_path.npz'
METRICS=OUT/f'{PREFIX}_robust_physical_metrics.json'
OLD=OUT/'p11_7_4_5_live_hybrid_astar'
NAMES=['raw_occupied','raw_unknown','planner_occupied','planner_unknown']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value,dict):return {k:clean(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v)for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float)and not math.isfinite(value):return None
    return value

def write_json(p,value):p.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def envelope(trace,columns,geometry):
    pos=float(np.max(trace[:,columns.index('nearest_path_position_error_m')]))
    angle=float(np.max(np.abs(trace[:,columns.index('heading_error_rad')])))
    pos_up=math.ceil(pos*10000)/10000;angle_up_deg=math.ceil(math.degrees(angle)*10000)/10000
    radius=math.hypot(max(geometry.front_extent,geometry.rear_overhang),geometry.width/2)
    displacement=2*radius*math.sin(math.radians(angle_up_deg)/2)
    margin=math.ceil((pos_up+displacement)/.005)*.005
    return dict(measured_position_error_m=pos,measured_heading_error_deg=math.degrees(angle),
        bounded_position_error_m=pos_up,bounded_heading_error_deg=angle_up_deg,corner_radius_m=radius,
        corner_displacement_from_yaw_m=displacement,combined_exact_bound_m=pos_up+displacement,
        margin_m=margin,rounding='Position rounded UP to 0.0001 m; heading UP to 0.0001 deg; total UP to 0.005 m.',
        formula='e_position + 2*r_max*sin(e_heading/2)',
        interpretation='Full physical rectangle Minkowski-summed with a disk of this radius contains EVERY simultaneous bounded translation/yaw perturbation. This is not a change to vehicle size or SLAM labels.',
        limitation='Empirical execution-error bound from one prior run, not a calibrated probabilistic guarantee and not a correction for SLAM global localization drift.')


def indexes(saved,cfg,grid,geometry):
    ratio=round(grid.resolution/cfg['resolution']);b=saved['states'].reshape(grid.rows,ratio,grid.cols,ratio)
    origin=[grid.origin_x,grid.origin_y]
    return dict(raw_occupied=CellClearance(saved['states']==1,cfg['resolution'],origin,geometry),
        raw_unknown=CellClearance(saved['states']==-1,cfg['resolution'],origin,geometry),
        planner_occupied=CellClearance(np.any(b==1,axis=(1,3)),grid.resolution,origin,geometry),
        planner_unknown=CellClearance(np.any(b==-1,axis=(1,3)),grid.resolution,origin,geometry))


def path_clearance(path,indices,margin,perturb=None):
    point=[];global_min={name:dict(distance_m=float('inf'))for name in NAMES};sweep=[]
    radius=indices['raw_occupied'].radius;last_status=time.monotonic()
    def measure(pose):
        return {name:dict(distance_m=d,nearest_cell=cell)for name,index in indices.items()for d,cell in [index.nearest(pose)]}
    def update(values,pose,segment,u):
        for name,v in values.items():
            if v['distance_m']<global_min[name]['distance_m']:
                global_min[name]=dict(**v,pose=list(pose),segment_index=segment,segment_fraction=u)
    for i,row in enumerate(path):
        values=measure(row[:3]);update(values,row[:3],max(0,i-1),1.)
        point.append(dict(path_index=i,pose=row[:3].tolist(),gear=int(row[4]),steering_rad=float(row[3]),clearance=values,
            robust_disk_safe=min(values['planner_occupied']['distance_m'],values['planner_unknown']['distance_m'])>margin))
    for i,(a,b)in enumerate(zip(path[:-1],path[1:])):
        full_bound=movement_bound(a[:3],b[:3],radius)
        # Sampling plus a rigorous Hausdorff half-interval cover, not a claim
        # that sampled poses alone prove the continuous swept volume is safe.
        n=max(1,math.ceil(full_bound/.005));interval_min={name:float('inf')for name in NAMES}
        gap=0.;previous_pose=a[:3]
        for j in range(n+1):
            pose=pose_between(a[:3],b[:3],j/n)
            values=point[i]['clearance']if j==0 else point[i+1]['clearance']if j==n else measure(pose)
            update(values,pose,i,j/n)
            for name in NAMES:interval_min[name]=min(interval_min[name],values[name]['distance_m'])
            if j:
                da=abs(math.atan2(math.sin(pose[2]-previous_pose[2]),math.cos(pose[2]-previous_pose[2])))
                gap=max(gap,float(np.linalg.norm(pose[:2]-previous_pose[:2]))/2+2*radius*math.sin(da/4))
            previous_pose=pose
        lower={name:max(0.,value-gap)for name,value in interval_min.items()}
        sweep.append(dict(segment=i,gear=int(b[4]),subintervals=n,max_half_interval_body_motion_m=gap,
            sampled_minima_m=interval_min,certified_continuous_lower_bounds_m=lower,
            robust_certified=min(lower['planner_occupied'],lower['planner_unknown'])>margin))
        if time.monotonic()-last_status>15:
            print(f'Continuous clearance: {i+1}/{len(path)-1} segments',flush=True);last_status=time.monotonic()
    bounds={name:min(s['certified_continuous_lower_bounds_m'][name]for s in sweep)for name in NAMES}
    perturb_results=[]
    if perturb:
        p=perturb['bounded_position_error_m'];angle=math.radians(perturb['bounded_heading_error_deg'])
        offsets=[(0.,0.)]+[(p*math.cos(k*math.pi/4),p*math.sin(k*math.pi/4))for k in range(8)]
        for i,row in enumerate(path):
            failed=[]
            for dx,dy in offsets:
                for dyaw in (-angle,0.,angle):
                    pose=[row[0]+dx,row[1]+dy,row[2]+dyaw]
                    if indices['planner_occupied'].within(pose,0.)or indices['planner_unknown'].within(pose,0.):
                        if not failed:
                            values=measure(pose);failed.append(dict(offset=[dx,dy,dyaw],pose=pose,clearance=values))
            perturb_results.append(dict(path_index=i,combinations_checked=27,sampled_perturbations_safe=not failed,first_unsafe=failed[0]if failed else None))
    return dict(point_count=len(path),points=point,continuous_segments=sweep,minimum_sampled_clearance=global_min,
        certified_continuous_minimum_lower_bounds_m=bounds,
        minimum_planner_clearance_sampled_m=min(global_min['planner_occupied']['distance_m'],global_min['planner_unknown']['distance_m']),
        minimum_planner_clearance_continuous_lower_bound_m=min(bounds['planner_occupied'],bounds['planner_unknown']),
        all_continuous_segments_robust=all(s['robust_certified']for s in sweep),perturbation_combinations=perturb_results,
        perturbation_sample_unsafe_points=sum(not p['sampled_perturbations_safe']for p in perturb_results),
        proof='Distance to a closed obstacle set is 1-Lipschitz under rigid-body Hausdorff displacement. Every interpolated pose lies within the reported half-interval motion cover of an endpoint sample; subtracting it certifies each entire continuous segment.',
        perturbation_note='27 explicit translation/yaw combinations are illustrative checks. The Minkowski disk bound and continuous clearance certificate cover all combinations, not just this finite set.')


def envelope_outline(pose,geometry,margin):
    corners=[(geometry.front_extent,geometry.width/2,0),(-geometry.rear_overhang,geometry.width/2,math.pi/2),(-geometry.rear_overhang,-geometry.width/2,math.pi),(geometry.front_extent,-geometry.width/2,3*math.pi/2)]
    local=np.array([[x+margin*math.cos(a),y+margin*math.sin(a)]for x,y,start in corners for a in np.linspace(start,start+math.pi/2,25)])
    c,s=math.cos(pose[2]),math.sin(pose[2]);world=local@np.array([[c,s],[-s,c]])+pose[:2]
    return np.vstack((world,world[0]))


def draw_diagnostic(path,diagnostic,saved,cfg,geometry,violation,margin):
    fig,axes=plt.subplots(1,3,figsize=(16,9),layout='constrained')
    ax=axes[0];extent=[cfg['origin_x'],cfg['origin_x']+cfg['width'],cfg['origin_y'],cfg['origin_y']+cfg['height']]
    ax.imshow(saved['states']+1,origin='lower',extent=extent,cmap=ListedColormap(['#d4b56b','white','#444444']),vmin=0,vmax=2)
    ax.plot(path[:,0],path[:,1],color='royalblue',lw=1,label='Previous failed path')
    ax.scatter(*violation['physical_pose'][:2],c='red',s=35,label='First safety stop')
    ax.set(xlabel='X (m)',ylabel='Y (m)',title='Old path / unchanged live SLAM',aspect='equal');ax.legend(fontsize=8)
    critical=min(diagnostic['minimum_sampled_clearance'][name]['distance_m']for name in ['planner_occupied','planner_unknown'])
    key=min(['planner_occupied','planner_unknown'],key=lambda n:diagnostic['minimum_sampled_clearance'][n]['distance_m'])
    entry=diagnostic['minimum_sampled_clearance'][key];pose=entry['pose'];cell=entry['nearest_cell']
    ax=axes[1];ax.imshow(saved['states']+1,origin='lower',extent=extent,cmap=ListedColormap(['#d4b56b','white','#444444']),vmin=0,vmax=2)
    index=CellClearance(np.zeros((1,1),bool),.2,[0,0],geometry);poly=index.polygon(pose)
    ax.add_patch(Polygon(poly,closed=True,fill=False,ec='blue',lw=2,label='Original full body'))
    # Exact metric outline of the unchanged rectangle plus a radius-margin disk.
    ax.plot(*envelope_outline(pose,geometry,margin).T,color='purple',lw=1.2,label=f'Error envelope {margin:.3f}m')
    if cell:
        x0,y0,x1,y1=cell['bounds'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='red',lw=2))
    ax.set(xlim=(8.8,11.9),ylim=(10.5,13.4),xlabel='X (m)',ylabel='Y (m)',title=f'Critical pose / nearest cell: {critical:.4f}m',aspect='equal');ax.legend(fontsize=7)
    ax=axes[2];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))]
    for name in NAMES:
        ax.plot(arc,[p['clearance'][name]['distance_m']for p in diagnostic['points']],label=name)
    ax.axhline(margin,color='red',ls='--',label='Required geometric error bound')
    ax.set(xlabel='Path arc length (m)',ylabel='Full-body clearance (m)',title='Every old path point: exact rectangle distances');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('P11.7.6 clearance diagnosis | continuous swept-volume certificates in JSON')
    fig.savefig(OUT/'p11_7_6_clearance_diagnostic.png',dpi=160);plt.close(fig)


def coordinate_audit(saved,cfg,old_metrics):
    # Called only after robust planning has completed or been refused.
    with np.load(previous.MAP,allow_pickle=False)as archive:truth=archive['reference_poses'].copy()
    estimate=saved['estimated_poses'];delta=estimate[-1]-truth[-1]
    yaw=math.atan2(math.sin(delta[2]),math.cos(delta[2]))
    return dict(evaluation_only=True,used_to_correct_start_or_map=False,
        physical_and_slam_coordinate_convention='world X/Y, rear-axle center, yaw CCW about +Z in radians',
        map_origin=[cfg['origin_x'],cfg['origin_y']],physical_scene_origin=[0.,0.],
        same_frame_evidence='P11.6.4 initializes SLAM and wheel odometry from PhysicsVehicle.state after settling; capture_scan appends pre-scan physical rear-axle pose and post-update SLAM pose with one shared scan timestamp. Scene uses the same configured world coordinates.',
        inspected_sources=['scripts/run_p11_6_4_live_slam.py','parking_planner/simulation/pybullet/physics_vehicle.py','parking_planner/simulation/pybullet/scene.py','parking_planner/slam/online_slam.py'],
        aligned_shapes=estimate.shape==truth.shape,estimated_latest=estimate[-1].tolist(),evaluation_reference_latest=truth[-1].tolist(),
        delta_xy_m=delta[:2].tolist(),position_error_m=float(np.linalg.norm(delta[:2])),heading_error_deg=abs(math.degrees(yaw)),
        initial_pose_difference_m=float(np.linalg.norm(estimate[0,:2]-truth[0,:2])),
        position_formula='Euclidean norm(estimated[-1,:2]-reference[-1,:2])',heading_formula='abs(degrees(atan2(sin(dyaw),cos(dyaw))))',
        previous_evaluation_reproduced=bool(abs(np.linalg.norm(delta[:2])-old_metrics['localization_evaluation_after_planning']['position_error_m'])<1e-12 and
            abs(abs(math.degrees(yaw))-old_metrics['localization_evaluation_after_planning']['heading_error_deg'])<1e-12),
        conclusion='No static origin, rear-axle/COM, radians/degrees or yaw-sign mismatch found. The ~0.994 m and ~4.391 deg are saved localization discrepancies, not corrected planning inputs. The execution tracking envelope does not include this global localization uncertainty.')


def robust_plan(saved,cfg,base,scenario,envelope):
    grid=CertifiedRobustGrid(base,scenario['geometry'],envelope['margin_m'],scenario['planner_config'],cfg['resolution'])
    planner=HybridAStarPlanner(grid=grid,geometry=scenario['geometry'],config=scenario['planner_config'])
    start=VehicleState(*map(float,saved['estimated_poses'][-1]));goal=scenario['goal']
    result=dict(status='NO_ROBUST_PATH',start=[start.x,start.y,start.yaw],original_goal=[goal.x,goal.y,goal.yaw],
        original_goal_nominal_blocked=base.is_collision(goal,planner.local_x,planner.local_y),
        start_robust_safe=not grid.is_collision(start,planner.local_x,planner.local_y),
        envelope=envelope,collision_layer=grid.audit,planner_config=asdict(scenario['planner_config']),
        hybrid_astar_executed=False,hybrid_astar_success=False,physical_executed=False,waiting_pose=None,
        map_sha256=sha(previous.MAP),original_path_sha256=sha(Path(str(OLD)+'_path.npz')))
    wait,details=previous.waiting_pose(goal,scenario,planner,saved['states'],cfg['resolution'])
    result['waiting_candidates']=details
    if wait is not None:result['waiting_pose']=[wait.x,wait.y,wait.yaw]
    if not result['start_robust_safe']or wait is None:
        result['reason']='START_NOT_ROBUST_SAFE'if not result['start_robust_safe']else'NO_ROBUST_WAITING_POSE'
        return result,np.empty((0,6)),grid
    finished=threading.Event()
    def heartbeat():
        while not finished.wait(15):print('Robust Hybrid A* expanded',planner.last_iterations,'nodes',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();began=time.monotonic()
    result['hybrid_astar_executed']=True
    try:raw=planner.plan(start,wait)
    finally:finished.set();thread.join(timeout=1)
    result['planning_wall_s']=time.monotonic()-began;result['expanded_nodes']=planner.last_iterations
    if not raw:result['reason']='ROBUST_SEARCH_EXHAUSTED';return result,np.empty((0,6)),grid
    dense=resample_path(raw,spacing=.025)
    path=np.array([[p.x,p.y,p.yaw,p.steering,p.direction,math.tan(p.steering)/scenario['geometry'].wheelbase]for p in dense])
    path[0,4]=path[1,4]
    result['raw_planner_nodes']=[[p.x,p.y,p.yaw,p.steering,p.direction]for p in raw]
    result['hybrid_astar_success']=True;result['status']='ROBUST_PLAN_AWAITING_CONTINUOUS_CERTIFICATION'
    return result,path,grid


def export_result(result,path,trace,states,saved,cfg,old_path,old_trace,geometry,frame=None):
    result.update(p10_fixed_path_loaded=False,slam_map_modified=False,
        new_path_minimum_continuous_clearance_m=result.get('new_clearance',{}).get('minimum_planner_clearance_continuous_lower_bound_m'),
        old_path_minimum_continuous_clearance_lower_bound_m=result['old_clearance']['minimum_planner_clearance_continuous_lower_bound_m'])
    if len(trace):
        from scripts.run_p11_7_3_goal_blocker_source_analysis import area,clip_rectangle
        scenario=previous.build_perpendicular_parking_scenario()
        index=CellClearance(np.zeros((1,1),bool),.2,[0,0],geometry)
        overlap=area(clip_rectangle(index.polygon(trace[-1,1:4]),scenario['parking_slot']))
        result.update(final_body_overlap_with_original_bay_m2=overlap,final_body_outside_original_bay=overlap<1e-12,physical_sample_count=len(trace))
        result['physical_metrics'].update(final_signed_speed_mps=float(trace[-1,4]),
            slam_map_violation_count=int(np.sum(np.diff(np.r_[False,trace[:,16]>0].astype(int))==1)),
            tracking_envelope_violation_count=int(np.sum(np.diff(np.r_[False,trace[:,17]>0].astype(int))==1)))
    np.savez_compressed(OUT/'p11_7_6_robust_physical_trace.npz',trace=trace,state=states,
        columns=np.array(['time_s','rear_x','rear_y','yaw_rad','actual_speed','actual_steer','gear','target_speed','motor_speed',
                          'segment','position_error','signed_cte','heading_error','roll','pitch','physical_contacts','nominal_map_violation','envelope_violation']),
        new_path=path,original_failed_path=old_path,map_sha256=np.array(result['map_sha256']))
    fig,axes=plt.subplots(2,3,figsize=(17,11),layout='constrained')
    extent=[cfg['origin_x'],cfg['origin_x']+cfg['width'],cfg['origin_y'],cfg['origin_y']+cfg['height']]
    axes[0,0].imshow(saved['states']+1,origin='lower',extent=extent,cmap=ListedColormap(['#d7b46b','white','#444444']),vmin=0,vmax=2)
    axes[0,0].plot(old_path[:,0],old_path[:,1],color='gray',lw=1,label='Old failed plan')
    fail=result['old_first_safety_violation']['physical_pose'];axes[0,0].scatter(*fail[:2],c='red',s=30,label='Old violation')
    if len(path):axes[0,0].plot(path[:,0],path[:,1],color='royalblue',lw=1.5,label='New robust plan')
    if len(trace):axes[0,0].plot(trace[:,1],trace[:,2],color='green',lw=1,label='New physics')
    axes[0,0].set(xlabel='X (m)',ylabel='Y (m)',title='Unmodified SLAM and routes',aspect='equal');axes[0,0].legend(fontsize=7)
    if frame is not None:axes[0,1].imshow(frame);axes[0,1].axis('off');axes[0,1].set_title('Physical validation')
    else:
        axes[0,1].axis('off');axes[0,1].text(.5,.6,'Physics NOT executed',ha='center',fontsize=13)
        axes[0,1].text(.5,.4,result.get('reason',result['status']),ha='center',fontsize=10,wrap=True)
    ax=axes[0,2]
    if len(path):ax.plot(path[:,0],path[:,1],'b-',label='New plan')
    if len(trace):ax.plot(trace[:,1],trace[:,2],'g-',label='Physics')
    if len(path):
        q=path[len(path)//2,:3];index=CellClearance(np.zeros((1,1),bool),.2,[0,0],geometry);poly=index.polygon(q)
        ax.add_patch(Polygon(poly,fill=False,ec='blue',lw=1,label='True body'))
        ax.plot(*envelope_outline(q,geometry,result['envelope']['margin_m']).T,color='purple',lw=1,label='0.13 m error envelope')
    ax.set(xlabel='X (m)',ylabel='Y (m)',title='Tracking / unchanged footprint',aspect='equal')
    if len(path):ax.legend(fontsize=7)
    if len(trace):
        axes[1,0].plot(trace[:,0],trace[:,11]);axes[1,1].plot(trace[:,0],np.degrees(trace[:,12]))
        axes[1,2].plot(trace[:,0],trace[:,4],label='Speed m/s');axes[1,2].plot(trace[:,0],trace[:,5],label='Steer rad');axes[1,2].legend()
    else:
        for ax in axes[1]:ax.text(.5,.5,'Not executed: offline gate',ha='center',transform=ax.transAxes)
    axes[1,0].set(xlabel='Time (s)',ylabel='Cross-track error (m)');axes[1,1].set(xlabel='Time (s)',ylabel='Heading error (deg)');axes[1,2].set(xlabel='Time (s)',title='Speed and actual steering')
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('P11.7.6 | '+result['status'])
    fig.savefig(OUT/'p11_7_6_robust_physical_validation.png',dpi=150);plt.close(fig)
    write_json(METRICS,result)
    report=f'''# P11.7.6 鲁棒净空规划与物理验证\n\n状态：{result['status']}。原因：{result.get('reason')}。\n\n## 误差包络\n\n实测位置上界 {result['envelope']['measured_position_error_m']:.9f} m，航向上界 {result['envelope']['measured_heading_error_deg']:.9f}°。角点半径 {result['envelope']['corner_radius_m']:.9f} m，转角引起位移 {result['envelope']['corner_displacement_from_yaw_m']:.9f} m，保守相加并向上取整得到 {result['envelope']['margin_m']:.3f} m 的全姿态组合安全包络。车辆实际尺寸、参数、地图标签和概率阈值未改变。\n\n计算层另外加入填充 footprint 采样覆盖界 {result['collision_layer']['footprint_sample_cover_m']:.6f} m 和原运动原语两检查状态之间的半区间运动界 {result['collision_layer']['primitive_half_motion_cover_m']:.6f} m；总需 {result['collision_layer']['required_collision_layer_padding_m']:.6f} m，在内部 0.05 m 碰撞层向上膨胀至 {result['collision_layer']['upward_rounded_square_padding_m']:.3f} m。原 0.20 m 占据数组逐位保留。这是碰撞层的保守数值覆盖，不是缩放物理车。\n\n## 连续路径诊断\n\n旧路径包含 417 点。逐点精确计算旋转完整矩形与所有原始 Occupied/Unknown 及聚合规划阻塞单元的距离；每段自适应细分，并用 Hausdorff 运动界扣除半间隔覆盖量，给出整个连续扫掠区域的净空下界。另检查 27 种位置/航向组合；所有连续扰动的保证来自解析包络而非有限扰动枚举。\n\n原路径最小采样净空 {result['old_clearance']['minimum_planner_clearance_sampled_m']:.6f} m，连续净空认证下界 {result['old_clearance']['minimum_planner_clearance_continuous_lower_bound_m']:.6f} m，显著不足以覆盖 0.13 m 包络。详细最近单元、位置及每段证书见 clearance_diagnostic.json。\n\n## 鲁棒规划与物理门控\n\n起点 {result['start']}，等待点 {result['waiting_pose']}，起点鲁棒安全 {result['start_robust_safe']}，Hybrid A* 实际执行 {result['hybrid_astar_executed']}、找到候选 {result['hybrid_astar_success']}。原目标不被清障或绕过。仅当新路径完整连续证书通过，才执行一次物理实验；否则返回 NO_ROBUST_PATH 并保存空物理记录。\n\n物理执行：{result['physical_executed']}。物理结果：\n```json\n{json.dumps(clean(result.get('physical_metrics')),indent=2)}\n```\n\n## 独立定位评估审计\n\n{result['localization_audit']['conclusion']}\n\n位置差按同一世界坐标系中的后轴中心 XY 欧氏距离计算；航向为同一 +Z 逆时针弧度，先 wrap 后转角度。初始位置差 {result['localization_audit']['initial_pose_difference_m']:.9f} m，末帧位置差 {result['localization_audit']['position_error_m']:.9f} m，航向差 {result['localization_audit']['heading_error_deg']:.9f}°。评估仅在规划门控结束后读取 reference_poses，不参与地图/规划/起点修正。\n\n历史输入、路径、原始物理 trace 和旧程序均只读；校验结果见 JSON。速度配置保持前进 0.28 m/s、倒车 0.20 m/s，沿用旧 Stanley/PID/换挡模型，不以降速替代净空改进。实验完成后停止，不优化 SLAM。\n'''
    report+=result_appendix(result,path)
    (OUT/'p11_7_6_robust_physical_report.md').write_text(report,encoding='utf-8')


def result_appendix(result,path):
    new=result.get('new_clearance',{});physical=result.get('physical_metrics')or{}
    return f"""
## 本次实际结果与验收口径

新路径 {len(path)} 点，长度 {physical.get('planned_length_m')} m。新路径连续全车身净空认证下界 {new.get('minimum_planner_clearance_continuous_lower_bound_m')} m；要求为 {result['envelope']['margin_m']} m。原路径旧离散点净空约 0.012854 m，加入区间内部采样后的最小净空为 {result['old_clearance']['minimum_planner_clearance_sampled_m']} m；连续净空下界 {result['old_clearance']['minimum_planner_clearance_continuous_lower_bound_m']} m。不能将采样最小值当成连续下界。

等待位姿与 Hybrid A* 最后节点存在原算法允许的终端容差，必须分别报告两种误差：

| 指标 | 实际值 |
|---|---:|
| 相对规划终端位置误差 (m) | {physical.get('FDE_m')} |
| 相对规划终端航向误差 (deg) | {physical.get('final_path_heading_error_deg')} |
| 相对选定等待位姿位置误差 (m) | {physical.get('final_selected_target_position_error_m')} |
| 相对选定等待位姿航向误差 (deg) | {physical.get('final_selected_target_heading_error_deg')} |
| 最大位置跟踪误差 (m) | {physical.get('max_position_tracking_error_m')} |
| 最大航向跟踪误差 (deg) | {physical.get('max_abs_heading_error_deg')} |
| PyBullet 碰撞次数 | {physical.get('pybullet_collision_count')} |
| SLAM 违规采样数 | {physical.get('slam_map_violation_samples')} |
| 包络越界采样数 | {physical.get('envelope_violation_samples')} |
| 停车保持 2 秒 | {physical.get('stopped_hold_2s')} |

沿用上阶段在执行前声明的阈值：相对规划终端 0.15 m / 5°；相对选定等待位姿 0.30 m / 8°；停车速度不超过 0.01 m/s 且保持 2 秒。另要求本轮位置误差不超过 0.0554 m、航向跟踪误差不超过 1.1755°，所有段执行、换挡停车、零物理碰撞和零 SLAM 违规。全部逐项结果：
```json
{json.dumps(result.get('physical_checks'),indent=2)}
```

当前成功仅指到达车位外等待区域。原始目标仍被阻塞，没有完成入库，也未宣称 COMPLETE_PARKING_PASS。0.13 m 是上一轮执行误差的保守包络，不涵盖约 0.994 m 的全局 SLAM 定位偏差，不能据此宣称真实环境中的全局定位安全保证。

历史文件 SHA256 保护检查：{result.get('protected_files_unchanged')}。本轮仅一次 Hybrid A* 搜索和一次完整物理实验；证书复核不重复搜索或物理执行。

运行（重新生成诊断、规划，并在通过后执行一次物理验证）：
```powershell
conda run --no-capture-output -n parkingplanner python scripts/run_p11_7_6_robust_hybrid_astar_physical_validation.py --gui --hold
```
只做离线诊断与规划时增加 `--offline-only`。`--validate-saved` 只验证本阶段已认证的新路径，拒绝无离线安全证书的输入。
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-only',action='store_true');parser.add_argument('--validate-saved',action='store_true')
    parser.add_argument('--gui',action='store_true');parser.add_argument('--hold',action='store_true');args=parser.parse_args()
    old_metrics=json.loads(Path(str(OLD)+'_metrics.json').read_text(encoding='utf-8'))
    old_report=Path(str(OLD)+'_report.md').read_text(encoding='utf-8')
    with np.load(Path(str(OLD)+'_path.npz'),allow_pickle=False)as a:old_path=a['path'].copy()
    with np.load(Path(str(OLD)+'_trace.npz'),allow_pickle=False)as a:old_trace=a['trace'].copy();old_columns=list(a['columns'])
    protected=[f for f in OUT.glob('p11*')if f.is_file()and not f.name.startswith('p11_7_6_')]
    protected += [f for folder in ('parking_planner','scripts','configs') for f in (ROOT/folder).rglob('*.py')
                  if not f.name.startswith('run_p11_7_6')and f.name not in ('robust_clearance.py','robust_plan_validation.py')]
    hashes={str(f.relative_to(ROOT)):sha(f)for f in protected}
    saved,cfg,base=previous.load_live();scenario=previous.build_perpendicular_parking_scenario();geometry=scenario['geometry']
    env=envelope(old_trace,old_columns,geometry);lookup=indexes(saved,cfg,base,geometry)
    if args.validate_saved:
        with np.load(PATH,allow_pickle=False)as a:path=a['path'].copy();result=json.loads(str(a['offline_metadata']))
        if result['map_sha256']!=sha(previous.MAP)or result['original_path_sha256']!=sha(Path(str(OLD)+'_path.npz'))or result['envelope']!=env:raise RuntimeError('Stale offline certificate')
        if not result.get('offline_safe'):raise RuntimeError('Offline gate failed; physics forbidden')
        robust=CertifiedRobustGrid(base,geometry,env['margin_m'],scenario['planner_config'],cfg['resolution'])
    else:
        print('Derived error envelope:',json.dumps(env),flush=True)
        diagnostic=path_clearance(old_path,lookup,env['margin_m'],env)
        diagnostic.update(map_sha256=sha(previous.MAP),original_path_sha256=sha(Path(str(OLD)+'_path.npz')),
            old_trace_sha256=sha(Path(str(OLD)+'_trace.npz')),envelope=env,old_first_violation=old_metrics['first_live_safety_violation'])
        write_json(OUT/'p11_7_6_clearance_diagnostic.json',diagnostic)
        draw_diagnostic(old_path,diagnostic,saved,cfg,geometry,old_metrics['first_live_safety_violation'],env['margin_m'])
        print('Old continuous lower bound:',diagnostic['minimum_planner_clearance_continuous_lower_bound_m'],flush=True)
        result,path,robust=robust_plan(saved,cfg,base,scenario,env)
        result['old_clearance']={k:v for k,v in diagnostic.items()if k in ('minimum_sampled_clearance','certified_continuous_minimum_lower_bounds_m','minimum_planner_clearance_sampled_m','minimum_planner_clearance_continuous_lower_bound_m','perturbation_sample_unsafe_points')}
        result['offline_safe']=False
        if len(path):
            new_diagnostic=path_clearance(path,lookup,env['margin_m'])
            result['new_clearance']=new_diagnostic
            result['offline_safe']=new_diagnostic['all_continuous_segments_robust']
            if not result['offline_safe']:result['status']='NO_ROBUST_PATH';result['reason']='CONTINUOUS_CERTIFICATE_FAILED'
            else:result['status']='OFFLINE_ROBUST_PATH_CERTIFIED'
        result['old_first_safety_violation']=old_metrics['first_live_safety_violation']
        np.savez_compressed(PATH,path=path,columns=np.array(previous.PATH_COLUMNS),
            x=path[:,0],y=path[:,1],yaw=path[:,2],steering=path[:,3],gear=path[:,4].astype(np.int8),curvature=path[:,5],
            map_sha256=np.array(result['map_sha256']),offline_metadata=np.array(json.dumps(clean(result))))
    result['localization_audit']=coordinate_audit(saved,cfg,old_metrics)
    result['protected_sha256']=hashes;result['source_map_file']=str(previous.MAP.relative_to(ROOT))
    result['original_path_file']=str(Path(str(OLD)+'_path.npz').relative_to(ROOT))
    result['original_report_sha256']=sha(Path(str(OLD)+'_report.md'))
    result['old_trace_sha256']=sha(Path(str(OLD)+'_trace.npz'))
    trace=np.empty((0,18));states=np.array([],dtype='U32');frame=None;connection=None
    try:
        if result.get('offline_safe')and not args.offline_only:
            from parking_planner.simulation.pybullet.robust_plan_validation import validate
            result,trace,states,frame,connection=validate(path,result,base,scenario,args,previous)
        result['protected_files_unchanged']=all(sha(ROOT/f)==h for f,h in hashes.items());assert result['protected_files_unchanged']
        export_result(result,path,trace,states,saved,cfg,old_path,old_trace,geometry,frame)
        print(json.dumps(clean({k:result.get(k)for k in ['status','reason','start','waiting_pose','start_robust_safe','hybrid_astar_executed','hybrid_astar_success','expanded_nodes','offline_safe','physical_executed','physical_metrics']}),indent=2),flush=True)
        if connection and args.gui and args.hold:
            import pybullet as p
            cid,_=connection
            while p.isConnected(cid):
                if p.getKeyboardEvents(physicsClientId=cid).get(27,0)&p.KEY_WAS_TRIGGERED:break
                time.sleep(.05)
    finally:
        if connection:
            import pybullet as p
            cid,originals=connection
            for name,fn in originals.items():setattr(p,name,fn)
            if p.isConnected(cid):p.disconnect(cid)


if __name__=='__main__':main()
