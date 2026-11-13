"""P11.7.4-.5 read-only live-map planning and independent physical validation."""
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
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import resample_path,path_metrics
from parking_planner.vehicle.ackermann import VehicleState
from scripts.run_p11_7_1_live_map_plan_preview import make_planner_grid,footprint_polygon

OUT=ROOT/'results';MAP=OUT/'p11_6_4_live_slam_map.npz'
PREFIX='p11_7_4_5_live_hybrid_astar'
PATH_FILE=OUT/f'{PREFIX}_path.npz'
# Declared before planning/physics, not relaxed after seeing the experiment.
LIMITS=dict(path_fde_m=.15,path_heading_error_deg=5.,selected_target_position_m=.30,
    selected_target_heading_deg=8.,max_cross_track_m=.15,max_heading_error_deg=8.,
    stop_speed_mps=.01,hold_s=2.,max_collision_episodes=0,max_roll_pitch_deg=5.,
    max_physics_s=240.,min_completion_percent=98.)
PATH_COLUMNS=['x','y','yaw','steering','gear','curvature']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_live():
    # Whitelist only permitted planning fields; reference_poses is NOT loaded.
    names=['states','log_odds','observed','probabilities','estimated_poses','odometry_poses','scan_timestamps_s','config']
    with np.load(MAP,allow_pickle=False)as archive:
        saved={name:archive[name].copy() for name in names}
    cfg=json.loads(str(saved['config'].item()))
    poses=saved['estimated_poses']
    if poses.ndim!=2 or poses.shape[1]!=3 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Invalid live estimated poses')
    if saved['odometry_poses'].shape!=poses.shape or saved['scan_timestamps_s'].shape!=(len(poses),):
        raise ValueError('Pose/odometry/timestamp histories do not align')
    if not np.isfinite(saved['odometry_poses']).all() or not np.isfinite(saved['scan_timestamps_s']).all() or np.any(np.diff(saved['scan_timestamps_s'])<0):
        raise ValueError('Invalid live odometry/timestamps')
    expected=(math.ceil(cfg['height']/cfg['resolution']),math.ceil(cfg['width']/cfg['resolution']))
    if saved['states'].shape!=expected:raise ValueError('Map shape does not match metadata')
    if not np.isin(saved['states'],[-1,0,1]).all():raise ValueError('Invalid map labels')
    probability=1/(1+np.exp(-saved['log_odds']));reclassified=np.full(expected,-1,dtype=np.int8)
    reclassified[saved['observed']&(probability<=cfg['free_threshold'])]=0
    reclassified[saved['observed']&(probability>=cfg['occupied_threshold'])]=1
    if not np.array_equal(reclassified,saved['states']) or not np.allclose(probability,saved['probabilities']):
        raise ValueError('Saved occupancy arrays disagree')
    return saved,cfg,make_planner_grid(saved['states'],cfg)


def footprint_check(pose,planner,states,raw_res):
    g=planner.grid;c,s=math.cos(pose.yaw),math.sin(pose.yaw)
    x=pose.x+c*planner.local_x-s*planner.local_y;y=pose.y+s*planner.local_x+c*planner.local_y
    ix=np.floor((x-g.origin_x)/g.resolution).astype(int);iy=np.floor((y-g.origin_y)/g.resolution).astype(int)
    outside=bool(np.any(ix<0)|np.any(iy<0)|np.any(ix>=g.cols)|np.any(iy>=g.rows))
    cells=sorted(set(zip(iy.tolist(),ix.tolist())))
    ratio=round(g.resolution/raw_res);blocked=[]
    for r,col in cells:
        if 0<=r<g.rows and 0<=col<g.cols and g.occupied[r,col]:
            b=states[r*ratio:(r+1)*ratio,col*ratio:(col+1)*ratio]
            blocked.append(dict(row=r,col=col,unknown=int(np.sum(b==-1)),occupied=int(np.sum(b==1))))
    rx=np.floor((x-g.origin_x)/raw_res).astype(int);ry=np.floor((y-g.origin_y)/raw_res).astype(int)
    r0=max(0,int(ry.min()));r1=min(states.shape[0],int(ry.max())+1)
    c0=max(0,int(rx.min()));c1=min(states.shape[1],int(rx.max())+1)
    around=states[max(0,r0-10):min(states.shape[0],r1+10),max(0,c0-10):min(states.shape[1],c1+10)]
    return dict(pose=[float(pose.x),float(pose.y),float(pose.yaw)],valid_position=bool(np.isfinite([pose.x,pose.y,pose.yaw]).all() and not outside),
        footprint_blocked=bool(g.is_collision(pose,planner.local_x,planner.local_y)),outside_map=outside,
        sampled_cells=len(cells),blocked_planner_cells=blocked,unknown_near_footprint_05m_bbox_plus_05m=int(np.sum(around==-1)),
        polygon=footprint_polygon(pose,planner.geometry).tolist())


def waiting_pose(goal,scenario,planner,states,res):
    direction=np.array([math.cos(goal.yaw),math.sin(goal.yaw)])
    x0,y0,x1,y1=scenario['parking_slot'];p=np.array([goal.x,goal.y])
    intersections=[]
    for axis,bounds in [(0,[x0,x1]),(1,[y0,y1])]:
        if abs(direction[axis])<1e-12:continue
        for bound in bounds:
            distance=(bound-p[axis])/direction[axis]
            q=p+distance*direction
            if distance>0 and x0-1e-9<=q[0]<=x1+1e-9 and y0-1e-9<=q[1]<=y1+1e-9:
                intersections.append((distance,q))
    if not intersections:raise ValueError('Goal heading does not intersect bay exit')
    entrance=min(intersections,key=lambda x:x[0])[1]
    candidates=[]
    for extra in np.arange(0.,2.01,.2):
        offset=planner.geometry.rear_overhang+.40+float(extra)
        xy=entrance+offset*direction
        pose=VehicleState(float(xy[0]),float(xy[1]),goal.yaw)
        check=footprint_check(pose,planner,states,res)
        poly=footprint_polygon(pose,planner.geometry)[:-1]
        # Projection along outward goal heading: all car corners must be at
        # least 0.40 m outside the entrance plane.
        clearance=float(np.min((poly-entrance)@direction))
        reserve=[]
        for dx,dy in [(0,0),(.2,0),(-.2,0),(0,.2),(0,-.2),(.2,.2),(.2,-.2),(-.2,.2),(-.2,-.2)]:
            for dyaw in [0.,math.radians(5),-math.radians(5)]:
                q=VehicleState(pose.x+dx,pose.y+dy,pose.yaw+dyaw)
                reserve.append(not planner.grid.is_collision(q,planner.local_x,planner.local_y))
        turns=[dict(gear=d,steering=a,safe=planner._propagate(pose,d,a)is not None)
               for d in (1,-1) for a in (-planner.config.max_steer,planner.config.max_steer)]
        safe=not check['footprint_blocked'] and clearance>=.40-1e-9 and all(reserve) and all(t['safe']for t in turns)
        candidates.append(dict(**check,entrance_clearance_m=clearance,extra_outward_offset_m=float(extra),
            reserve_checks=27,reserve_all_free=all(reserve),turn_primitives=turns,accepted=safe))
        if safe:return pose,dict(entrance_xy=entrance.tolist(),outward_heading_xy=direction.tolist(),
            reverse_entry_direction_xy=(-direction).tolist(),rear_overhang_m=planner.geometry.rear_overhang,
            nominal_clearance_m=.40,candidates=candidates,selected=[pose.x,pose.y,pose.yaw])
    return None,dict(entrance_xy=entrance.tolist(),nominal_clearance_m=.40,candidates=candidates,selected=None)


def plan_live(saved,cfg,grid,scenario):
    start=VehicleState(*map(float,saved['estimated_poses'][-1]))
    goal=scenario['goal'];planner=HybridAStarPlanner(grid=grid,geometry=scenario['geometry'],config=scenario['planner_config'])
    start_check=footprint_check(start,planner,saved['states'],cfg['resolution'])
    goal_check=footprint_check(goal,planner,saved['states'],cfg['resolution'])
    result=dict(map_file=str(MAP.relative_to(ROOT)),map_sha256=sha(MAP),actual_start=[start.x,start.y,start.yaw],
        start_source='estimated_poses[-1]',original_goal=[goal.x,goal.y,goal.yaw],start_check=start_check,goal_check=goal_check,
        original_goal_status='GOAL_FOOTPRINT_BLOCKED' if goal_check['footprint_blocked'] else 'GOAL_FOOTPRINT_SAFE',
        goal_footprint_blocked=goal_check['footprint_blocked'],unknown_and_occupied_blocked=True,
        slam_resolution_m=cfg['resolution'],planner_resolution_m=grid.resolution,map_config=cfg,
        planner_config=asdict(scenario['planner_config']),vehicle_geometry=asdict(scenario['geometry']),
        minimum_turning_radius_m=scenario['geometry'].wheelbase/math.tan(scenario['planner_config'].max_steer),
        hybrid_astar_executed=False,hybrid_astar_success=False,physical_validation_executed=False,complete_parking=False,
        waiting_pose=None,selected_target=None,expanded_nodes=0,limits=LIMITS,p10_fixed_path_loaded=False,
        ground_truth_used_for_planning=False,reference_poses_loaded_during_planning=False)
    target=goal
    if goal_check['footprint_blocked']:
        wait,details=waiting_pose(goal,scenario,planner,saved['states'],cfg['resolution'])
        result['waiting_pose_analysis']=details;result['waiting_pose']=details['selected'];target=wait
    if start_check['footprint_blocked'] or not start_check['valid_position']:
        result['status']='START_FOOTPRINT_BLOCKED';return result,[],planner
    if target is None:
        result['status']='WAIT_FOOTPRINT_BLOCKED';return result,[],planner
    result['selected_target']=[target.x,target.y,target.yaw]
    print('Planning from',result['actual_start'],'to',result['selected_target'],'original goal:',result['original_goal_status'],flush=True)
    done=threading.Event()
    def heartbeat():
        while not done.wait(15):print('Hybrid A* expanded',planner.last_iterations,'nodes',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();began=time.monotonic()
    result['hybrid_astar_executed']=True
    try:raw=planner.plan(start,target)
    finally:done.set();thread.join(timeout=1)
    result['planning_wall_time_s']=time.monotonic()-began;result['expanded_nodes']=planner.last_iterations
    if not raw:
        result['status']='HYBRID_ASTAR_NO_SAFE_PATH';return result,[],planner
    dense=resample_path(raw,spacing=.025)
    bad=[i for i,p in enumerate(dense) if grid.is_collision(p,planner.local_x,planner.local_y)]
    result['dense_footprint_collision_indices']=bad
    result['raw_planner_path']=[[p.x,p.y,p.yaw,p.steering,p.direction]for p in raw]
    if bad:
        result['status']='DENSE_PATH_SAFETY_FAILED';return result,[],planner
    result['hybrid_astar_success']=True;result['status']='NEW_SAFE_PATH_READY_FOR_PHYSICS'
    result['path_metrics']=path_metrics(dense);result['dense_points']=len(dense)
    result['planned_endpoint']=[dense[-1].x,dense[-1].y,dense[-1].yaw]
    return result,dense,planner


def save_path(dense,result):
    path=np.array([[p.x,p.y,p.yaw,p.steering,p.direction,math.tan(p.steering)/result['vehicle_geometry']['wheelbase']]for p in dense],float).reshape(-1,6)
    if len(path)>1:
        path[0,4]=path[1,4]
        if 'path_metrics'in result:
            result['planner_helper_switches_including_unused_start_label']=result['path_metrics']['gear_switches']
            result['path_metrics']['gear_switches']=int(np.sum(path[2:,4]!=path[1:-1,4]))
    np.savez_compressed(PATH_FILE,path=path,columns=np.array(PATH_COLUMNS),x=path[:,0],y=path[:,1],yaw=path[:,2],
        steering=path[:,3],gear=path[:,4].astype(np.int8),curvature=path[:,5],
        start_pose=np.array(result['actual_start']),original_goal=np.array(result['original_goal']),
        selected_target=np.array(result['selected_target']if result['selected_target']is not None else [np.nan]*3),
        original_planner_nodes=np.array(result.get('raw_planner_path',[]),float).reshape(-1,5),
        map_sha256=np.array(result['map_sha256']),planning_metadata=np.array(json.dumps(result)),
        gear_convention=np.array('gear[i] describes edge i-1 -> i; gear[0]=gear[1]; yaw is body heading in both directions'))
    return path


def planned_segments(path,wheelbase):
    from parking_planner.simulation.pybullet.parking_state_machine import RecordedSegment
    if len(path)<2:raise ValueError('No planned motion')
    edges=path[1:,4].astype(int)
    if not np.isin(edges,[-1,1]).all():raise ValueError('Invalid planned gear')
    boundaries=[0]+(np.flatnonzero(np.diff(edges))+1).tolist()+[len(path)-1]
    segments=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        points=path[a:b+1];gear=int(path[a+1,4]);distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1))]
        if distance[-1]<.03:raise ValueError('Degenerate planned gear segment')
        peak=.28 if gear>0 else .20
        speed=np.minimum(peak,np.minimum(.03+np.sqrt(.25*distance),np.sqrt(.5*np.maximum(distance[-1]-distance,.002))))
        speed[-1]=0.;steer=points[:,3].copy();steer[0]=steer[1]
        ref=np.column_stack((points[:,:3],steer,gear*speed,np.full(len(points),gear),distance/peak))
        segments.append(RecordedSegment(ref,a,b,gear,.6))
    return segments


def validation(path,result,grid,scenario,args):
    # Imported only once a fresh safe path exists. No P10 archive or loader.
    import pybullet as p
    from parking_planner.simulation.pybullet.physics_vehicle import PhysicsVehicle
    from parking_planner.simulation.pybullet.scene import ParkingScene
    from parking_planner.simulation.pybullet.parking_state_machine import ParkingStateMachine
    from parking_planner.simulation.pybullet.hud import EdgeStatusBars
    from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS
    from parking_planner.simulation.pybullet.trajectory import wrap
    from scripts.run_p9_5_3_keyboard_exploration import collision
    segments=planned_segments(path,scenario['geometry'].wheelbase)
    cid=p.connect(p.GUI if args.gui else p.DIRECT,options=WINDOW_OPTIONS if args.gui else '')
    rows=[];phases=[];originals={};forbidden=[];frame=None;collisions=[];gear_events=[];previous_contacts=set();paused=False
    try:
        dt=1/240;p.setTimeStep(dt,physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=cid)
        scene=ParkingScene(cid,scenario);car=PhysicsVehicle(cid,scenario['geometry'],result['actual_start'])
        machine=ParkingStateMachine(segments,scenario['geometry'].wheelbase)
        planner=HybridAStarPlanner(grid=grid,geometry=scenario['geometry'],config=scenario['planner_config'])
        for name in ('resetBasePositionAndOrientation','resetBaseVelocity','resetJointState','applyExternalForce','applyExternalTorque'):
            fn=getattr(p,name);originals[name]=fn
            def guard(*a,_fn=fn,_name=name,**kw):
                body=a[0]if a else kw.get('bodyUniqueId',kw.get('objectUniqueId'))
                if body==car.body:forbidden.append(_name);raise RuntimeError('Forbidden direct vehicle motion')
                return _fn(*a,**kw)
            setattr(p,name,guard)
        hud=None;previous=None
        if args.gui:
            scene.camera_target=[9.,9.,0.];scene.configure_gui();hud=EdgeStatusBars(cid)
            for a,b in zip(path[:-1],path[1:]):
                p.addUserDebugLine([*a[:2],.035],[*b[:2],.035],[.08,.3,1.]if b[4]>0 else [1.,.48,.08],2,physicsClientId=cid)
        t=0.;next_status=None
        for step in range(481):
            state=car.state();error=machine.trackers[0].errors(state)
            contact={c[2]for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid)if c[2]in scene.obstacle_ids and(c[8]<=0 or c[9]>0)}
            if collision(VehicleState(state['x'],state['y'],state['yaw']),scenario):contact.add(-1)
            for obstacle in contact-previous_contacts:collisions.append(dict(time_s=t,obstacle=obstacle,phase='SETTLING'))
            previous_contacts=contact
            live_collision=grid.is_collision(VehicleState(state['x'],state['y'],state['yaw']),planner.local_x,planner.local_y)
            rows.append([t,state['x'],state['y'],state['yaw'],state['speed'],state['steering'],state['direction'],0.,0.,0,
                error['position_error'],error['cross_track'],error['heading_error'],state['roll'],state['pitch'],len(contact),int(live_collision)])
            phases.append('SETTLING')
            if contact or live_collision:machine.emergency('INITIAL_PHYSICAL_OR_LIVE_COLLISION',t,state)
            if step<480:car.command(0.,machine.steering,dt);p.stepSimulation(physicsClientId=cid);t+=dt
        while not machine.finished:
            wall=time.monotonic();state=car.state()
            if args.gui:
                keys=p.getKeyboardEvents(physicsClientId=cid)
                if keys.get(32,0)&p.KEY_WAS_TRIGGERED:paused=not paused
                if any(keys.get(k,0)&p.KEY_WAS_TRIGGERED for k in (27,ord('q'),p.B3G_F6)):
                    paused=False;machine.emergency('USER_STOP',t,state)
            if paused:time.sleep(.02);continue
            motor,steer,err,target=machine.update(state,dt,t)
            if machine.phase!=next_status:
                print(f'Physics t={t:.2f}s {machine.phase} segment={machine.index+1}',flush=True);next_status=machine.phase
            old=car.direction;car.command(motor,steer,dt)
            if car.direction!=old:gear_events.append(dict(time_s=t,from_gear=old,to_gear=car.direction,speed_before_mps=state['speed']))
            p.stepSimulation(physicsClientId=cid);t+=dt;state=car.state()
            idx=max(0,machine.index);error=machine.trackers[idx].errors(state)
            contact={c[2]for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid)if c[2]in scene.obstacle_ids and(c[8]<=0 or c[9]>0)}
            if collision(VehicleState(state['x'],state['y'],state['yaw']),scenario):contact.add(-1)
            for obstacle in contact-previous_contacts:collisions.append(dict(time_s=t,obstacle=obstacle,pose=[state['x'],state['y'],state['yaw']]))
            previous_contacts=contact
            live_collision=grid.is_collision(VehicleState(state['x'],state['y'],state['yaw']),planner.local_x,planner.local_y)
            rows.append([t,state['x'],state['y'],state['yaw'],state['speed'],state['steering'],state['direction'],target,motor,
                machine.index,error['position_error'],error['cross_track'],error['heading_error'],state['roll'],state['pitch'],len(contact),int(live_collision)])
            phases.append(machine.phase)
            if contact and machine.phase!='EMERGENCY_STOP':machine.emergency('PHYSICAL_COLLISION',t,state)
            if live_collision and machine.phase not in('EMERGENCY_STOP','COMPLETE'):machine.emergency('ACTUAL_FOOTPRINT_ENTERED_BLOCKED_LIVE_CELL',t,state)
            if max(abs(state['roll']),abs(state['pitch']))>math.radians(LIMITS['max_roll_pitch_deg']) and machine.phase not in('EMERGENCY_STOP','COMPLETE'):
                machine.emergency('EXCESSIVE_TILT',t,state)
            if t>LIMITS['max_physics_s']and machine.phase not in('EMERGENCY_STOP','COMPLETE'):machine.emergency('TIMEOUT',t,state)
            if args.gui:
                if len(rows)%4==0:
                    point=[state['x'],state['y'],.055]
                    if previous is not None:p.addUserDebugLine(previous,point,[.05,.85,.3],3,physicsClientId=cid)
                    previous=point
                    hud.update(f'LIVE SLAM VALIDATION | {machine.phase} | {t:.2f}s | {machine.progress:.1f}% | gear={state["direction"]:+d}',
                        f'v={state["speed"]:+.3f} target={target:+.3f}m/s | steer={math.degrees(state["steering"]):+.1f}deg | yaw={math.degrees(state["yaw"]):+.1f}deg | CTE={error["cross_track"]:+.3f}m',
                        'SPACE pause/resume | F6 emergency | Q/ESC stop','BLUE forward / ORANGE reverse / GREEN physics')
                time.sleep(max(0,dt-(time.monotonic()-wall)))
        trace=np.array(rows).reshape(-1,17);final=car.state();terminal=result['selected_target']
        fde=float(np.linalg.norm(np.array([final['x'],final['y']])-path[-1,:2]))
        heading=abs(math.degrees(float(wrap(final['yaw']-path[-1,2]))))
        selected_error=float(np.linalg.norm(np.array([final['x'],final['y']])-terminal[:2]))
        selected_heading=abs(math.degrees(float(wrap(final['yaw']-terminal[2]))))
        checks=dict(machine_complete=machine.phase=='COMPLETE' and machine.failure is None,
            no_physical_collision=not collisions,no_live_map_collision=not np.any(trace[:,16]),
            path_fde=fde<=LIMITS['path_fde_m'],path_heading=heading<=LIMITS['path_heading_error_deg'],
            selected_position=selected_error<=LIMITS['selected_target_position_m'],selected_heading=selected_heading<=LIMITS['selected_target_heading_deg'],
            max_cte=float(np.max(np.abs(trace[:,11])))<=LIMITS['max_cross_track_m'],
            max_heading=float(np.max(np.abs(np.degrees(trace[:,12]))))<=LIMITS['max_heading_error_deg'],
            stable_roll_pitch=bool(np.max(np.abs(trace[:,13:15]))<=math.radians(LIMITS['max_roll_pitch_deg'])),
            stopped=abs(final['speed'])<=LIMITS['stop_speed_mps'],
            stopped_hold_2s=bool(len(trace)>=480 and np.all(np.abs(trace[-480:,4])<=LIMITS['stop_speed_mps'])),no_pose_resets=not forbidden,
            safe_gears=all(abs(e['speed_before_mps'])<=.015 for e in gear_events),
            all_segments_executed=len(machine.segment_events)==len(segments))
        valid=all(checks.values())
        result.update(physical_validation_executed=True,physical_validation_passed=valid,physical_checks=checks,
            status=('APPROACH_VALIDATED_GOAL_BLOCKED'if result['goal_footprint_blocked']else'COMPLETE_PARKING_PASS')if valid else'PHYSICAL_VALIDATION_FAILED',
            complete_parking=bool(valid and not result['goal_footprint_blocked']),failure_reason=machine.failure,
            collision_occurred=bool(collisions),live_map_collision_occurred=bool(np.any(trace[:,16])),collision_events=collisions,final_physical_state=final,
            physical_metrics=dict(ADE_m=float(np.mean(trace[:,10])),ADE_definition='mean nearest rear-axle reference polyline distance over physical samples; not a timestamp-aligned replay metric',
                FDE_m=fde,final_path_heading_error_deg=heading,final_selected_target_position_error_m=selected_error,
                final_selected_target_heading_error_deg=selected_heading,max_abs_cross_track_m=float(np.max(np.abs(trace[:,11]))),
                max_abs_heading_error_deg=float(np.max(np.abs(np.degrees(trace[:,12])))),
                planned_length_m=float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()),
                actual_distance_m=float(np.linalg.norm(np.diff(trace[:,1:3],axis=0),axis=1).sum()),total_time_s=t,
                forward_time_s=float(np.sum(trace[:,4]>.01)*dt),reverse_time_s=float(np.sum(trace[:,4]<-.01)*dt),
                direction_changes=sum(e['from_gear']!=0 for e in gear_events),collision_count=len(collisions)),
            shift_events=[{('configured_hold_s'if k=='original_recorded_hold_s'else k):v for k,v in e.items()}for e in machine.shifts],
            gear_events=gear_events,state_transitions=machine.events,vehicle_parameters=car.parameters(),forbidden_motion_calls=forbidden,
            speed_source='Configured new-plan control profile: forward 0.28 m/s, reverse 0.20 m/s, arc-distance start/stop ramps; NOT measured P10 speed.')
        if args.gui:
            hud.update(f'{result["status"]} | {t:.2f}s | {machine.progress:.1f}%',
                f'v={final["speed"]:+.5f}m/s | steer={math.degrees(final["steering"]):+.1f}deg | yaw={math.degrees(final["yaw"]):+.1f}deg | FDE={fde:.3f}m',
                'Original goal remains blocked; no complete-parking claim'if result['goal_footprint_blocked']else'Validation finished',
                'BLUE forward / ORANGE reverse / GREEN physics')
            camera=p.getDebugVisualizerCamera(physicsClientId=cid)
            frame=np.array(p.getCameraImage(1100,800,viewMatrix=camera[2],projectionMatrix=camera[3],renderer=p.ER_BULLET_HARDWARE_OPENGL,physicsClientId=cid)[2],dtype=np.uint8).reshape(800,1100,4)
        return result,trace,np.array(phases),frame,(cid,originals)
    except BaseException:
        for name,fn in originals.items():setattr(p,name,fn)
        if p.isConnected(cid):p.disconnect(cid)
        raise



def diagnose_validation(result,path,trace,saved,cfg,grid,scenario):
    """Read-only post-experiment diagnosis; never retries or modifies a path."""
    if not len(trace):return result
    from parking_planner.simulation.pybullet.trajectory import wrap
    final=result['final_physical_state'];goal=scenario['goal']
    result['final_original_goal_position_error_m']=float(np.linalg.norm(np.array([final['x'],final['y']])-[goal.x,goal.y]))
    result['final_original_goal_heading_error_deg']=abs(math.degrees(float(wrap(final['yaw']-goal.yaw))))
    poly=footprint_polygon(VehicleState(final['x'],final['y'],final['yaw']),scenario['geometry'])[:-1]
    x0,y0,x1,y1=scenario['parking_slot']
    result['final_body_inside_original_bay']=bool(poly[:,0].min()>=x0 and poly[:,0].max()<=x1 and poly[:,1].min()>=y0 and poly[:,1].max()<=y1)
    result['reached_selected_target']=bool(result['physical_metrics']['final_selected_target_position_error_m']<=LIMITS['selected_target_position_m'] and
                                          result['physical_metrics']['final_selected_target_heading_error_deg']<=LIMITS['selected_target_heading_deg'])
    result['reached_original_goal']=bool(result['complete_parking'])
    blocked=np.flatnonzero(trace[:,16]>0)
    if not len(blocked):return result
    from scripts.run_p11_7_3_goal_blocker_source_analysis import clip_rectangle,area
    i=int(blocked[0]);row=trace[i]
    planner=HybridAStarPlanner(grid=grid,geometry=scenario['geometry'],config=scenario['planner_config'])
    check=footprint_check(VehicleState(*row[1:4]),planner,saved['states'],cfg['resolution'])
    actual_poly=np.array(check['polygon'])[:-1]
    for cell in check['blocked_planner_cells']:
        x=grid.origin_x+cell['col']*grid.resolution;y=grid.origin_y+cell['row']*grid.resolution
        bounds=[x,y,x+grid.resolution,y+grid.resolution]
        cell.update(center=[x+grid.resolution/2,y+grid.resolution/2],bounds=bounds,
            actual_footprint_positive_overlap_m2=area(clip_rectangle(actual_poly,bounds)))
    segments=planned_segments(path,scenario['geometry'].wheelbase);segment=segments[int(row[9])]
    index=int(np.argmin(np.linalg.norm(segment.reference[:,:2]-row[1:3],axis=1)))
    near=segment.reference[index,:3]
    result['first_live_safety_violation']=dict(trace_index=i,time_s=float(row[0]),physical_pose=row[1:4].tolist(),
        actual_speed_mps=float(row[4]),signed_cte_m=float(row[11]),local_heading_error_deg=math.degrees(float(row[12])),
        actual_full_footprint_check=check,nearest_original_planned_pose=near.tolist(),
        nearest_planned_pose_blocked=bool(grid.is_collision(VehicleState(*near),planner.local_x,planner.local_y)),
        xy_difference_from_nearest_planned_pose_m=(row[1:3]-near[:2]).tolist(),
        finding='The planned path samples are safe, but the physical tracking deviation moves part of the actual full body into an Occupied-derived conservative planning cell. There is positive area overlap, not merely an ignored contact. Static waiting-pose reserve does not certify tracking-error clearance along the whole route.',
        response='Original policy retained; actuator braking and emergency stop, no retry, clearing, threshold relaxation or SLAM optimization.')
    return result

def export(result,path,trace,phases,saved,cfg,scenario,frame=None):
    tracecols=['time_s','rear_x','rear_y','yaw_rad','signed_speed_mps','actual_mean_front_steer_rad','gear','target_speed_mps',
        'motor_speed_mps','segment','nearest_path_position_error_m','signed_cte_m','heading_error_rad','roll_rad','pitch_rad','physical_contacts','live_map_blocked']
    np.savez_compressed(OUT/f'{PREFIX}_trace.npz',trace=trace,columns=np.array(tracecols),state=phases,
        planned_path=path,planned_columns=np.array(PATH_COLUMNS),map_sha256=np.array(result['map_sha256']))
    fig,axes=plt.subplots(2,3,figsize=(17,11),layout='constrained')
    ax=axes[0,0];extent=[cfg['origin_x'],cfg['origin_x']+cfg['width'],cfg['origin_y'],cfg['origin_y']+cfg['height']]
    ax.imshow(saved['states']+1,origin='lower',extent=extent,cmap=ListedColormap(['#d5b66b','white','#444444']),vmin=0,vmax=2,interpolation='nearest')
    ax.plot(*footprint_polygon(VehicleState(*result['actual_start']),scenario['geometry']).T,color='green',lw=1.4)
    ax.plot(*footprint_polygon(scenario['goal'],scenario['geometry']).T,color='red',lw=1.4)
    if result['waiting_pose']is not None:ax.plot(*footprint_polygon(VehicleState(*result['waiting_pose']),scenario['geometry']).T,color='purple',lw=1.4)
    if len(path):
        for gear,color in [(1,'royalblue'),(-1,'darkorange')]:
            selected=np.where(path[:,4]==gear,path[:,0],np.nan);ys=np.where(path[:,4]==gear,path[:,1],np.nan)
            ax.plot(selected,ys,color=color,lw=1.5,label='Forward'if gear>0 else'Reverse')
    if len(trace):ax.plot(trace[:,1],trace[:,2],color='green',lw=1,label='Physical execution')
    ax.set(xlim=(0,22),ylim=(0,22),title='Saved live SLAM / new plan / physics',xlabel='X (m)',ylabel='Y (m)',aspect='equal')
    if len(path):ax.legend(fontsize=7)
    if frame is not None:axes[0,1].imshow(frame);axes[0,1].axis('off');axes[0,1].set_title('Independent physical validation scene')
    else:axes[0,1].axis('off');axes[0,1].text(.5,.5,'Physics not executed',ha='center')
    ax=axes[0,2]
    if len(path):ax.plot(path[:,0],path[:,1],'b-',label='New Hybrid A*')
    if len(trace):
        ax.plot(trace[:,1],trace[:,2],color='green',label='Physics');stride=max(1,len(trace)//12)
        for row in trace[::stride]:ax.plot(*footprint_polygon(VehicleState(*row[1:4]),scenario['geometry']).T,color='green',lw=.4,alpha=.4)
    ax.set(xlabel='X (m)',ylabel='Y (m)',title='Reference vs actual footprint motion',aspect='equal')
    if len(path):ax.legend(fontsize=8)
    if len(trace):
        axes[1,0].plot(trace[:,0],trace[:,4],label='Actual');axes[1,0].plot(trace[:,0],trace[:,7],'--',label='Target');axes[1,0].legend()
        axes[1,1].plot(trace[:,0],np.degrees(trace[:,5]));axes[1,2].plot(trace[:,0],trace[:,11])
    else:
        for ax in axes[1]:ax.text(.5,.5,'Not executed',ha='center',transform=ax.transAxes)
    axes[1,0].set(xlabel='Physics time (s)',ylabel='Speed (m/s)',title='Signed speed')
    axes[1,1].set(xlabel='Physics time (s)',ylabel='Actual steer (deg)',title='Front wheel angle')
    axes[1,2].set(xlabel='Physics time (s)',ylabel='Cross-track error (m)',title='Real-state feedback error')
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle(f'Live SLAM -> new Hybrid A* -> physics | {result["status"]}',fontsize=13)
    fig.savefig(OUT/'p11_7_4_5_live_hybrid_astar_physical_validation.png',dpi=150);plt.close(fig)
    (OUT/f'{PREFIX}_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    report=f'''# P11.7.4–P11.7.5 Live SLAM 新规划与物理验证\n\n最终状态：{result['status']}。原目标：{result['original_goal_status']}。\n\n地图来源 `{result['map_file']}`，SHA256 `{result['map_sha256']}`。规划起点严格为 estimated_poses[-1]：{result['actual_start']}。原始目标 {result['original_goal']}；等待位姿 {result['waiting_pose']}。原始 0.05 m 地图、0.20 m 聚合图、Unknown/Occupied 保守阻塞策略及完整 4.5×1.8 m footprint 均未改变。\n\n等待位置由车位出口、目标朝向、后悬和 0.40 m 余量推导；要求整车在车位外，27 个位置/航向扰动检查及四个最大转角运动原语全部无碰撞。没有用 GT 或 P10 轨迹进行规划。\n\nHybrid A* 实际执行：{result['hybrid_astar_executed']}；成功：{result['hybrid_astar_success']}；扩展节点：{result['expanded_nodes']}。物理验证执行：{result['physical_validation_executed']}。完整泊车：{result['complete_parking']}。\n\n路径经现有 resample_path 以 0.025 m 间隔插值，并逐点按原规划器完整 footprint 检查，失败则不执行物理验证。路径文件明确保存 x/y/yaw/steering/gear/curvature；原始规划节点同时保留。物理验证从本次估计起点创建独立车辆，不等于恢复探索时真实车辆的状态。\n\n复用原 P11.3 PhysicsVehicle、P11.5 状态机及 Stanley/PID。新规划段配置速度轮廓和 0.6 s 换挡等待，不使用原执行速度或时间。车辆只创建时定位；运行时禁止直接位姿、速度、关节状态重置及外加运动力。实际接触和原二维完整矩形检测同时记录；偏入实时地图阻塞格会制动终止。\n\n物理指标：\n```json\n{json.dumps(result.get('physical_metrics'),indent=2)}\n```\n\n失败原因：{result.get('failure_reason')}。未执行的物理指标保留空值，不作为零碰撞或成功。所有实际状态及原规划路径见 NPZ。当前阶段不会修改 SLAM 或进行其他优化。\n\n运行：`conda run --no-capture-output -n parkingplanner python scripts/run_p11_7_4_5_live_slam_hybrid_astar_physical_validation.py --gui --hold`。\n'''
    if result.get('first_live_safety_violation'):
        failure=result['first_live_safety_violation']
        report+='\n## 首次安全终止的只读定位\n\n'
        report+=f"时刻 {failure['time_s']:.6f} s；实际位姿 {failure['physical_pose']}；横向偏差 {failure['signed_cte_m']:.6f} m，局部航向误差 {failure['local_heading_error_deg']:.6f}°。最近规划位姿仍通过原检查：{not failure['nearest_planned_pose_blocked']}。\n\n"
        report+='阻塞栅格及来源：\n```json\n'+json.dumps(failure['actual_full_footprint_check']['blocked_planner_cells'],indent=2)+'\n```\n\n'
        report+='原因是规划路线的局部净空未能容纳本次实际跟踪偏差。等待点本身通过余量检查，不代表整条路线具有同样余量。本轮仅分析并保留失败记录，未重规划、清障、修改控制器或放宽安全策略。\n'
    (OUT/f'{PREFIX}_report.md').write_text(report,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui',action='store_true');parser.add_argument('--hold',action='store_true')
    parser.add_argument('--plan-only',action='store_true',help='Save this run new plan without physics')
    parser.add_argument('--validate-saved-plan',action='store_true',help='Continue the new plan from THIS stage, only if map/hash match')
    args=parser.parse_args()
    if args.plan_only and args.validate_saved_plan:parser.error('Choose one staging option')
    # Preserve existing P11 results/code. Do not even read a P10 trajectory archive.
    protected=[f for f in OUT.glob('p11*') if f.is_file() and not f.name.startswith('p11_7_4_5_')]
    protected += list((ROOT/'parking_planner').rglob('*.py'))+list((ROOT/'configs').rglob('*.py'))
    hashes={str(f.relative_to(ROOT)):sha(f)for f in protected}
    saved,cfg,grid=load_live();scenario=build_perpendicular_parking_scenario()
    if args.validate_saved_plan:
        with np.load(PATH_FILE,allow_pickle=False)as archive:
            path=archive['path'].copy();result=json.loads(str(archive['planning_metadata'].item()))
        if result['map_sha256']!=sha(MAP)or not np.array_equal(np.array(result['actual_start']),saved['estimated_poses'][-1]):raise RuntimeError('Saved new plan is stale')
        if result['limits']!=LIMITS:raise RuntimeError('Validation thresholds changed')
        planner=HybridAStarPlanner(grid=grid,geometry=scenario['geometry'],config=scenario['planner_config'])
        if any(grid.is_collision(VehicleState(*r[:3]),planner.local_x,planner.local_y)for r in path):raise RuntimeError('Saved new path is blocked')
    else:
        result,dense,planner=plan_live(saved,cfg,grid,scenario);path=save_path(dense,result)
    result['live_input_array_inventory']={name:dict(shape=list(value.shape),dtype=str(value.dtype)) for name,value in saved.items()}
    result['latest_odometry_pose']=saved['odometry_poses'][-1].tolist()
    result['latest_scan_timestamp_s']=float(saved['scan_timestamps_s'][-1])
    result['configured_planned_segment_hold_s']=.6
    result['planned_motion_segment_count']=len(planned_segments(path,scenario['geometry'].wheelbase))if len(path)>1 else 0
    # Evaluation-only reference poses may be opened ONLY after planning finishes.
    with np.load(MAP,allow_pickle=False)as archive:
        if 'reference_poses'in archive:
            truth=archive['reference_poses'][-1];estimate=np.array(result['actual_start'])
            angle=math.atan2(math.sin(estimate[2]-truth[2]),math.cos(estimate[2]-truth[2]))
            result['localization_evaluation_after_planning']=dict(position_error_m=float(np.linalg.norm(estimate[:2]-truth[:2])),heading_error_deg=abs(math.degrees(angle)),
                role='evaluation only; not initial physical state, planner start, or map construction')
    trace=np.empty((0,17));phases=np.array([],dtype='U32');frame=None;connection=None
    result.setdefault('physical_validation_passed',False);result.setdefault('collision_occurred',None);result.setdefault('physical_metrics',None)
    try:
        if result['hybrid_astar_success']and len(path)>1 and not args.plan_only:
            result,trace,phases,frame,connection=validation(path,result,grid,scenario,args)
        result=diagnose_validation(result,path,trace,saved,cfg,grid,scenario)
        result['previous_files_unchanged']=all(sha(ROOT/f)==h for f,h in hashes.items());result['protected_sha256']=hashes
        assert result['previous_files_unchanged']
        result['path_file_sha256']=sha(PATH_FILE)
        export(result,path,trace,phases,saved,cfg,scenario,frame)
        print(json.dumps({k:v for k,v in result.items()if k in('status','actual_start','original_goal_status','waiting_pose','hybrid_astar_executed','hybrid_astar_success','expanded_nodes','physical_validation_executed','physical_metrics','failure_reason','complete_parking')},indent=2),flush=True)
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
