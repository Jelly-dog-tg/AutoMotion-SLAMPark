"""P11.7.6 single-run validation using unchanged P11.3/P11.5 dynamics/controllers.
The stage-specific loop retains prior speeds and limits; adds exact-body map
checking and rejects execution beyond the empirical planning error envelope.
"""
import math
import time
import numpy as np
from parking_planner.vehicle.ackermann import VehicleState
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.robust_clearance import CellClearance

def validate(path,result,grid,scenario,args,previous):
    LIMITS=previous.LIMITS
    result.update(actual_start=result["start"], selected_target=result["waiting_pose"], goal_footprint_blocked=result["original_goal_nominal_blocked"])
    result["physical_executed"]=True
    result["predeclared_physical_limits"]=dict(LIMITS)
    result["predeclared_empirical_error_bounds"]={k:result["envelope"][k] for k in ("bounded_position_error_m","bounded_heading_error_deg")}
    planned_segments=previous.planned_segments
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
    exact=CellClearance(grid.occupied,grid.resolution,[grid.origin_x,grid.origin_y],scenario["geometry"])
    def actual_collision(state):
        v=VehicleState(state["x"],state["y"],state["yaw"])
        return grid.is_collision(v,planner.local_x,planner.local_y) or exact.within([v.x,v.y,v.yaw],0.)
    def outside_envelope(error):
        return error["position_error"]>result["envelope"]["bounded_position_error_m"] or abs(math.degrees(error["heading_error"]))>result["envelope"]["bounded_heading_error_deg"]
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
            live_collision=actual_collision(state)
            rows.append([t,state['x'],state['y'],state['yaw'],state['speed'],state['steering'],state['direction'],0.,0.,0,
                error['position_error'],error['cross_track'],error['heading_error'],state['roll'],state['pitch'],len(contact),int(live_collision),int(outside_envelope(error))])
            phases.append('SETTLING')
            if outside_envelope(error):machine.emergency('TRACKING_ENVELOPE_EXCEEDED',t,state)
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
            live_collision=actual_collision(state)
            rows.append([t,state['x'],state['y'],state['yaw'],state['speed'],state['steering'],state['direction'],target,motor,
                machine.index,error['position_error'],error['cross_track'],error['heading_error'],state['roll'],state['pitch'],len(contact),int(live_collision),int(outside_envelope(error))])
            phases.append(machine.phase)
            if outside_envelope(error) and machine.phase not in('EMERGENCY_STOP','COMPLETE'):machine.emergency('TRACKING_ENVELOPE_EXCEEDED',t,state)
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
        trace=np.array(rows).reshape(-1,18);final=car.state();terminal=result['selected_target']
        fde=float(np.linalg.norm(np.array([final['x'],final['y']])-path[-1,:2]))
        heading=abs(math.degrees(float(wrap(final['yaw']-path[-1,2]))))
        selected_error=float(np.linalg.norm(np.array([final['x'],final['y']])-terminal[:2]))
        selected_heading=abs(math.degrees(float(wrap(final['yaw']-terminal[2]))))
        checks=dict(tracking_envelope_maintained=not np.any(trace[:,17]),machine_complete=machine.phase=='COMPLETE' and machine.failure is None,
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
        result['reason']=machine.failure
        result['physical_metrics'].update(max_position_tracking_error_m=float(np.max(trace[:,10])), pybullet_collision_count=sum(e['obstacle']!=-1 for e in collisions), geometric_collision_count=sum(e['obstacle']==-1 for e in collisions), slam_map_violation_samples=int(np.sum(trace[:,16]>0)), envelope_violation_samples=int(np.sum(trace[:,17]>0)), reached_waiting_pose=bool(checks['selected_position'] and checks['selected_heading']), stopped_hold_2s=checks['stopped_hold_2s'])
        result['runtime_collision_check']='Every 240 Hz step: original planner checker PLUS exact complete-rectangle positive/contact intersection with all blocked SLAM cells; PyBullet obstacle contacts and existing scene polygon test.'
        return result,trace,np.array(phases),frame,(cid,originals)
    except BaseException:
        for name,fn in originals.items():setattr(p,name,fn)
        if p.isConnected(cid):p.disconnect(cid)
        raise


