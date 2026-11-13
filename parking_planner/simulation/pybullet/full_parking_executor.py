"""Shared audit and reverse sign checks; never generates parking reference data."""
import math
import numpy as np
import pybullet as p
from parking_planner.vehicle.ackermann import AckermannModel,VehicleState
from scripts.run_p9_5_3_keyboard_exploration import vehicle_polygon,rectangles_overlap
from .reverse_controller import ReverseTracker
from .physics_vehicle import PhysicsVehicle
from .scene import ParkingScene
from .trajectory import wrap

# Predeclared for this experiment, not adjusted after the full run.
LIMITS=dict(reference_final_position_m=.10,reference_final_heading_deg=3.,
    requested_goal_position_m=.25,requested_goal_heading_deg=5.,
    bay_body_margin_m=.03,final_abs_speed_mps=.01,final_hold_s=2.,
    gear_engagement_abs_speed_mps=.015,min_segment_completion_percent=98.,
    max_cross_track_m=.15,max_heading_error_deg=6.,max_roll_pitch_deg=5.,
    max_collision_episodes=0,max_execution_s=180.)


def body_and_bay(state,scenario):
    polygon=vehicle_polygon(VehicleState(state['x'],state['y'],state['yaw']),scenario['geometry'])
    x0,y0,x1,y1=scenario['parking_slot']
    margins=[float(np.min(polygon[:,0]-x0)),float(np.min(x1-polygon[:,0])),
             float(np.min(polygon[:,1]-y0)),float(np.min(y1-polygon[:,1]))]
    overlap=[i for i,box in enumerate(scenario['parked_vehicles']) if rectangles_overlap(polygon,box)]
    grid=scenario['grid']
    outside=bool(np.any(polygon[:,0]<grid.origin_x) or np.any(polygon[:,0]>grid.origin_x+grid.width) or
                 np.any(polygon[:,1]<grid.origin_y) or np.any(polygon[:,1]>grid.origin_y+grid.height))
    # Paint is not a physical wall. Track its crossing separately from contacts.
    crossing=False
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        for x in (x0,x1):
            if abs(b[0]-a[0])>1e-12:
                u=(x-a[0])/(b[0]-a[0]);y=a[1]+u*(b[1]-a[1])
                crossing |= bool(0<=u<=1 and y0<=y<=y1)
    return polygon,margins,overlap,outside,crossing


def reverse_sign_checks(scenario):
    """Three short isolated physical diagnostics, NOT invented parking segments.

    Diagnostic references come from the EXISTING AckermannModel in an open
    aisle. Each car spawns with a small lateral/yaw error to exercise feedback.
    Parking execution exclusively uses the original saved P10 record.
    """
    results=[];dt=1/240
    for name,steer in [('straight',0.),('left_steer',.25),('right_steer',-.25)]:
        model=AckermannModel(wheelbase=scenario['geometry'].wheelbase)
        pose=VehicleState(7.,4.,0.,-.20);ref=[]
        for i in range(201):
            ref.append([pose.x,pose.y,pose.yaw,steer,-.20,-1,i*.05])
            pose=model.step(pose,acceleration=0.,steering=steer,dt=.05)
        ref[-1][4]=0.;ref=np.array(ref)
        cid=p.connect(p.DIRECT)
        try:
            p.setTimeStep(dt,physicsClientId=cid)
            p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=cid)
            scene=ParkingScene(cid,scenario)
            car=PhysicsVehicle(cid,scenario['geometry'],(7.,4.04,.015))
            for _ in range(480):car.command(0.,steer,dt);p.stepSimulation(physicsClientId=cid)
            tracker=ReverseTracker(ref,scenario['geometry'].wheelbase)
            initial=car.state();errs=[];min_speed=0.;command_signs=[];contacts=0
            for step in range(20*240):
                state=car.state();motor,angle,err,target=tracker.update(state,dt)
                command_signs.append(motor)
                car.command(motor,angle,dt);p.stepSimulation(physicsClientId=cid)
                current=car.state();errs.append(tracker.errors(current));min_speed=min(min_speed,current['speed'])
                contacts+=len([c for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid) if c[2] in scene.obstacle_ids])
                if tracker.stage=='COMPLETE':break
            final=car.state();yaw_change=float(wrap(final['yaw']-initial['yaw']))
            expected=(abs(yaw_change)<.08 if steer==0 else yaw_change*np.sign(steer)<-.05)
            passed=bool(tracker.stage=='COMPLETE' and min_speed<-.1 and expected and contacts==0 and
                        max(abs(e['cross_track']) for e in errs)<.10 and max(command_signs)<=0. and
                        abs(errs[-1]['cross_track'])<abs(errs[0]['cross_track']))
            results.append(dict(case=name,passed=passed,completed=tracker.stage=='COMPLETE',
                yaw_change_deg=math.degrees(yaw_change),initial_cte_m=errs[0]['cross_track'],final_cte_m=errs[-1]['cross_track'],
                max_abs_cte_m=max(abs(e['cross_track']) for e in errs),min_signed_speed_mps=min_speed,
                obstacle_contact_samples=contacts,role='isolated sign diagnostic, not parking trajectory'))
        finally:p.disconnect(cid)
    return results
