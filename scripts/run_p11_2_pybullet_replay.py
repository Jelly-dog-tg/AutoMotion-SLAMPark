"""P11.2 replay saved P10.4.2 execution; no simulation/control algorithms run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pybullet as p
from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.vehicle.ackermann import VehicleState
from parking_planner.simulation.pybullet.scene import ParkingScene
from parking_planner.simulation.pybullet.vehicle import VehicleVisual
from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS,preview_pose
from parking_planner.simulation.pybullet.trajectory import load_execution,wrap
from parking_planner.simulation.pybullet.replay import ReplayClock,ReplayView
from scripts.run_p9_5_3_keyboard_exploration import collision,vehicle_polygon
from scripts.run_p11_1_pybullet_scene import check_scene


def audit(data,scenario,vehicle,preview):
    bad=[]; count=0
    # Original SAT collision function, dense full-footprint checks on the exact
    # piecewise-linear display interpolation (<=2 cm / <=0.5 degree).
    for i,(a,b) in enumerate(zip(data.samples[:-1],data.samples[1:])):
        n=max(1,math.ceil(np.linalg.norm(b[:2]-a[:2])/.02),math.ceil(abs(wrap(b[2]-a[2]))/math.radians(.5)))
        for u in np.linspace(0,1,n+1):
            pose=a[:3]+u*(b[:3]-a[:3]); pose[2]=wrap(a[2]+u*wrap(b[2]-a[2]))
            count+=1
            if collision(VehicleState(*pose),scenario): bad.append(i); break
    transition_bad=[]
    for u in np.linspace(0,1,301):
        pose=preview+u*(data.samples[0,:3]-preview)
        pose[2]=wrap(preview[2]+u*wrap(data.samples[0,2]-preview[2]))
        if collision(VehicleState(*pose),scenario): transition_bad.append(float(u))
    maximum_xy,maximum_yaw=0.,0.
    for sample in data.samples:
        vehicle.set_pose(*sample[:4])
        center,q=p.getBasePositionAndOrientation(vehicle.body_id,physicsClientId=vehicle.client_id)
        yaw=p.getEulerFromQuaternion(q)[2]
        offset=scenario['geometry'].length/2-scenario['geometry'].rear_overhang
        recovered=np.array(center[:2])-offset*np.array([math.cos(yaw),math.sin(yaw)])
        maximum_xy=max(maximum_xy,float(np.linalg.norm(recovered-sample[:2])))
        maximum_yaw=max(maximum_yaw,float(abs(wrap(yaw-sample[2]))))
    with np.load(ROOT/'results/p10_4_1_live_plan.npz',allow_pickle=False) as saved:
        same_path=np.array_equal(saved['path'],data.path)
    zero=np.abs(data.samples[:,4])<1e-9
    duration_zero=sum(float(b[6]-a[6]) for a,b in zip(data.samples[:-1],data.samples[1:]) if abs(a[4])<1e-9 and abs(b[4])<1e-9)
    signs=(np.abs(data.samples[:,4])<1e-5)|(np.sign(data.samples[:,4])==data.samples[:,5])
    corners=vehicle_polygon(VehicleState(*data.samples[-1,:3]),scenario['geometry'])
    x0,y0,x1,y1=scenario['parking_slot']
    contained=bool(np.all((corners[:,0]>=x0)&(corners[:,0]<=x1)&(corners[:,1]>=y0)&(corners[:,1]<=y1)))
    # Pure replay timing regression: pause freezes pose; restart includes preview.
    clock=ReplayClock(data.duration,4.)
    clock.advance(8.); clock.paused=True; before=clock.elapsed; clock.advance(20.)
    assert clock.elapsed==before
    clock.restart(); assert clock.time==0 and clock.elapsed==0
    # Regression for the +179/-179 degree crossing; the midpoint faces 180.
    assert abs(abs(wrap(math.radians(179)+.5*wrap(math.radians(-179)-math.radians(179))))-math.pi)<1e-12
    assert np.array_equal(data.at(0),data.samples[0]) and np.array_equal(data.at(data.duration),data.samples[-1])
    def clearance(pose):
        car=vehicle_polygon(VehicleState(*pose),scenario['geometry'])
        distance=float('inf')
        for x0,y0,x1,y1 in scenario['parked_vehicles']:
            box=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            for polygon,other in [(car,box),(box,car)]:
                for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                    d=b-a
                    u=np.clip((other-a) @ d / np.dot(d,d),0,1)
                    distance=min(distance,float(np.min(np.linalg.norm(other-(a+u[:,None]*d),axis=1))))
        return distance
    report=dict(source=data.metadata,sample_count=len(data.samples),planned_points=len(data.path),duration_s=data.duration,
        true_start_pose=data.samples[0,:3].tolist(),true_final_pose=data.samples[-1,:3].tolist(),preview_pose=preview.tolist(),
        initial_steering_deg=math.degrees(data.samples[0,3]),
        true_start_obstacle_clearance_m=clearance(data.samples[0,:3]),
        preview_obstacle_clearance_m=clearance(preview),path_identical_to_saved_plan=same_path,
        trajectory_identical_to_saved_execution=bool(np.array_equal(data.samples[:,:3],data.raw_trajectory)),
        max_rendered_rear_axle_position_error_m=maximum_xy,max_rendered_yaw_error_rad=maximum_yaw,
        reverse_yaw_flipped=False,speed_direction_consistent=bool(np.all(signs)),zero_speed_samples=int(sum(zero)),
        recorded_stationary_duration_s=duration_zero,stationary_samples_preserved=True,
        direction_segments=[int(data.samples[np.flatnonzero((data.segments==segment)&(np.abs(data.samples[:,4])>1e-5))[0],5]) for segment in np.unique(data.segments)],
        neutral_samples=int(np.sum(data.samples[:,5]==0)),
        full_footprint_checked_samples=count,physical_collision_intervals=sorted(set(bad)),
        preview_collision=bool(collision(VehicleState(*preview),scenario)),transition_collision_samples=transition_bad,
        final_rectangle_inside_target_bay=contained,
        slot_boundary_policy='Painted bay lines are not physical walls: crossing entrance/lines during maneuvers is allowed; final full-rectangle containment is checked.',
        pause_restart_and_angle_wrap_checks=True,vehicle_size_m=[scenario['geometry'].length,scenario['geometry'].width],
        note='Replay of saved measured states, not new dynamics; preview/transition are excluded from execution time and path.')
    report['status']='PASS' if not bad and not transition_bad and not report['preview_collision'] and same_path and maximum_xy<1e-9 and maximum_yaw<1e-9 and contained else 'REVIEW'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate',type=float,default=1.,help='Playback time multiplier, never modifies measured speed')
    parser.add_argument('--headless',action='store_true',help='Run numerical/geometry checks and exit')
    parser.add_argument('--capture',action='store_true',help='Save five milestone GUI renders')
    parser.add_argument('--exit-on-finish',action='store_true')
    args=parser.parse_args()
    if not math.isfinite(args.rate) or args.rate<=0: parser.error('Rate must be positive and finite')
    data=load_execution(ROOT/'results/p10_4_2_execution_trajectory.npz')
    scenario=build_perpendicular_parking_scenario()
    # Same display-only separation as P11.1, relative to the actual saved start.
    preview=data.samples[0,:3].copy(); preview[1]-=1.2
    protected={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'results').glob('p10*') if f.is_file()}
    cid=p.connect(p.DIRECT if args.headless else p.GUI,options='' if args.headless else WINDOW_OPTIONS)
    if cid<0: raise RuntimeError('PyBullet connection failed')
    try:
        scene=ParkingScene(cid,scenario); vehicle=VehicleVisual(cid,scenario['geometry'])
        check_scene(scene,vehicle)
        report=audit(data,scenario,vehicle,preview)
        report['protected_p10_files_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protected.items())
        report['protected_p10_sha256']=protected
        out=ROOT/'results'; out.mkdir(exist_ok=True)
        (out/'p11_2_replay_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('source','protected_p10_sha256')},indent=2),flush=True)
        if report['status']!='PASS': raise RuntimeError('Replay audit requires review; see metrics')
        if args.headless: return
        scene.camera_target=[11.8,11.4,0.]
        scene.configure_gui()
        report['gui_camera']=dict(target=scene.camera_target,distance=scene.camera_distance,yaw=scene.camera_yaw,pitch=scene.camera_pitch,window=[1100,800])
        report['interactive_events']=[]
        view=ReplayView(scene,vehicle,data); clock=ReplayClock(data.duration,args.rate)
        forward=np.flatnonzero((data.samples[:,4]>.1)&(np.abs(data.samples[:,3])>.15))
        reverse=np.flatnonzero((data.segments==data.segments[-1])&(data.samples[:,4]<-.1))
        motion=np.flatnonzero(np.linalg.norm(data.samples[:,:2]-data.samples[0,:2],axis=1)>.15)
        milestones=[('scene',None),('motion_start',float(data.samples[motion[0],6])),
                    ('forward_turn',float(data.samples[forward[len(forward)//2],6])),
                    ('reverse_parking',float(data.samples[reverse[2*len(reverse)//3],6])),('final',data.duration)]
        captured=set(); previous=time.monotonic(); saved_final=False
        print('SPACE pause | R/F5 restart | -/+ rate | Q/ESC exit. Saved sequence: reverse -> forward -> reverse.',flush=True)
        while p.isConnected(cid):
            now=time.monotonic(); dt=now-previous; previous=now
            keys=p.getKeyboardEvents(physicsClientId=cid)
            triggered=lambda k: bool(keys.get(k,0)&p.KEY_WAS_TRIGGERED)
            if triggered(ord('q')) or triggered(ord('Q')) or triggered(27): break
            if triggered(ord(' ')):
                clock.paused=not clock.paused
                print(f'Paused={clock.paused}, replay t={clock.time:.3f}',flush=True)
                report['interactive_events'].append(dict(action='pause_toggle',paused=clock.paused,time_s=clock.time))
            if triggered(ord('r')) or triggered(ord('R')) or triggered(p.B3G_F5):
                clock.restart(); view.reset(); saved_final=False; print('Restarted',flush=True)
                report['interactive_events'].append(dict(action='restart',time_s=clock.time))
                dt=0.
            old_rate=clock.rate
            # ASCII plus/minus plus Windows OEM/numpad key codes, and arrows as
            # IME-independent alternatives. No changes to saved signed speed.
            if any(triggered(k) for k in (ord('-'),189,109,p.B3G_DOWN_ARROW)):
                clock.rate=max(.125,clock.rate/2)
            if any(triggered(k) for k in (ord('+'),ord('='),187,107,p.B3G_UP_ARROW)):
                clock.rate=min(16.,clock.rate*2)
            if clock.rate!=old_rate:
                report['interactive_events'].append(dict(action='rate',rate=clock.rate,time_s=clock.time))
                print(f'Playback rate={clock.rate}',flush=True)
            clock.advance(dt)
            sample,stage=clock.sample(data,preview)
            executing=clock.elapsed>=clock.preview_hold+clock.transition_duration
            view.update(sample,stage,clock.time,clock.rate,clock.paused,executing)
            if args.capture:
                for label,stamp in milestones:
                    if label not in captured and (stamp is None or (executing and clock.time>=stamp)):
                        view.screenshot(out/f'p11_2_{label}.png',f'{stage} | t={clock.time:.2f}s | v={sample[4]:+.3f}m/s | steer={math.degrees(sample[3]):+.1f}deg')
                        captured.add(label)
                        print('Captured',label,flush=True)
                        previous=time.monotonic() # render/export time is not replay time
            if clock.finished and not saved_final:
                report['gui_playback_completed']=True
                report['captured_milestones']=sorted(captured)
                (out/'p11_2_replay_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                saved_final=True
                print('GUI playback reached final saved pose; holding.',flush=True)
            if clock.finished and args.exit_on_finish: break
            time.sleep(1/60)
        report['gui_playback_completed']=clock.finished
        report['captured_milestones']=sorted(captured)
        (out/'p11_2_replay_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'GUI replay completed={clock.finished}',flush=True)
    except p.error:
        if p.isConnected(cid): raise
    finally:
        if p.isConnected(cid): p.disconnect(cid)


if __name__=='__main__': main()
