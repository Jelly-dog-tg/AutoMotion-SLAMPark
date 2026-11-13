"""P11.7.9: one physical car, fresh local dual SLAM, map-selected bay, planning/control.

--demo performs a short encoder-terminated exploration then runs the same
P/plan/A workflow; it is not a recorded path and reads no historical map.
"""
import argparse,json,math,sys,time,threading,hashlib
from pathlib import Path
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pybullet as p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.simulation.pybullet.physics_vehicle import PhysicsVehicle
from parking_planner.simulation.pybullet.scene import ParkingScene
from parking_planner.simulation.pybullet.hud import EdgeStatusBars
from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS
from parking_planner.simulation.pybullet.parking_state_machine import ParkingStateMachine
from parking_planner.sensing.dual_lidar import DualLidar360
from parking_planner.sensing.lidar_2d import LidarConfig
from parking_planner.sensing.encoder_imu import EncoderIMU
from parking_planner.slam.fresh_local_slam import FreshLocalSLAM
from parking_planner.planning.local_bay import recognize_bay,LocalSweepGrid
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import resample_path
from parking_planner.planning.robust_clearance import CellClearance
from parking_planner.vehicle.ackermann import VehicleState
from run_p11_7_4_5_live_slam_hybrid_astar_physical_validation import planned_segments
from run_p11_7_6_robust_hybrid_astar_physical_validation import path_clearance,indexes,clean
from run_p11_7_3_goal_blocker_source_analysis import area,clip_rectangle

OUT=ROOT/'results'/'p11_7_9';DT=1/240
LIMITS=dict(position_m=.10,heading_deg=2.,stop_speed_mps=.01,hold_s=2.,execution_margin_m=.13,
    reverse_speed_mps=.10,forward_speed_mps=.25,maximum_physics_s=240.,max_tracking_m=.10,max_heading_deg=2.)
COLUMNS=['time_s','estimate_x','estimate_y','estimate_yaw','encoder_speed','actual_steer','gear','target_speed','motor_speed',
    'cross_track','heading_error','segment','eval_world_x','eval_world_y','eval_world_yaw','eval_speed','physical_contacts']


def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def wrap(x):return math.atan2(math.sin(x),math.cos(x))


def independent_audit(trace,phases,path,actual_local,actual_yaw,geometry,obstacles,recognized_bay):
    """Post-run only: truth is never returned to the live estimator/controller."""
    moving=np.flatnonzero(np.abs(trace[:,15])>.01)
    events=[]
    for i in np.flatnonzero(np.diff(trace[:,6])!=0)+1:
        events.append(dict(time_s=float(trace[i,0]),from_gear=int(trace[i-1,6]),to_gear=int(trace[i,6]),
            actual_speed_before_mps=float(trace[i-1,15]),encoder_speed_before_mps=float(trace[i-1,4])))
    errors=[];heading=[]
    auto=np.array([x not in ('EXPLORING','STOP_FOR_PLAN','FINAL_STOP')for x in phases])
    if len(path)>1:
        a=path[:-1,:2];edge=np.diff(path[:,:2],axis=0);ds2=np.sum(edge*edge,axis=1)
        dyaw=np.arctan2(np.sin(np.diff(path[:,2])),np.cos(np.diff(path[:,2])))
        for start in range(0,len(trace),1024):
            pos=actual_local[start:start+1024];u=np.clip(np.sum((pos[:,None]-a)*edge,axis=2)/np.maximum(ds2,1e-15),0,1)
            closest=a+u[:,:,None]*edge;d2=np.sum((closest-pos[:,None])**2,axis=2);j=np.argmin(d2,axis=1);k=np.arange(len(pos))
            errors.extend(np.sqrt(d2[k,j]));refyaw=path[j,2]+u[k,j]*dyaw[j]
            delta=actual_yaw[start:start+len(pos)]-refyaw;heading.extend(np.abs(np.degrees(np.arctan2(np.sin(delta),np.cos(delta)))))
    body_index=CellClearance(np.zeros((1,1),bool),.05,[0,0],geometry);boxes=np.array(obstacles);geometric_samples=0
    for row in trace:
        if np.any(body_index.exact_distances(row[[12,13,14]],boxes)<=1e-10):geometric_samples+=1
    recognized_inside=None
    if recognized_bay and recognized_bay.get('slot_polygon')is not None:
        slot=np.array(recognized_bay['slot_polygon']);body=body_index.polygon(np.r_[actual_local[-1],actual_yaw[-1]])
        e=np.roll(slot,-1,axis=0)-slot;delta=body[:,None]-slot[None]
        cross=e[None,:,0]*delta[:,:,1]-e[None,:,1]*delta[:,:,0]
        recognized_inside=bool(np.all(cross>=-1e-9)or np.all(cross<=1e-9))
    return dict(actual_gear_events=events,all_actual_shifts_below_stop_threshold=all(abs(e['actual_speed_before_mps'])<=.01 for e in events),
        last_moving_actual_gear=int(trace[moving[-1],6])if len(moving)else None,
        physical_reference_ADE_m=float(np.mean(np.array(errors)[auto]))if errors and auto.any()else None,
        physical_reference_max_position_m=float(np.max(np.array(errors)[auto]))if errors and auto.any()else None,
        physical_reference_max_heading_deg=float(np.max(np.array(heading)[auto]))if heading and auto.any()else None,
        physical_path_terminal_error_m=float(np.linalg.norm(actual_local[-1]-path[-1,:2]))if len(path)else None,
        distance_definition='Nearest rear-axle reference polyline distance during automatic phases; NOT timestamp-aligned. Terminal error separately compares final actual rear axle to last path node.',
        geometric_obstacle_overlap_samples=geometric_samples,full_body_inside_recognized_region=recognized_inside,
        pose_envelope_actual_max_m=float(np.max(np.linalg.norm(actual_local-trace[:,1:3],axis=1))),
        independent_evaluation_only=True)


class Experiment:
    def __init__(self,args):
        self.args=args;self.scenario=build_perpendicular_parking_scenario();self.geometry=self.scenario['geometry']
        # Historical final state is used ONLY to create the physical initial condition.
        with np.load(ROOT/'results/p11_7_6_robust_physical_trace.npz',allow_pickle=False)as a:
            names=list(a['columns']);row=a['trace'][-1];initial=[float(row[names.index(k)])for k in ('rear_x','rear_y','yaw_rad')]
            steer=float(row[names.index('actual_steer')]);gear=int(row[names.index('gear')])
        self.cid=p.connect(p.GUI if args.gui else p.DIRECT,options=WINDOW_OPTIONS if args.gui else '')
        p.setTimeStep(DT,physicsClientId=self.cid);p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=self.cid)
        self.scene=ParkingScene(self.cid,self.scenario);self.car=PhysicsVehicle(self.cid,self.geometry,initial);self.car.direction=gear
        self.originals={};self.forbidden=[]
        for name in ('resetBasePositionAndOrientation','resetBaseVelocity','resetJointState','applyExternalForce','applyExternalTorque'):
            fn=getattr(p,name);self.originals[name]=fn
            def guard(*a,_fn=fn,_name=name,**kw):
                body=a[0]if a else kw.get('bodyUniqueId',kw.get('objectUniqueId'))
                if body==self.car.body:self.forbidden.append(_name);raise RuntimeError('Forbidden direct vehicle motion')
                return _fn(*a,**kw)
            setattr(p,name,guard)
        self.imu=EncoderIMU(self.car);headings=[]
        for _ in range(480):
            self.car.command(0.,steer,DT);p.stepSimulation(physicsClientId=self.cid);headings.append(self.imu.sample(DT)['heading_rad'])
        zero=math.atan2(np.mean(np.sin(headings[-240:])),np.mean(np.cos(headings[-240:])))
        self.dual=DualLidar360(self.cid,self.car.body,LidarConfig(num_beams=720),rear_offset=(.6,0.));self.slam=FreshLocalSLAM(self.dual,zero)
        # Isolated offline evaluator's origin; NEVER read by SLAM/planner/controller.
        actual=self.car.state();self.eval_origin=np.array([actual[k]for k in ('x','y','yaw')])
        self.t=0.;self.step_count=0;self.mode='EXPLORING';self.action=None;self.speed=0.;self.steer=steer
        self.rows=[];self.phases=[];self.scan_packets=[];self.scan_times=[];self.machine=None;self.path=np.empty((0,6));self.goal=None;self.bay=None
        self.planning_events=[];self.collisions=[];self.last_contacts=set();self.loss_reason=None;self.terminal=False;self.stop_stable=0.
        self.poses=[];self.goal_clicks=[];self.window=None;self.hud=None;self.plan_grid=None;self.plan_report=None
        self.scan()
        if args.gui and hasattr(p,'COV_ENABLE_KEYBOARD_SHORTCUTS'):p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS,0,physicsClientId=self.cid)
        if args.gui:
            self.scene.configure_gui();self.hud=EdgeStatusBars(self.cid)
            if not args.no_map_window:
                plt.switch_backend('TkAgg')
                from run_p11_6_4_live_slam import LiveMapWindow
                self.window=LiveMapWindow(self.slam.mapper)
                self.window.fig.canvas.mpl_connect('button_press_event',self.click)
                self.window.fig.canvas.mpl_connect('key_press_event',self.map_key)
                self.scan_artist=self.window.ax.scatter([],[],s=2,c='deepskyblue',alpha=.5)
                self.goal_artist,=self.window.ax.plot([],[],c='magenta')
                self.plan_artist,=self.window.ax.plot([],[],c='royalblue',lw=1)

    def click(self,event):
        if self.mode not in('EXPLORING','PLAN_READY')or event.inaxes!=self.window.ax:return
        if event.xdata is None:return
        if len(self.goal_clicks)==2:self.goal_clicks=[]
        self.goal_clicks.append([event.xdata,event.ydata])
        if len(self.goal_clicks)==2:
            center=np.array(self.goal_clicks[0]);v=np.array(self.goal_clicks[1])-center
            if np.linalg.norm(v)<.2:return
            yaw=math.atan2(v[1],v[0]);d=v/np.linalg.norm(v);rear=center-(self.geometry.length/2-self.geometry.rear_overhang)*d
            self.goal=np.r_[rear,yaw];self.bay=dict(goal=self.goal.tolist(),source='two user map clicks: vehicle center then body front',slot_polygon=None)
            poly=self.body(self.goal);self.goal_artist.set_data(*np.vstack([poly,poly[0]]).T);self.window.fig.canvas.draw_idle()

    def map_key(self,event):
        if event.key=='enter':self.action='PLAN'
        elif event.key and event.key.lower()in('p','a','r','q'):self.action={'p':'PLAN','a':'AUTO','r':'EXPLORE','q':'EXIT'}[event.key.lower()]

    def body(self,pose):return CellClearance(np.zeros((1,1),bool),.05,[0,0],self.geometry).polygon(pose)

    def scan(self):
        # Hardware-only world coordinates for physically correct rays, not estimation input.
        packet=self.dual.capture(self.car.state());self.slam.observe(packet,self.t)
        self.scan_packets.append(packet);self.scan_times.append(self.t);self.poses.append(self.slam.pose.copy())
        if self.window is not None:
            self.window.update(self.slam.mapper,self.poses,dict(icp=sum(h['accepted']for h in self.slam.history),odom=sum(not h['accepted']for h in self.slam.history)))
            q=self.slam.pose;c,s=math.cos(q[2]),math.sin(q[2]);pts=packet['points_local']@np.array([[c,s],[-s,c]])+q[:2]
            self.scan_artist.set_offsets(pts);self.window.fig.canvas.flush_events()

    def feedback(self,m):
        q=self.slam.pose
        return dict(x=q[0],y=q[1],yaw=q[2],speed=m['speed_mps'],steering=m['steering_rad'],direction=self.car.direction)

    def safety_margin(self):
        # Explicit engineering/probabilistic budget; not a claimed hard certificate.
        views=max(1,len({(round(q[0]/.20),round(q[1]/.20))for q in self.poses}))
        sigma=.015/math.sqrt(min(views,3));range_budget=3*sigma
        # Plan for up to 3 m of well-supported parallel-wall dead reckoning.
        pose_budget=.02+.02*3.
        yaw_budget=2*math.hypot(3.5,.9)*math.sin(math.radians(.1)/2)
        margin=.13+range_budget+self.slam.mapper.resolution/math.sqrt(2)+pose_budget+yaw_budget
        return margin,dict(execution_m=.13,range_three_sigma_m=range_budget,independent_view_bins=views,
            local_pose_engineering_budget_m=pose_budget,relative_imu_corner_budget_m=yaw_budget,
            cell_center_quantization_m=self.slam.mapper.resolution/math.sqrt(2),total_m=margin,
            caveat='Engineering local uncertainty budget; Gaussian 3-sigma is per-measurement probabilistic and NOT a deterministic or joint confidence certificate. Slip/model error is monitored using scan innovations, IMU and final independent evaluation.')

    def plan(self):
        if self.goal is None:
            self.bay=recognize_bay(self.slam.mapper,self.slam.pose,self.geometry);self.goal=np.array(self.bay['goal'])
        margin,budget=self.safety_margin();substep=.025;radius=math.hypot(3.5,.9)
        half=substep/2+2*radius*math.sin(substep/self.geometry.wheelbase*math.tan(math.radians(35))/4)
        grid=LocalSweepGrid(self.slam.mapper,self.geometry,margin,half)
        config=replace(self.scenario['planner_config'],xy_resolution=.08,yaw_resolution=math.radians(2),step_size=.20,steer_samples=7,
            goal_xy_tolerance=.045,goal_yaw_tolerance=math.radians(.7),heuristic_weight=1.8,heading_heuristic_weight=.4)
        planner=HybridAStarPlanner(grid=grid,geometry=self.geometry,config=config)
        self.planning_events.append(dict(time_s=self.t,start=self.slam.pose.tolist(),goal=self.goal.tolist(),budget=budget))
        print('Fresh goal:',self.goal,'bay',self.bay,'margin',margin,flush=True)
        if grid.is_collision(VehicleState(*self.slam.pose),planner.local_x,planner.local_y):raise ValueError('Current complete body plus local margin blocked; continue observing')
        if grid.is_collision(VehicleState(*self.goal),planner.local_x,planner.local_y):raise ValueError('Selected goal body/uncertainty envelope blocked; explore or select again')
        done=threading.Event()
        def heartbeat():
            while not done.wait(15):print('Fresh Hybrid A* expanded',planner.last_iterations,flush=True)
        th=threading.Thread(target=heartbeat,daemon=True);th.start()
        try:raw=planner.plan(VehicleState(*self.slam.pose),VehicleState(*self.goal))
        finally:done.set();th.join(timeout=1)
        if not raw:raise ValueError('NO_ROBUST_PATH; continue exploring or choose another goal')
        dense=resample_path(raw,spacing=.025)
        path=np.array([[q.x,q.y,q.yaw,q.steering,q.direction,math.tan(q.steering)/self.geometry.wheelbase]for q in dense])
        if path[-1,4]!=-1:raise ValueError('Planner terminal segment is forward; reverse-entry requirement not satisfied')
        lookup=indexes(dict(states=self.slam.mapper.states),self.slam.mapper.config,grid,self.geometry)
        sweep=path_clearance(path,lookup,margin)
        if not sweep['all_continuous_segments_robust']:raise ValueError('Continuous swept-body margin failed')
        self.path=path;self.plan_grid=grid;self.plan_report=dict(iterations=planner.last_iterations,config=asdict(config),budget=budget,continuous=sweep)
        self.mode='PLAN_READY';self.speed=0.;self.loss_reason=None
        if self.window:
            self.plan_artist.set_data(path[:,0],path[:,1]);poly=self.body(self.goal);self.goal_artist.set_data(*np.vstack([poly,poly[0]]).T);self.window.fig.canvas.flush_events()
        print('Plan ready:',len(path),'points; press A to drive',flush=True)

    def start_auto(self):
        if self.mode!='PLAN_READY':return
        if not len(self.path)or self.goal is None or np.linalg.norm(self.path[-1,:2]-self.goal[:2])>.06 or abs(wrap(self.path[-1,2]-self.goal[2]))>math.radians(1.):
            self.mode='STOP_FOR_PLAN';self.action=None;return
        self.machine=ParkingStateMachine(planned_segments(self.path,self.geometry.wheelbase),self.geometry.wheelbase)
        self.machine.STOP_SPEED=.01;self.machine.FINAL_HOLD=2.
        # Keep original Stanley/PID and tracker algorithms; cap the PID's target
        # and motor output at the requested low reverse speed, no gain changes.
        for tracker,segment in zip(self.machine.trackers,self.machine.segments):
            limit=.10 if segment.direction<0 else .25
            compute=tracker.pid.compute
            tracker.pid.compute=lambda target,speed,dt,fn=compute,cap=limit:fn(float(np.clip(target,-cap,cap)),speed,dt)
            update=tracker.update
            def limited(state,dt,fn=update,cap=limit):
                motor,steer,error,target=fn(state,dt)
                return float(np.clip(motor,-cap,cap)),steer,error,float(np.clip(target,-cap,cap))
            tracker.update=limited
        self.mode='AUTO';print('AUTO starts with estimated pose only',flush=True)

    def keys(self):
        if not self.args.gui:return
        keys=p.getKeyboardEvents(physicsClientId=self.cid)
        hit=lambda k:bool(keys.get(k,0)&p.KEY_WAS_TRIGGERED)
        if hit(ord('p'))or hit(ord('P')):self.action='PLAN'
        if hit(ord('a'))and self.mode=='PLAN_READY':self.action='AUTO'
        if hit(ord('r')):self.action='EXPLORE'
        if hit(27)or hit(ord('q')):self.action='EXIT'
        if hit(32):self.speed=0.;self.action='EMERGENCY'if self.mode=='AUTO'else None
        if self.mode=='EXPLORING':
            if hit(ord('w')):self.speed=min(.25,self.speed+.05)
            if hit(ord('s')):self.speed=max(-.12,self.speed-.05)
            if hit(ord('a')):self.steer=min(math.radians(35),self.steer+math.radians(3))
            if hit(ord('d')):self.steer=max(-math.radians(35),self.steer-math.radians(3))
            if hit(ord('c')):self.steer=0.

    def run(self):
        last_phase='';m=self.imu.sample(DT);brake_time=0.;demo_stage=0;planning_failures=0
        while not self.terminal and self.t<LIMITS['maximum_physics_s']:
            wall=time.monotonic();self.keys();state=self.feedback(m)
            if self.args.demo and self.mode=='EXPLORING':
                if self.slam.pose[0]<.8 and demo_stage==0:self.speed=.20;self.steer=0.
                else:self.action='PLAN';demo_stage=1
            if self.action in('EXIT','EMERGENCY'):
                self.loss_reason='USER_STOP';self.speed=0.;self.mode='FINAL_STOP';self.action=None
            if self.action=='EXPLORE':self.mode='EXPLORING';self.machine=None;self.action=None;self.speed=0.
            if self.action=='PLAN'and self.mode in('EXPLORING','PLAN_READY'):self.mode='STOP_FOR_PLAN';self.speed=0.;self.action=None
            if self.action=='AUTO':self.start_auto();self.action=None
            motor=self.speed if self.mode=='EXPLORING'else 0.;steer=self.steer;target=motor;err=dict(cross_track=0.,heading_error=0.)
            if self.mode=='STOP_FOR_PLAN':
                if abs(m['speed_mps'])<.01:brake_time+=DT
                else:brake_time=0.
                if brake_time>=.5:
                    self.scan()
                    try:self.plan()
                    except ValueError as exc:
                        print('Planning refused:',exc,flush=True)
                        planning_failures+=1;self.loss_reason=str(exc);self.goal=None
                        if self.args.demo:
                            self.mode='FINAL_STOP'
                        else:self.mode='EXPLORING'
                    brake_time=0.
                    if self.args.demo and self.mode=='PLAN_READY':self.start_auto()
            if self.mode=='AUTO':
                motor,steer,err,target=self.machine.update(state,DT,self.t)
                if self.machine.phase!=last_phase:
                    print(f't={self.t:.2f} {self.machine.phase} segment={self.machine.index+1}',flush=True);last_phase=self.machine.phase
                if self.machine.finished:self.mode='COMPLETE'if self.machine.failure is None else'FAILED';self.terminal=True
                if abs(err['cross_track'])>.10 or abs(math.degrees(err['heading_error']))>2.:
                    self.machine.emergency('TRACKING_ERROR_ENVELOPE',self.t,state)
                if self.slam.lost>8 or self.slam.dead_reckoning_distance>3.:self.machine.emergency('LOCALIZATION_LOSS_OR_DISTANCE_BOUND',self.t,state)
                corner_error=err.get('position_error',abs(err['cross_track']))+2*math.hypot(3.5,.9)*math.sin(abs(err['heading_error'])/2)
                if corner_error>.13:self.machine.emergency('COMBINED_EXECUTION_ENVELOPE',self.t,state)
                # Actual localization-based body check against current live labels.
                safety=LocalSweepGrid(self.slam.mapper,self.geometry,self.safety_margin()[0]-.13,.001)
                if safety.is_collision(VehicleState(*self.slam.pose),None,None):self.machine.emergency('LOCAL_MAP_BODY_VIOLATION',self.t,state)
            if self.mode=='FINAL_STOP':
                motor=0.;target=0.
                if abs(m['speed_mps'])<.01:self.stop_stable+=DT
                else:self.stop_stable=0.
                if self.stop_stable>=2.:self.terminal=True
            # Safe short-horizon exploration, using fresh estimated pose only.
            if self.mode=='EXPLORING'and abs(motor)>0:
                q=self.slam.pose.copy();q[:2]+=motor*.5*np.array([math.cos(q[2]),math.sin(q[2])])
                if LocalSweepGrid(self.slam.mapper,self.geometry,.13,0.).is_collision(VehicleState(*q),None,None):motor=0.;self.speed=0.
            self.car.command(motor,steer,DT);p.stepSimulation(physicsClientId=self.cid);self.t+=DT;self.step_count+=1
            m=self.imu.sample(DT);self.slam.predict(m)
            if self.step_count%48==0:
                self.scan()
                if self.step_count%480==0:print(f'Local t={self.t:.1f}s mode={self.mode} estimate={self.slam.pose.tolist()}',flush=True)
            # Truth is recorded for independent post-run evaluation only.
            actual=self.car.state();contacts={c[2]for c in p.getContactPoints(bodyA=self.car.body,physicsClientId=self.cid)if c[2]in self.scene.obstacle_ids and(c[8]<=0 or c[9]>0)}
            for body in contacts-self.last_contacts:self.collisions.append(dict(time_s=self.t,body=body))
            self.last_contacts=contacts
            if contacts:
                self.loss_reason='PHYSICAL_CONTACT';self.mode='FINAL_STOP';self.speed=0.
            self.rows.append([self.t,*self.slam.pose,m['speed_mps'],m['steering_rad'],self.car.direction,target,motor,err['cross_track'],err['heading_error'],self.machine.index if self.machine else -1,
                actual['x'],actual['y'],actual['yaw'],actual['speed'],len(contacts)])
            self.phases.append(self.machine.phase if self.machine and self.mode in('AUTO','COMPLETE','FAILED')else self.mode)
            if self.hud and self.step_count%12==0:
                phase=self.machine.phase if self.mode=='AUTO'else self.mode
                self.hud.update(f'FRESH LOCAL SLAM | {phase} | t={self.t:.1f}s | gear={self.car.direction:+d}',
                    f'est=({self.slam.pose[0]:+.2f},{self.slam.pose[1]:+.2f}) yaw={math.degrees(self.slam.pose[2]):+.1f}deg | v={m["speed_mps"]:+.3f} steer={math.degrees(m["steering_rad"]):+.1f}deg',
                    'W/S speed | A/D steer | C center | SPACE brake | P plan | A auto after plan | R explore | Q/ESC stop',
                    'BLUE plan / CYAN live rays / map clicks: center then body front; ENTER confirm')
            if self.args.gui and not self.args.fast:time.sleep(max(0.,DT-(time.monotonic()-wall)))
        if not self.terminal:self.loss_reason='TIMEOUT';self.mode='FAILED'
        if self.machine and self.machine.failure:self.loss_reason=self.machine.failure
        self.car.command(0.,self.steer,DT)

    def save(self,protected):
        trace=np.array(self.rows);final=trace[-1];o=self.eval_origin;c,s=math.cos(o[2]),math.sin(o[2]);rotation=np.array([[c,s],[-s,c]])
        actual_local=(trace[:,12:14]-o[:2])@rotation.T;actual_yaw=np.arctan2(np.sin(trace[:,14]-o[2]),np.cos(trace[:,14]-o[2]))
        position=None;heading=None;goal_world=None;inside=False
        if self.goal is not None:
            position=float(np.linalg.norm(actual_local[-1]-self.goal[:2]));heading=abs(math.degrees(wrap(actual_yaw[-1]-self.goal[2])))
            goal_world=np.r_[self.goal[:2]@rotation+o[:2],wrap(self.goal[2]+o[2])]
        poly=self.body(np.r_[final[12:14],final[14]]);bay=self.scenario['parking_slot'];inside=bool(area(clip_rectangle(poly,bay))>=self.geometry.length*self.geometry.width-1e-8)
        last480=trace[-480:];hold=bool(len(last480)==480 and np.all(np.abs(last480[:,15])<=.01))
        reverse=bool(final[6]==-1 and any(trace[:,15]<-.01));forward=bool(any(trace[:,15]>.01))
        audit=independent_audit(trace,self.phases,self.path,actual_local,actual_yaw,self.geometry,self.scenario['parked_vehicles'],self.bay)
        checks=dict(physical_motion=bool(np.linalg.norm(np.diff(trace[:,12:14],axis=0),axis=1).sum()>.5),forward_motion=forward,
            automatic_complete=bool(self.mode=='COMPLETE'and self.machine and self.machine.failure is None and self.machine.phase=='COMPLETE'),full_body_inside_original_slot=inside,
            final_position=position is not None and position<=.10,final_heading=heading is not None and heading<=2.,
            final_actual_reverse=reverse,stopped=abs(final[15])<=.01,hold_2s=hold,no_collision=not self.collisions,
            no_direct_pose_changes=not self.forbidden,continuous_path_verified=bool(self.plan_report and self.plan_report['continuous']['all_continuous_segments_robust']),
            safe_shift_events=bool(self.machine and all(abs(e['engagement_speed_mps'])<=.01 for e in self.machine.shifts)))
        checks.update(all_actual_shifts_stopped=audit['all_actual_shifts_below_stop_threshold'],geometric_no_collision=audit['geometric_obstacle_overlap_samples']==0,
            last_motion_reverse=audit['last_moving_actual_gear']==-1,recognized_bay_full_body=audit['full_body_inside_recognized_region']is not False,
            physical_reference_consistency=audit['physical_reference_max_position_m']is not None and audit['physical_reference_max_position_m']<=.10 and audit['physical_reference_max_heading_deg']<=2.,
            local_pose_budget_validated=audit['pose_envelope_actual_max_m']<=.08)
        status='COMPLETE_PARKING_PASS'if all(checks.values())else'FAILED'
        result=dict(stage='P11.7.9',status=status,checks=checks,limits=LIMITS,reason=self.loss_reason,
            independent_physical_audit=audit,
            fresh_map=True,historical_online_map_loaded=False,old_fixed_trajectory_loaded=False,vehicle_created_once=True,
            local_origin_definition='Initial rear axle center and initial body heading; algorithm pose=(0,0,0). World origin transform stored below is evaluation-only.',
            actual_terminal_world_initialization_source='P11.7.6 final physical trace only; not fed to SLAM/map/planning/controller',
            goal=self.goal,bay_recognition=self.bay,planning=self.plan_report,planning_events=self.planning_events,
            final_position_error_m=position,final_heading_error_deg=heading,final_speed_mps=float(final[15]),final_gear=int(final[6]),
            body_inside_original_slot=inside,hold_2s=hold,physical_collision_count=len(self.collisions),collision_events=self.collisions,
            total_time_s=self.t,actual_distance_m=float(np.linalg.norm(np.diff(trace[:,12:14],axis=0),axis=1).sum()),
            max_localization_error_m=float(np.max(np.linalg.norm(actual_local-trace[:,1:3],axis=1))),
            max_localization_heading_deg=float(np.max(np.abs(np.degrees(np.arctan2(np.sin(actual_yaw-trace[:,3]),np.cos(actual_yaw-trace[:,3])))))),
            forward_time_s=float(np.sum(trace[:,15]>.01)*DT),reverse_time_s=float(np.sum(trace[:,15]<-.01)*DT),
            state_transitions=self.machine.events if self.machine else[],shift_events=self.machine.shifts if self.machine else[],
            scan_count=len(self.scan_packets),distinct_observation_bins=len({(round(q[0]/.2),round(q[1]/.2))for q in self.poses}),
            accepted_scan_matches=sum(h['accepted']for h in self.slam.history),failed_scan_matches=sum(not h['accepted']for h in self.slam.history[1:]),
            localization_fallback=dict(maximum_allowed_distance_m=3.,uncertainty_growth_m_per_m=.02,maximum_observed_distance_m=max(h['dead_reckoning_distance_m']for h in self.slam.history),degenerate_scan_count=sum(h['degenerate_streak']>0 for h in self.slam.history)),
            lidar_configuration=dict(num_beams=720,fov_deg=270.,max_range_m=12.,range_std_m=.015,mounts=self.dual.mounts.tolist()),
            software_parameter_changes='Only new planner configuration, stricter STOP_SPEED=.01 and PID target/motor speed ceilings; original model/controllers unchanged.',
            historical_sha256=protected,protected_files_unchanged=all(sha(ROOT/f)==digest for f,digest in protected.items()))
        OUT.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(OUT/'p11_7_9_local_map.npz',**self.slam.mapper.arrays(),estimated_poses=np.array(self.poses),scan_timestamps_s=np.array(self.scan_times),
            raw_dual_ranges_m=np.array([[q['raw'][i]['ranges_m']for i in range(2)]for q in self.scan_packets]),raw_dual_valid=np.array([[q['raw'][i]['valid']for i in range(2)]for q in self.scan_packets]),
            raw_self_hits=np.array([[q['raw'][i]['self_hits']for i in range(2)]for q in self.scan_packets]),
            mounts=self.dual.mounts,raw_angles_rad=self.scan_packets[0]['raw'][0]['angles_rad'],slam_history=np.array(json.dumps(clean(self.slam.history))))
        np.savez_compressed(OUT/'p11_7_9_parking_path.npz',path=self.path,columns=np.array(['x','y','yaw','steering','gear','curvature']),goal=np.array([])if self.goal is None else self.goal,interval_gear=self.path[1:,4].astype(np.int8),
            speed_ceiling_mps=np.where(self.path[:,4]<0,-.10,.25),gear_semantics=np.array('Row i gear applies to interval i-1 to i. Row0 is original planner start label, not a moving interval. Speed ceiling is configured, not recorded execution.'))
        np.savez_compressed(OUT/'p11_7_9_physical_trace.npz',trace=trace,columns=np.array(COLUMNS),phase=np.array(self.phases),actual_local_evaluation=actual_local,
            actual_yaw_evaluation=actual_yaw,evaluation_world_origin=o)
        (OUT/'p11_7_9_metrics.json').write_text(json.dumps(clean(result),indent=2,ensure_ascii=False),encoding='utf-8')
        plt.switch_backend('Agg');fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained');ext=[-12,12,-12,12]
        for ax in axes[0,:2]:
            ax.imshow(self.slam.mapper.states+1,origin='lower',extent=ext,cmap=matplotlib.colors.ListedColormap(['gray','white','black']),vmin=0,vmax=2)
            ax.plot(trace[:,1],trace[:,2],c='green',label='Estimated');ax.plot(*actual_local.T,c='cyan',lw=.7,label='Actual (evaluation)')
            if len(self.path):ax.plot(self.path[:,0],self.path[:,1],c='blue',label='Fresh Hybrid A*')
            ax.set(xlabel='Local X (m)',ylabel='Local Y (m)',aspect='equal')
        axes[0,0].set_title('Fresh SLAM; no historical online map');axes[0,0].legend(fontsize=7)
        axes[0,1].set(xlim=(-9,4),ylim=(-4,4),title='Plan / real physical trajectory')
        for q,color in [(np.r_[actual_local[-1],actual_yaw[-1]],'green')]+([]if self.goal is None else[(self.goal,'red')]):
            poly=self.body(q);axes[0,1].plot(*np.vstack([poly,poly[0]]).T,c=color)
        if self.bay and self.bay.get('slot_polygon')is not None:
            poly=np.array(self.bay['slot_polygon']);axes[0,1].plot(*np.vstack([poly,poly[0]]).T,c='goldenrod')
        view=p.computeViewMatrixFromYawPitchRoll([10.,12.,0],17,35,-62,0,2);proj=p.computeProjectionMatrixFOV(55,1.375,.1,100)
        if p.isConnected(self.cid):
            image=np.asarray(p.getCameraImage(1100,800,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_BULLET_HARDWARE_OPENGL if self.args.gui else p.ER_TINY_RENDERER,physicsClientId=self.cid)[2]).reshape(800,1100,4)
            axes[0,2].imshow(image);axes[0,2].set_title('Actual final physical state')
        else:axes[0,2].text(.1,.5,'GUI closed; recorded physical trace retained',transform=axes[0,2].transAxes)
        axes[0,2].axis('off')
        axes[1,0].plot(trace[:,0],trace[:,15],label='Actual speed');axes[1,0].plot(trace[:,0],trace[:,7],label='Target');axes[1,0].legend();axes[1,0].set(ylabel='m/s',xlabel='Time (s)')
        axes[1,1].plot(trace[:,0],np.degrees(trace[:,5]));axes[1,1].set(xlabel='Time (s)',ylabel='Steer (deg)')
        axes[1,2].plot(trace[:,0],trace[:,9],label='CTE m');axes[1,2].plot(trace[:,0],np.degrees(trace[:,10]),label='Heading deg');axes[1,2].legend();axes[1,2].set(xlabel='Time (s)')
        fig.suptitle('P11.7.9 | '+status);fig.savefig(OUT/'p11_7_9_final_validation.png',dpi=140);plt.close(fig)
        report=f'''# P11.7.9 Fresh Local SLAM → Physical Reverse Parking\n\n状态 **{status}**。真实物理运动：{checks['physical_motion']}；最终完整入库：{inside}。\n\n全新 24×24 m、0.05 m 局部地图，初始后轴 (0,0,0)，不读取旧地图或固定轨迹。前后 LiDAR 沿用原射线模拟器，本阶段配置 720 束/270°/12 m/0.015 m，以真实密集射线解决细栅格覆盖缝隙，后向屋顶传感器安装在后轴前方 0.6 m 并朝后，消除两个朝外传感器之间的近处盲区；保留各自安装射线起点；编码器提供位移、IMU 提供相对航向、现有 PointToLine ICP 对新局部 submap 配准，沿用原匹配阈值。实际采集 {len(self.scan_packets)} 组双传感器扫描，不同 0.2 m 观测位置数 {result['distinct_observation_bins']}。\n\n车位：{json.dumps(clean(self.bay),ensure_ascii=False)}。仅使用新地图长直线边界识别；支持地图两次点击（车身中心、车头方向）并确认，不硬编码旧目标。\n\n探索阶段有实际物理前进；自动泊车只在地图 footprint、定位预算与连续扫掠检查通过后开始。复用 Hybrid A*、ParkingStateMachine、ForwardTracker、ReverseTracker、Stanley/PID 和 PhysicsVehicle。编码器/IMU/局部 SLAM 状态进入控制器，世界位姿仅传感器仿真及离线评估。车只创建一次，探索结束不重置/重建。\n\n安全预算为显式工程概率裕度，不宣称 Gaussian 三西格玛为确定性硬界。执行包络 0.13 m 保留；细栅格精确矩形碰撞、每运动原语半区间位移覆盖和连续插值净空检查覆盖扫掠。旧算法参数及车辆尺寸未改。\n\n最终位置误差 {position} m，航向误差 {heading}°，实际速度 {final[15]} m/s；保持两秒 {hold}；碰撞 {len(self.collisions)}；最后挡位 {final[6]}。全部验收项：\n```json\n{json.dumps(clean(checks),indent=2)}\n```\n\n失败原因：{self.loss_reason}。完整实际轨迹、估计轨迹、原始双向扫描、换挡日志和连续几何检查均保留。工程概率裕度不构成确定性误差证书。历史校验：{result['protected_files_unchanged']}。\n\n启动：`conda run --no-capture-output -n parkingplanner python scripts/run_p11_7_9_fresh_local_parking.py --gui --hold`。W/S 调整低速、A/D 转向、C 回正、SPACE 制动；P 停车并规划；地图点击车身中心再点车头方向，ENTER 确认；计划就绪 A 自动执行；R 返回探索；Q/ESC 急停退出。`--demo --fast --no-map-window` 在同一个物理车上进行短程编码器终止的探索并调用相同规划/执行流程，不是旧路径回放。输出目录已存在指标时拒绝覆盖。\n'''
        (OUT/'p11_7_9_report.md').write_text(report,encoding='utf-8');print(json.dumps(clean({k:result[k]for k in ('status','checks','reason','final_position_error_m','final_heading_error_deg','max_localization_error_m')}),indent=2),flush=True)
        return result

    def close(self):
        for name,fn in self.originals.items():setattr(p,name,fn)
        if p.isConnected(self.cid):p.disconnect(self.cid)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    for flag in ('gui','demo','fast','hold','no-map-window'):parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();OUT=args.output_dir.resolve()
    if (OUT/'p11_7_9_metrics.json').exists():raise FileExistsError('Refusing to overwrite historical P11.7.9 result directory')
    protected={str(f.relative_to(ROOT)):sha(f)for f in (ROOT/'results').rglob('*')if f.is_file()and OUT not in f.parents}
    e=Experiment(args)
    try:
        try:e.run()
        except p.error:
            if p.isConnected(e.cid):raise
            if not e.rows:raise
            e.mode='FAILED';e.loss_reason='GUI_DISCONNECTED';e.terminal=True
        e.save(protected)
        if args.hold and args.gui:
            while p.isConnected(e.cid):
                if p.getKeyboardEvents(physicsClientId=e.cid).get(27,0)&p.KEY_WAS_TRIGGERED:break
                time.sleep(.05)
    finally:e.close()


if __name__=='__main__':main()
