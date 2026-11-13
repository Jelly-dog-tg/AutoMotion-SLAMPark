"""Reverse-only physical tracking, with rear-axle Stanley travel-frame feedback."""
import math
import numpy as np
from parking_planner.control.stanley_controller import StanleyController
from parking_planner.control.speed_controller import SpeedController
from parking_planner.planning.hybrid_astar import PathPoint
from parking_planner.vehicle.ackermann import VehicleState
from .forward_tracker import ForwardTracker


class ReverseTracker:
    def __init__(self,reference,wheelbase):
        if not np.any(reference[:,5]==-1): raise ValueError('Not a recorded reverse segment')
        self.reference=reference
        self.path=[PathPoint(float(r[0]),float(r[1]),float(r[2]),-1,float(r[3])) for r in reference]
        self.ds=np.linalg.norm(np.diff(reference[:,:2],axis=0),axis=1)
        self.arc=np.r_[0.,np.cumsum(self.ds)];self.length=float(self.arc[-1])
        self.stanley=StanleyController(wheelbase=wheelbase,gain=.65,softening_speed=.45,
                                      max_steer=math.radians(35.),search_ahead=30)
        self.pid=SpeedController(kp=2.5,ki=.35,kd=.02,max_acceleration=.4,max_speed=.22)
        self.last_index=0;self.progress=0.;self.motor_speed=0.
        self.stage='REVERSE';self.stopped_time=0.;self.last_steering=float(reference[0,3])

    errors=ForwardTracker.errors

    def update(self,state,dt):
        errors=self.errors(state);self.progress=max(self.progress,errors['arc'])
        remaining=max(0.,self.length-self.progress)
        endpoint=float(np.linalg.norm(np.array([state['x'],state['y']])-self.reference[-1,:2]))
        if self.stage=='COMPLETE': return 0.,self.last_steering,errors,0.
        recorded=abs(float(np.interp(self.progress,self.arc,self.reference[:,4])))
        peak=float(np.max(np.abs(self.reference[:,4])))
        magnitude=min(.22,.20*recorded/peak,math.sqrt(2*.25*remaining))
        target=-magnitude
        feedback=VehicleState(state['x'],state['y'],state['yaw'],state['speed'])
        # direction=-1 makes existing Stanley use the REAR axle; its travel
        # heading is body yaw+pi. Only steering feedback is direction-signed;
        # body yaw and curvature feedforward are never flipped.
        output=self.stanley.compute(feedback,self.path,direction=-1,last_index=self.last_index)
        self.last_index=max(self.last_index,output.target_index)
        if self.stage=='BRAKING' or (remaining<=.012 and endpoint<.10):
            self.stage='BRAKING';target=0.
        else: self.last_steering=output.steering
        acceleration=self.pid.compute(target,state['speed'],dt)
        self.motor_speed=float(np.clip(self.motor_speed+acceleration*dt,-.22,0.))
        if target==0.:
            self.motor_speed=0.
            if abs(state['speed'])<.025:
                self.stopped_time+=dt
                if self.stopped_time>=.5: self.stage='COMPLETE'
            else: self.stopped_time=0.
        return self.motor_speed,self.last_steering,errors,target
