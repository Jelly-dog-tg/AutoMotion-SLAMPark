"""Forward-only physical-state tracking adapter around existing Stanley/PID."""
import math
import numpy as np
from parking_planner.control.stanley_controller import StanleyController
from parking_planner.control.speed_controller import SpeedController
from parking_planner.planning.hybrid_astar import PathPoint
from parking_planner.vehicle.ackermann import VehicleState
from .trajectory import wrap


def first_forward_segment(data):
    start=int(np.flatnonzero((data.samples[:,5]==1)&(data.samples[:,4]>1e-6))[0])
    stop=start+1
    while stop<len(data.samples) and data.samples[stop,5]==1 and data.samples[stop,4]>1e-6:
        stop+=1
    if stop==len(data.samples): raise ValueError('Forward segment has no recorded stop')
    # Include the FIRST stopping event; never include following gear transitions.
    segment=data.samples[start:stop+1].copy()
    if segment[-1,5]!=1 or segment[-1,4]!=0:
        # A switch without a zero-speed sample is not claimed to contain a stop.
        raise ValueError('Forward segment ends in an unrecorded stop/gear switch')
    return segment,start,stop


class ForwardTracker:
    def __init__(self,reference,wheelbase):
        self.reference=reference
        self.path=[PathPoint(float(r[0]),float(r[1]),float(r[2]),1,float(r[3])) for r in reference]
        self.ds=np.linalg.norm(np.diff(reference[:,:2],axis=0),axis=1)
        self.arc=np.r_[0.,np.cumsum(self.ds)]
        self.length=float(self.arc[-1])
        self.stanley=StanleyController(wheelbase=wheelbase,gain=.65,softening_speed=.45,
                                       max_steer=math.radians(35.),search_ahead=30)
        self.pid=SpeedController(kp=2.5,ki=.35,kd=.02,max_acceleration=.4,max_speed=.30)
        self.last_index=0;self.progress=0.;self.motor_speed=0.
        self.stage='FORWARD';self.stopped_time=0.;self.last_steering=float(reference[0,3])

    def errors(self,state):
        a=self.reference[:-1,:2];edges=np.diff(self.reference[:,:2],axis=0)
        position=np.array([state['x'],state['y']])
        u=np.clip(np.sum((position-a)*edges,axis=1)/np.maximum(self.ds**2,1e-15),0,1)
        closest=a+u[:,None]*edges
        i=int(np.argmin(np.sum((closest-position)**2,axis=1)))
        yaw=wrap(self.reference[i,2]+u[i]*wrap(self.reference[i+1,2]-self.reference[i,2]))
        delta=position-closest[i]
        return dict(position_error=float(np.linalg.norm(delta)),
                    cross_track=float(delta @ np.array([-math.sin(yaw),math.cos(yaw)])),
                    heading_error=float(wrap(yaw-state['yaw'])),
                    arc=float(self.arc[i]+u[i]*self.ds[i]),reference_index=i)

    def update(self,state,dt):
        errors=self.errors(state)
        self.progress=max(self.progress,errors['arc'])
        remaining=max(0.,self.length-self.progress)
        endpoint=float(np.linalg.norm(np.array([state['x'],state['y']])-self.reference[-1,:2]))
        terminal_heading=abs(float(wrap(state['yaw']-self.reference[-1,2])))
        if self.stage=='COMPLETE': return 0.,self.last_steering,errors,0.
        # Preserve the shape of recorded speed ramp/braking, with a configured
        # low-speed ceiling. This is a new closed-loop run, not timed replay.
        recorded=float(np.interp(self.progress,self.arc,self.reference[:,4]))
        peak=float(np.max(self.reference[:,4]))
        target=min(.30,.28*recorded/peak,math.sqrt(2*.25*remaining))
        feedback=VehicleState(state['x'],state['y'],state['yaw'],state['speed'])
        result=self.stanley.compute(feedback,self.path,1,self.last_index)
        self.last_index=max(self.last_index,result.target_index)
        if self.stage=='BRAKING' or (remaining<=.012 and endpoint<.10):
            # Hold the preceding wheel angle through braking and standstill.
            # Unwinding loaded tires can physically pull the rigid car backwards.
            self.stage='BRAKING';target=0.
        else:
            self.last_steering=result.steering
        acceleration=self.pid.compute(target,state['speed'],dt)
        self.motor_speed=float(np.clip(self.motor_speed+acceleration*dt,0,.30))
        if target==0.:
            # Explicit actuator brake once the segment ends; no negative drive.
            self.motor_speed=0.
            if abs(state['speed'])<.025:
                self.stopped_time+=dt
                if self.stopped_time>=.5: self.stage='COMPLETE'
            else: self.stopped_time=0.
        return self.motor_speed,self.last_steering,errors,target
