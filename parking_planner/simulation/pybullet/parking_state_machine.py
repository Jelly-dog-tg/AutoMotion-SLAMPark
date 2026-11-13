"""Recorded segment extraction and continuous stop-before-shift supervisor."""
from dataclasses import dataclass
import numpy as np
from .forward_tracker import ForwardTracker
from .reverse_controller import ReverseTracker


@dataclass
class RecordedSegment:
    reference: np.ndarray
    start: int
    stop: int
    direction: int
    hold_after_s: float = 0.


def split_recorded_segments(data):
    segments=[]
    for group in np.unique(data.segments):
        indices=np.flatnonzero(data.segments==group)
        moving=indices[np.abs(data.samples[indices,4])>1e-6]
        if not len(moving): raise ValueError('Saved segment contains no moving records')
        start=int(moving[0]);direction=int(data.samples[start,5])
        if direction not in (-1,1): raise ValueError('Moving frame lacks valid direction')
        if np.any(np.sign(data.samples[moving,4])!=direction) or np.any(data.samples[moving,5]!=direction):
            raise ValueError('Mixed signed motion in saved segment')
        stop=int(moving[-1])+1
        if stop>=len(data.samples) or abs(data.samples[stop,4])>1e-9:
            raise ValueError('Missing original stopping record')
        if not segments: start=0 # preserve actual full-execution initial pose
        reference=data.samples[start:stop+1].copy()
        if np.any((reference[:,5]!=direction)&(np.abs(reference[:,4])>1e-6)):
            raise ValueError('Direction-switch data included in motion segment')
        segments.append(RecordedSegment(reference,start,stop,direction))
    for a,b in zip(segments[:-1],segments[1:]):
        if not np.allclose(a.reference[-1,:3],b.reference[0,:3],atol=1e-9,rtol=0):
            raise ValueError('Discontinuous recorded handoff poses')
        a.hold_after_s=float(b.reference[0,6]-a.reference[-1,6])
    return segments


class ParkingStateMachine:
    STOP_SPEED=.015
    STOP_CONFIRM=.30
    FINAL_HOLD=2.

    def __init__(self,segments,wheelbase):
        self.segments=segments
        self.trackers=[ForwardTracker(s.reference,wheelbase) if s.direction>0 else ReverseTracker(s.reference,wheelbase) for s in segments]
        self.index=-1;self.pending=0;self.phase='SHIFT_TO_REVERSE' if segments[0].direction<0 else 'SHIFT_TO_FORWARD'
        self.stable=0.;self.phase_time=0.;self.finished=False;self.failure=None
        self.target_speed=0.;self.motor=0.;self.steering=self.first_steer(0)
        self.events=[];self.shifts=[];self.segment_events=[];self.current_shift=None;self.active_since=None
        self.last_errors=self.trackers[0].errors(dict(x=segments[0].reference[0,0],y=segments[0].reference[0,1],yaw=segments[0].reference[0,2]))

    def first_steer(self,index):
        ref=self.segments[index].reference
        moving=np.flatnonzero(np.abs(ref[:,4])>1e-6)
        return float(ref[moving[0],3])

    def change(self,phase,t,state):
        if phase==self.phase:return
        self.events.append(dict(time_s=t,from_state=self.phase,to_state=phase,
            actual_speed_mps=state['speed'],target_speed_mps=self.target_speed,gear=state['direction']))
        self.phase=phase;self.phase_time=0.;self.stable=0.

    def emergency(self,reason,t,state):
        self.failure=reason;self.motor=0.;self.target_speed=0.;self.change('EMERGENCY_STOP',t,state)

    @property
    def progress(self):
        total=sum(tr.length for tr in self.trackers)
        done=sum(tr.length for tr in self.trackers[:max(0,self.index)])
        if self.index>=0:done+=min(self.trackers[self.index].progress,self.trackers[self.index].length)
        return 100*done/total

    def update(self,state,dt,t):
        self.phase_time+=dt
        self.motor=0.;self.target_speed=0.
        if self.index>=0:self.last_errors=self.trackers[self.index].errors(state)
        if self.phase in ('FORWARD','REVERSE','DECELERATING'):
            tracker=self.trackers[self.index]
            self.motor,self.steering,self.last_errors,self.target_speed=tracker.update(state,dt)
            if tracker.stage=='BRAKING':self.change('DECELERATING',t,state)
            if tracker.stage=='COMPLETE':
                self.segment_events.append(dict(segment=self.index,direction=self.segments[self.index].direction,
                    start_time_s=self.active_since,end_time_s=t,actual_pose=[state['x'],state['y'],state['yaw']],
                    reference_pose=self.segments[self.index].reference[-1,:3].tolist(),progress_percent=100*tracker.progress/tracker.length,
                    stop_speed_mps=state['speed']))
                self.motor=0.;self.target_speed=0.
                self.change('FINAL_BRAKING' if self.index==len(self.segments)-1 else 'STOPPED',t,state)
        elif self.phase=='STOPPED':
            if abs(state['speed'])<=self.STOP_SPEED:self.stable+=dt
            else:self.stable=0.
            if self.stable>=self.STOP_CONFIRM:
                self.pending=self.index+1
                self.current_shift=dict(from_segment=self.index,to_segment=self.pending,
                    from_direction=self.segments[self.index].direction,to_direction=self.segments[self.pending].direction,
                    start_time_s=t,entry_speed_mps=state['speed'],original_recorded_hold_s=self.segments[self.index].hold_after_s)
                self.change('SHIFT_TO_FORWARD' if self.segments[self.pending].direction>0 else 'SHIFT_TO_REVERSE',t,state)
        elif self.phase.startswith('SHIFT_TO_'):
            desired=self.segments[self.pending].direction
            self.steering=self.first_steer(self.pending)
            aligned=abs(state['steering']-self.steering)<.04
            if abs(state['speed'])<=self.STOP_SPEED and aligned:self.stable+=dt
            else:self.stable=0.
            recorded=self.segments[self.index].hold_after_s if self.index>=0 else .5
            if self.stable>=max(.5,recorded):
                # Small requested speed engages the UNCHANGED vehicle gearbox.
                # Its internal .35 s standstill interlock still applies.
                self.motor=desired*.02
            if state['direction']==desired:
                if self.current_shift is not None:
                    self.current_shift.update(end_time_s=t,duration_s=t-self.current_shift['start_time_s'],
                        engagement_speed_mps=state['speed'],actual_direction=state['direction'])
                    self.shifts.append(self.current_shift);self.current_shift=None
                self.index=self.pending;self.active_since=t
                self.change('FORWARD' if desired>0 else 'REVERSE',t,state)
        elif self.phase in ('FINAL_BRAKING','EMERGENCY_STOP'):
            if abs(state['speed'])<=.01:self.stable+=dt
            else:self.stable=0.
            if self.stable>=self.FINAL_HOLD:
                self.finished=True
                if self.phase!='EMERGENCY_STOP':self.change('COMPLETE',t,state)
        return self.motor,self.steering,self.last_errors,self.target_speed
