"""Read P10.4.2 numeric execution archive without rerunning any algorithm."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


def wrap(a):
    return np.arctan2(np.sin(a),np.cos(a))


@dataclass
class ReplayData:
    # Columns: x,y,yaw,steering(rad),signed speed,direction,timestamp(relative s).
    samples: np.ndarray
    segments: np.ndarray
    path: np.ndarray
    raw_trajectory: np.ndarray
    metadata: dict

    @property
    def duration(self):
        return float(self.samples[-1,6])

    def at(self,time_s):
        if time_s <= 0:
            return self.samples[0].copy()
        if time_s >= self.duration:
            return self.samples[-1].copy()
        i = int(np.searchsorted(self.samples[:,6],time_s,side='right')-1)
        j = i+1
        a,b = self.samples[i],self.samples[j]
        ratio = (time_s-a[6])/(b[6]-a[6])
        value = a+ratio*(b-a)
        value[2] = wrap(a[2]+ratio*wrap(b[2]-a[2]))
        value[5] = a[5] # gear is a discrete event, never interpolated
        value[6] = time_s
        return value


def load_execution(path):
    path = Path(path)
    with np.load(path,allow_pickle=False) as archive:
        trajectory = archive['trajectory'].copy()
        records = archive['records'].copy()
        columns = list(archive['record_columns'])
        planned = archive['path'].copy()
    required = ['true_x','true_y','true_yaw','steering_deg','speed_mps','direction','time_s','segment_index']
    if any(k not in columns for k in required):
        raise ValueError('Execution archive is missing required measured fields')
    if trajectory.shape != (len(records)+1,3) or planned.ndim != 2 or planned.shape[1] != 5 or len(records)<2:
        raise ValueError('Invalid P10 archive shapes')
    if not all(np.all(np.isfinite(a)) for a in (trajectory,records,planned)):
        raise ValueError('Nonfinite execution data')
    get = lambda name: records[:,columns.index(name)]
    if not np.array_equal(trajectory[1:],np.column_stack([get(k) for k in required[:3]])):
        raise ValueError('Saved physical trajectory disagrees with true pose records')
    times = get('time_s')
    differences = np.diff(times)
    if np.any(differences < 0) or not np.any(differences > 0):
        raise ValueError('Execution timestamps must be nondecreasing')
    dt = float(np.median(differences[differences>0]))
    # Initial trajectory pose precedes the first post-step record by one DT.
    timestamps = np.r_[times[0]-dt,times]
    timestamps -= timestamps[0]
    samples = np.column_stack((trajectory,np.r_[0.,np.deg2rad(get('steering_deg'))],
                                np.r_[0.,get('speed_mps')],np.r_[get('direction')[0],get('direction')],timestamps))
    if not np.all(np.isin(samples[:,5],[-1,0,1])):
        raise ValueError('Invalid gear direction')
    segments = np.r_[get('segment_index')[0],get('segment_index')].astype(int)
    return ReplayData(samples,segments,planned,trajectory,dict(source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),record_columns=columns,
        initial_dt_s=dt,absolute_initial_time_s=float(times[0]-dt),duplicate_timestamps=int(np.sum(differences==0)),
        gear_phase_saved=False,mode='P10.4.2 recorded physical trajectory',
        note='Steering converted degrees to radians; body yaw unchanged in reverse; no resampling or sample removal.'))
