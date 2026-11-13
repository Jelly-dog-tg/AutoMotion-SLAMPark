"""Two existing PyBullet LiDARs with explicit SE(2) mounting transforms.

Fusion is for registration only. Map updates retain the two distinct ray origins.
"""
import math
import numpy as np
from parking_planner.sensing.pybullet_lidar import PyBulletLidar2D
from parking_planner.sensing.lidar_2d import LidarConfig
from parking_planner.vehicle.ackermann import VehicleState


class DualLidar360:
    def __init__(self,cid,body,front_config=None,rear_offset=(-.6,0.)):
        self.config=front_config or LidarConfig(num_beams=180)
        self.mounts=np.array([[0.,0.,0.],[*rear_offset,math.pi]])
        self.sensors=[PyBulletLidar2D(cid,body,self.config,height_m=1.25)for _ in range(2)]
        # Independent noise streams, fixed and reproducible. Front preserves its seed.
        self.sensors[1].rng=np.random.default_rng(20261113)

    def capture(self,physical_sensor_state):
        raw=[];points=[];origins=[];directions=[];ends=[];valid=[];source=[]
        x,y,yaw=[physical_sensor_state[k]for k in ('x','y','yaw')]
        c,s=math.cos(yaw),math.sin(yaw)
        for i,(sensor,mount)in enumerate(zip(self.sensors,self.mounts)):
            dx,dy,angle=mount
            scan=sensor.scan(dict(x=x+c*dx-s*dy,y=y+s*dx+c*dy,yaw=yaw+angle))
            # Original simulator excludes every self-body return from valid points.
            # If any self ray was suppressed, no invalid ray is used to clear map cells.
            theta=scan['angles_rad']+angle
            ray=np.column_stack((np.cos(theta),np.sin(theta)))
            endpoint=mount[:2]+scan['ranges_m'][:,None]*ray
            raw.append(scan);points.append(endpoint[scan['valid']]);origins.append(np.tile(mount[:2],(len(theta),1)))
            directions.append(ray);ends.append(endpoint);valid.append(scan['valid']);source.extend([i]*len(theta))
        bearings=np.concatenate([r['angles_rad']+m[2]for r,m in zip(raw,self.mounts)])
        bearings=np.arctan2(np.sin(bearings),np.cos(bearings));ordered=np.sort(bearings)
        return dict(raw=raw,points_local=np.concatenate(points),ray_origins_local=np.concatenate(origins),
            ray_directions_local=np.concatenate(directions),ray_endpoints_local=np.concatenate(ends),
            valid=np.concatenate(valid),sensor_index=np.asarray(source,dtype=np.int8),angles_rad=bearings,
            largest_angular_gap_deg=float(np.degrees(np.max(np.diff(np.r_[ordered,ordered[0]+2*math.pi])))),
            mounts=self.mounts.copy(),self_hits=sum(r['self_hits']for r in raw))

    def update_map(self,mapper,localized_base_pose,packet):
        """Existing inverse sensor model, separate ray origin for each sensor.

        Caller MUST gate the pose before invoking; this method does not certify it.
        """
        x,y,yaw=localized_base_pose;c,s=math.cos(yaw),math.sin(yaw)
        for mount,raw in zip(self.mounts,packet['raw']):
            dx,dy,angle=mount;scan={k:np.array(raw[k],copy=True)for k in ('angles_rad','ranges_m','valid')}
            if raw['self_hits']:scan['ranges_m'][~scan['valid']]=np.nan
            mapper.update(VehicleState(x+c*dx-s*dy,y+s*dx+c*dy,yaw+angle),scan,self.config)
