"""Simulated wheel encoders and IMU. No world XY is exposed to localization.

Simulator-only acquisition may read orientation/angular rate to model an IMU;
quantization/noise and their distinct error models are explicitly retained.
"""
import math
import numpy as np
import pybullet as p
from parking_planner.slam.online_slam import compose_pose


class EncoderIMU:
    def __init__(self,car):
        self.car=car;self.rng=np.random.default_rng(20261114);self.last=None
        self.encoder_tick_rad=2*math.pi/4096
        self.heading_noise_bound_rad=math.radians(.1)
        self.gyro_noise_bound_rad_s=math.radians(.02)

    def sample(self,dt):
        car=self.car
        joints=[p.getJointState(car.body,j,physicsClientId=car.cid)for j in car.rear_wheels]
        angles=np.round(np.array([j[0]for j in joints])/self.encoder_tick_rad)*self.encoder_tick_rad
        speeds=np.array([j[1]for j in joints])*car.config.wheel_radius_m
        q=p.getBasePositionAndOrientation(car.body,physicsClientId=car.cid)[1]
        heading=p.getEulerFromQuaternion(q)[2]+self.rng.uniform(-self.heading_noise_bound_rad,self.heading_noise_bound_rad)
        gyro=p.getBaseVelocity(car.body,physicsClientId=car.cid)[1][2]+self.rng.uniform(-self.gyro_noise_bound_rad_s,self.gyro_noise_bound_rad_s)
        distance=0. if self.last is None else float(np.mean(angles-self.last)*car.config.wheel_radius_m)
        self.last=angles
        steering=float(np.mean([p.getJointState(car.body,j,physicsClientId=car.cid)[0]for j in car.steering_joints]))
        return dict(wheel_angles_rad=angles,wheel_speed_mps=speeds,speed_mps=float(speeds.mean()),
            forward_increment_m=distance,heading_rad=math.atan2(math.sin(heading),math.cos(heading)),
            gyro_rad_s=float(gyro),steering_rad=steering,dt=dt)

    def advance(self,pose,measurement):
        # Gyro integration has no physical pose access. Absolute heading is
        # retained separately for map/world-frame consistency gating.
        return compose_pose(pose,[measurement['forward_increment_m'],0.,measurement['gyro_rad_s']*measurement['dt']])
