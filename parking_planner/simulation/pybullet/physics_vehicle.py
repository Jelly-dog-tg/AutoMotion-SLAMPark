"""P11.3 physical car: front steering hinges, passive front wheels, rear drive.

No pose reset API is exposed. Base pose is set only by createMultiBody at spawn.
All subsequent motion is produced by wheel contact forces and Bullet stepping.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
import pybullet as p


@dataclass(frozen=True)
class PhysicsVehicleConfig:
    chassis_mass_kg: float = 1100.
    wheel_mass_kg: float = 18.
    knuckle_mass_kg: float = 2.
    wheel_radius_m: float = .32
    wheel_width_m: float = .20
    chassis_height_m: float = .55
    chassis_com_height_m: float = .72
    tire_lateral_friction: float = 1.2
    tire_rolling_friction: float = .002
    tire_spinning_friction: float = .001
    drive_force_nm: float = 160.
    brake_force_nm: float = 300.
    steering_force_nm: float = 180.
    steering_rate_rad_s: float = 1.2
    max_steering_rad: float = math.radians(35.)
    max_speed_mps: float = 1.
    shift_stop_speed_mps: float = .035
    shift_hold_s: float = .35


class PhysicsVehicle:
    def __init__(self,client_id,geometry,pose,config=None):
        self.cid,self.geometry=client_id,geometry
        self.config=cfg=config or PhysicsVehicleConfig()
        self.com_offset=geometry.length/2-geometry.rear_overhang
        self.track=geometry.width-cfg.wheel_width_m
        visual=p.createVisualShape(p.GEOM_BOX,halfExtents=[geometry.length/2,geometry.width/2,cfg.chassis_height_m/2],
                                    rgbaColor=[.1,.48,.82,1],physicsClientId=client_id)
        chassis=p.createCollisionShape(p.GEOM_BOX,halfExtents=[geometry.length/2,geometry.width/2,cfg.chassis_height_m/2],
                                        physicsClientId=client_id)
        wheel_rotation=p.getQuaternionFromEuler([math.pi/2,0,0])
        wheel=p.createCollisionShape(p.GEOM_CYLINDER,radius=cfg.wheel_radius_m,height=cfg.wheel_width_m,
                                      collisionFrameOrientation=wheel_rotation,physicsClientId=client_id)
        wheel_visual=p.createVisualShape(p.GEOM_CYLINDER,radius=cfg.wheel_radius_m,length=cfg.wheel_width_m,
                                         visualFrameOrientation=wheel_rotation,rgbaColor=[.035,.04,.045,1],physicsClientId=client_id)
        masses=[];collisions=[];visuals=[];positions=[];parents=[];types=[];axes=[]
        def link(mass,shape,appearance,position,parent,joint,axis):
            masses.append(mass);collisions.append(shape);visuals.append(appearance)
            positions.append(position);parents.append(parent);types.append(joint);axes.append(axis)
        z=cfg.wheel_radius_m-cfg.chassis_com_height_m
        for side in (1,-1):
            link(cfg.wheel_mass_kg,wheel,wheel_visual,[-self.com_offset,side*self.track/2,z],0,p.JOINT_REVOLUTE,[0,1,0])
        for side in (1,-1):
            knuckle=len(masses)
            link(cfg.knuckle_mass_kg,-1,-1,[geometry.wheelbase-self.com_offset,side*self.track/2,z],0,p.JOINT_REVOLUTE,[0,0,1])
            link(cfg.wheel_mass_kg,wheel,wheel_visual,[0,0,0],knuckle+1,p.JOINT_REVOLUTE,[0,1,0])
        # Decorative fixed links are part of this physical multibody, not
        # separately teleported display objects. They have no collision hull.
        def decoration(position,size,color):
            shape=p.createVisualShape(p.GEOM_BOX,halfExtents=np.array(size)/2,rgbaColor=color,physicsClientId=client_id)
            link(0.,-1,shape,position,0,p.JOINT_FIXED,[0,0,0])
        decoration([-.1,0,.46],[2.1,1.48,.36],[.11,.19,.25,1])
        for side in (1,-1):
            decoration([geometry.length/2-.02,side*.6,.05],[.04,.3,.16],[1,.96,.65,1])
            decoration([-geometry.length/2+.02,side*.6,.05],[.04,.3,.16],[.95,.08,.07,1])
        decoration([.2,0,.65],[.9,.10,.02],[1,.77,.12,1])
        decoration([.65,0,.65],[.12,.36,.02],[1,.77,.12,1])
        x,y,yaw=pose
        base=[x+self.com_offset*math.cos(yaw),y+self.com_offset*math.sin(yaw),cfg.chassis_com_height_m+.02]
        count=len(masses)
        self.body=p.createMultiBody(baseMass=cfg.chassis_mass_kg,baseCollisionShapeIndex=chassis,
            baseVisualShapeIndex=visual,basePosition=base,baseOrientation=p.getQuaternionFromEuler([0,0,yaw]),
            linkMasses=masses,linkCollisionShapeIndices=collisions,linkVisualShapeIndices=visuals,
            linkPositions=positions,linkOrientations=[[0,0,0,1]]*count,
            linkInertialFramePositions=[[0,0,0]]*count,linkInertialFrameOrientations=[[0,0,0,1]]*count,
            linkParentIndices=parents,linkJointTypes=types,linkJointAxis=axes,physicsClientId=client_id)
        self.rear_wheels=[0,1];self.steering_joints=[2,4];self.front_wheels=[3,5]
        self.wheel_joints=self.rear_wheels+self.front_wheels
        for joint in range(count):
            p.setJointMotorControl2(self.body,joint,p.VELOCITY_CONTROL,force=0,physicsClientId=client_id)
        for joint in self.wheel_joints:
            p.changeDynamics(self.body,joint,lateralFriction=cfg.tire_lateral_friction,
                rollingFriction=cfg.tire_rolling_friction,spinningFriction=cfg.tire_spinning_friction,
                restitution=0.,linearDamping=.01,angularDamping=.01,physicsClientId=client_id)
        p.changeDynamics(self.body,-1,linearDamping=.01,angularDamping=.01,restitution=0.,physicsClientId=client_id)
        self.direction=0;self.phase='STOPPED';self.hold_time=0.
        self.command_speed=0.;self.command_steering=0.
        self.command(0.,0.,0.)

    def state(self):
        position,q=p.getBasePositionAndOrientation(self.body,physicsClientId=self.cid)
        rotation=np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)
        rear=np.array(position)-rotation @ np.array([self.com_offset,0.,0.])
        linear,angular=p.getBaseVelocity(self.body,physicsClientId=self.cid)
        rear_velocity=np.array(linear)+np.cross(angular,rear-np.array(position))
        roll,pitch,yaw=p.getEulerFromQuaternion(q)
        steer=[p.getJointState(self.body,j,physicsClientId=self.cid)[0] for j in self.steering_joints]
        return dict(x=float(rear[0]),y=float(rear[1]),z=float(rear[2]),yaw=yaw,
                    speed=float(np.dot(rear_velocity,rotation[:,0])),steering=float(np.mean(steer)),
                    roll=roll,pitch=pitch,direction=self.direction,phase=self.phase,
                    steering_joints=steer)

    def command(self,speed,steering,dt):
        """Wheel-speed actuator with stop-before-reverse interlock; no trajectory control."""
        if not all(math.isfinite(x) for x in (speed,steering,dt)) or dt<0:
            raise ValueError('Invalid actuator command')
        cfg=self.config
        requested=max(-cfg.max_speed_mps,min(cfg.max_speed_mps,speed))
        steering=max(-cfg.max_steering_rad,min(cfg.max_steering_rad,steering))
        state=self.state();wanted=int(np.sign(requested))
        applied=requested
        if wanted and wanted!=self.direction:
            applied=0.
            self.phase='STOP_BEFORE_SHIFT'
            if abs(state['speed']) <= cfg.shift_stop_speed_mps:
                self.hold_time+=dt
                self.phase='SHIFT_HOLD'
                if self.hold_time>=cfg.shift_hold_s:
                    self.direction=wanted;self.hold_time=0.;applied=requested
                    self.phase='FORWARD' if wanted>0 else 'REVERSE'
            else: self.hold_time=0.
        elif wanted:
            self.hold_time=0.;self.phase='FORWARD' if wanted>0 else 'REVERSE'
        else:
            self.hold_time=0.;self.phase='BRAKING' if abs(state['speed'])>cfg.shift_stop_speed_mps else 'STOPPED'
        # Steering is an equivalent bicycle angle; inner/outer wheel Ackermann
        # angles share the same instantaneous center for a 2.7 m wheelbase.
        if abs(steering)<1e-8: angles=[0.,0.]
        else:
            radius=self.geometry.wheelbase/math.tan(steering)
            angles=[math.atan(self.geometry.wheelbase/(radius-y)) for y in (self.track/2,-self.track/2)]
        for joint,angle in zip(self.steering_joints,angles):
            p.setJointMotorControl2(self.body,joint,p.POSITION_CONTROL,targetPosition=angle,
                force=cfg.steering_force_nm,maxVelocity=cfg.steering_rate_rad_s,positionGain=.3,velocityGain=1.,physicsClientId=self.cid)
        for joint in self.rear_wheels:
            p.setJointMotorControl2(self.body,joint,p.VELOCITY_CONTROL,targetVelocity=applied/cfg.wheel_radius_m,
                force=cfg.brake_force_nm if applied==0 else cfg.drive_force_nm,physicsClientId=self.cid)
        for joint in self.front_wheels:
            p.setJointMotorControl2(self.body,joint,p.VELOCITY_CONTROL,targetVelocity=0.,
                force=cfg.brake_force_nm if applied==0 else 0.,physicsClientId=self.cid)
        self.command_speed,self.command_steering=applied,steering

    def parameters(self):
        return dict(**asdict(self.config),length_m=self.geometry.length,width_m=self.geometry.width,
                    wheelbase_m=self.geometry.wheelbase,track_width_m=self.track,
                    total_mass_kg=sum(p.getDynamicsInfo(self.body,j,physicsClientId=self.cid)[0]
                                      for j in range(-1,p.getNumJoints(self.body,physicsClientId=self.cid))),
                    chassis_inertia_kg_m2=list(p.getDynamicsInfo(self.body,-1,physicsClientId=self.cid)[2]))
