"""Kinematic display model referenced to the existing rear-axle center."""
import math
import pybullet as p


class VehicleVisual:
    """Visual-only bodies, +X forward, +Y left, yaw about +Z (radians).

    set_pose is an absolute display update, with no dynamics or control logic.
    The supplied steering angle is displayed on both front wheels directly.
    """
    def __init__(self, client_id, geometry):
        self.client_id, self.geometry = client_id, geometry
        self.parts = []
        self.wheels = {}
        self.pose = (0.,0.,0.,0.)
        self.wheel_radius, self.wheel_width = .32,.20
        self.track_width = geometry.width-self.wheel_width
        center = geometry.length/2-geometry.rear_overhang
        self.body_id = self._box((center,0,.68),(geometry.length,geometry.width,.60),(.10,.48,.82,1))
        self._box((center-.10,0,1.14),(geometry.length*.48,geometry.width*.83,.40),(.11,.19,.25,1))
        # White headlights (+X), red tail lights (-X), and a roof arrow.
        for side in (-1,1):
            self._box((geometry.front_extent-.018,side*geometry.width*.34,.75),(.04,.30,.16),(1.,.96,.65,1))
            self._box((-geometry.rear_overhang+.018,side*geometry.width*.34,.75),(.04,.30,.16),(.95,.08,.07,1))
        self._box((center+.22,0,1.351),(.95,.10,.02),(1.,.77,.12,1))
        for side in (-1,1):
            self._box((center+.56,side*.13,1.351),(.40,.08,.02),(1.,.77,.12,1),local_yaw=-side*.7)
        for axle,x in [('rear',0.),('front',geometry.wheelbase)]:
            for side,y in [('left',self.track_width/2),('right',-self.track_width/2)]:
                # Cylinders default to a Z axis; rotate onto the local wheel axle Y.
                visual = p.createVisualShape(p.GEOM_CYLINDER,radius=self.wheel_radius,length=self.wheel_width,
                                             rgbaColor=[.035,.04,.045,1],physicsClientId=client_id)
                body = p.createMultiBody(baseMass=0,baseVisualShapeIndex=visual,baseCollisionShapeIndex=-1,physicsClientId=client_id)
                self.parts.append((body,(x,y,self.wheel_radius),0.,True,axle=='front'))
                self.wheels[axle+'_'+side] = body
        self.set_pose(0,0,0,0)

    def _box(self, local_position, size, color, local_yaw=0.):
        visual = p.createVisualShape(p.GEOM_BOX,halfExtents=[v/2 for v in size],rgbaColor=color,physicsClientId=self.client_id)
        body = p.createMultiBody(baseMass=0,baseVisualShapeIndex=visual,baseCollisionShapeIndex=-1,physicsClientId=self.client_id)
        self.parts.append((body,local_position,local_yaw,False,False))
        return body

    def set_pose(self,x,y,yaw,steering_angle=0.):
        """Set rear-axle (x,y), global yaw and front-wheel display angle, radians."""
        if not all(math.isfinite(v) for v in (x,y,yaw,steering_angle)):
            raise ValueError('Pose and steering must be finite')
        c,s = math.cos(yaw),math.sin(yaw)
        for body,(lx,ly,lz),local_yaw,wheel,front in self.parts:
            position = [x+c*lx-s*ly,y+s*lx+c*ly,lz]
            heading = yaw+local_yaw+(steering_angle if front else 0.)
            orientation = p.getQuaternionFromEuler([math.pi/2 if wheel else 0.,0.,heading])
            p.resetBasePositionAndOrientation(body,position,orientation,physicsClientId=self.client_id)
        self.pose = (float(x),float(y),float(yaw),float(steering_angle))
