"""Static parking lot built directly from the existing 2D scenario."""
import math
from pathlib import Path
import pybullet as p


class ParkingScene:
    def __init__(self, client_id, scenario):
        self.client_id = client_id
        self.scenario = scenario
        self.bodies = []
        self.obstacle_ids = []
        self.marking_ids = []
        grid = scenario['grid']
        self.box((grid.origin_x+grid.width/2, grid.origin_y+grid.height/2, -.1),
                 (grid.width, grid.height, .2), (.20,.23,.26,1), collision=True)
        for i, (x0,y0,x1,y1) in enumerate(scenario['parked_vehicles']):
            color = (.68,.24,.16,1) if i == 0 else (.26,.42,.52,1)
            body = self.box(((x0+x1)/2,(y0+y1)/2,.7), (x1-x0,y1-y0,1.4), color, collision=True)
            self.obstacle_ids.append(body)
            # Decorative cabin stays inside the exact 2D obstacle footprint.
            self.box(((x0+x1)/2,(y0+y1)/2,1.58), ((x1-x0)*.75,(y1-y0)*.52,.36), (.12,.19,.24,1))
        x0,y0,x1,y1 = scenario['parking_slot']
        white, yellow = (.92,.94,.92,1), (1.,.72,.12,1)
        self.line((x0,y0),(x0,y1),yellow)
        self.line((x1,y0),(x1,y1),yellow)
        self.line((x0,y1),(x1,y1),yellow)
        # Adjacent parking outlines follow the existing parked-car rectangles.
        for a,b,c,d in scenario['parked_vehicles']:
            for start,end in [((a-.12,b-.12),(a-.12,d+.12)),((c+.12,b-.12),(c+.12,d+.12)),
                              ((a-.12,d+.12),(c+.12,d+.12))]:
                self.line(start,end,white)
        # Painted perimeter, aisle lane dashes and direction arrow: visual only.
        a,b = grid.origin_x,grid.origin_y
        c,d = a+grid.width,b+grid.height
        for start,end in [((a+.1,b+.1),(c-.1,b+.1)),((c-.1,b+.1),(c-.1,d-.1)),
                          ((c-.1,d-.1),(a+.1,d-.1)),((a+.1,d-.1),(a+.1,b+.1))]:
            self.line(start,end,white)
        for x in range(2,21,3):
            self.line((x,6.5),(x+1.3,6.5),white)
        self.line((14.5,8.7),(17.5,8.7),yellow,.12)
        self.line((17.5,8.7),(16.8,9.2),yellow,.12)
        self.line((17.5,8.7),(16.8,8.2),yellow,.12)
        # Small painted +X/+Y reference in the ground plane.
        self.line((1,1),(3,1),(1,.25,.2,1),.1)
        self.line((1,1),(1,3),(.25,.8,.35,1),.1)
        from .environment import build_background
        self.background_ids = build_background(self)
        p.setGravity(0,0,-9.81,physicsClientId=client_id)
        p.setRealTimeSimulation(0,physicsClientId=client_id)
        # Focus between the display vehicle and target bay; physical layout unchanged.
        self.camera_target = [9.3,11.5,0.]
        self.camera_distance = max(grid.width,grid.height)*.91
        self.camera_yaw, self.camera_pitch = 35., -62.

    def box(self, center, size, color, collision=False, yaw=0.):
        half = [v/2 for v in size]
        visual = p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color,physicsClientId=self.client_id)
        shape = p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=self.client_id) if collision else -1
        body = p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,baseVisualShapeIndex=visual,
                                 basePosition=center,baseOrientation=p.getQuaternionFromEuler([0,0,yaw]),physicsClientId=self.client_id)
        self.bodies.append(body)
        return body

    def line(self, start, end, color, width=.07):
        dx,dy = end[0]-start[0],end[1]-start[1]
        body = self.box(((start[0]+end[0])/2,(start[1]+end[1])/2,.006),
                        (math.hypot(dx,dy),width,.008),color,yaw=math.atan2(dy,dx))
        self.marking_ids.append(body)
        return body

    def configure_gui(self):
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,1,physicsClientId=self.client_id)
        # Start closer, then fit the physical lot corners to the actual GUI
        # projection/aspect ratio (including Windows display scaling).
        import numpy as np
        import time
        grid = self.scenario['grid']
        corners = np.array([[x,y,0.,1.] for x in (grid.origin_x,grid.origin_x+grid.width)
                            for y in (grid.origin_y,grid.origin_y+grid.height)])
        for _ in range(40):
            p.resetDebugVisualizerCamera(self.camera_distance,self.camera_yaw,self.camera_pitch,
                                        self.camera_target,physicsClientId=self.client_id)
            # GUI camera messages are asynchronous. Wait for the requested
            # camera before fitting; stale startup matrices must not inflate zoom.
            time.sleep(.04)
            camera = p.getDebugVisualizerCamera(physicsClientId=self.client_id)
            if (abs(camera[10]-self.camera_distance)>.01 or
                    not np.allclose(camera[11],self.camera_target,atol=.01)):
                continue
            view = np.array(camera[2]).reshape(4,4,order='F')
            projection = np.array(camera[3]).reshape(4,4,order='F')
            clip = corners @ (projection @ view).T
            ndc = clip[:,:2]/clip[:,3,None]
            if np.all(np.abs(ndc) <= .96):
                break
            self.camera_distance *= 1.02

    def save_screenshot(self, path, width=1440, height=1000, gui=False):
        """One artifact: rendered full-lot overview plus a vehicle detail view."""
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def render(target, distance, yaw, pitch, fov):
            view = p.computeViewMatrixFromYawPitchRoll(target,distance,yaw,pitch,0,2)
            projection = p.computeProjectionMatrixFOV(fov,width/height,.1,100)
            image = p.getCameraImage(width,height,viewMatrix=view,projectionMatrix=projection,
                                     lightDirection=[-3,-4,9],lightColor=[1,1,.95],shadow=1,
                                     lightAmbientCoeff=.55,lightDiffuseCoeff=.65,lightSpecularCoeff=.15,
                                     renderer=p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER,
                                     physicsClientId=self.client_id)
            return np.asarray(image[2],dtype=np.uint8).reshape(height,width,4)

        overview = render(self.camera_target,self.camera_distance,self.camera_yaw,self.camera_pitch,85)
        start, geometry = self.scenario['start'], self.scenario['geometry']
        offset = geometry.length/2-geometry.rear_overhang
        detail = render([start.x+offset*math.cos(start.yaw),start.y+offset*math.sin(start.yaw),.6],
                        7.5,35.,-32.,50)
        fig, axes = plt.subplots(1,2,figsize=(16,6),layout='constrained')
        for ax, pixels, title in zip(axes,[overview,detail],
                                     ['Full parking lot (existing 2D layout)','Vehicle detail: +X forward, steerable front wheels']):
            ax.imshow(pixels)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle('P11.1 | PyBullet rendered scene | rear-axle pose reference')
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=150)
        plt.close(fig)
