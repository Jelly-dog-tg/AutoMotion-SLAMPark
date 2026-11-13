"""Camera-fixed edge HUD. Visual backgrounds have no collision geometry."""
import numpy as np
import pybullet as p


class EdgeStatusBars:
    def __init__(self,client_id):
        self.cid=client_id
        self.panels=[]
        self.text=[]
        # Individual shape dimensions are set at creation, because camera
        # projection can change on resize. Shapes are tiny near-plane panels.
        self.last_projection=None
        self.last_content=None

    def update(self,main,telemetry,hints,legend='BLUE plan / GREEN recorded execution'):
        camera=p.getDebugVisualizerCamera(physicsClientId=self.cid)
        view=np.array(camera[2]).reshape(4,4,order='F')
        projection=np.array(camera[3]).reshape(4,4,order='F')
        inverse=np.linalg.inv(projection @ view)
        def world(x,y,z=-.5):
            q=inverse @ np.array([x,y,z,1.])
            return q[:3]/q[3]
        rotation=np.linalg.inv(view)[:3,:3]
        # Convert the camera orientation matrix to PyBullet quaternion via its
        # orthonormal right/up/back basis (stable for arbitrary orbit angles).
        from math import sqrt
        # Use PyBullet's Euler-free matrix conversion through trace branches.
        m=rotation; tr=np.trace(m)
        if tr>0:
            s=sqrt(tr+1.)*2
            q=[(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s,.25*s]
        else:
            i=int(np.argmax(np.diag(m))); j=(i+1)%3; k=(i+2)%3
            s=sqrt(1+m[i,i]-m[j,j]-m[k,k])*2
            q=[0.,0.,0.,0.];q[i]=.25*s;q[j]=(m[j,i]+m[i,j])/s;q[k]=(m[k,i]+m[i,k])/s;q[3]=(m[k,j]-m[j,k])/s
        specs=[(0.,.91,2.,.18),(0.,-.89,2.,.22),(.64,-.72,.70,.09)]
        projection_key=tuple(camera[3])
        if projection_key!=self.last_projection:
            for body in self.panels: p.removeBody(body,physicsClientId=self.cid)
            self.panels=[]
            for x,y,w,h in specs:
                size=[np.linalg.norm(world(x+w/2,y)-world(x-w/2,y)),
                      np.linalg.norm(world(x,y+h/2)-world(x,y-h/2)),.00002]
                shape=p.createVisualShape(p.GEOM_BOX,halfExtents=np.array(size)/2,
                                           rgbaColor=[.025,.035,.05,.82],physicsClientId=self.cid)
                self.panels.append(p.createMultiBody(baseMass=0,baseCollisionShapeIndex=-1,
                                     baseVisualShapeIndex=shape,physicsClientId=self.cid))
            self.last_projection=projection_key
        for body,(x,y,_,_) in zip(self.panels,specs):
            p.resetBasePositionAndOrientation(body,world(x,y),q,physicsClientId=self.cid)
        entries=[(main,-.96,.915,1.15),(telemetry,-.96,-.845,.95),
                 (hints,-.96,-.94,.78),(legend,.31,-.72,.70)]
        for i,(label,x,y,size) in enumerate(entries):
            old=self.text[i] if i<len(self.text) else -1
            item=p.addUserDebugText(label,world(x,y,-.53),textColorRGB=[.96,.97,1.],textSize=size,
                                    replaceItemUniqueId=old,physicsClientId=self.cid)
            if i<len(self.text): self.text[i]=item
            else: self.text.append(item)
        self.last_content=(main,telemetry,hints,legend)

    def close(self):
        for body in self.panels: p.removeBody(body,physicsClientId=self.cid)
        for item in self.text: p.removeUserDebugItem(item,physicsClientId=self.cid)
