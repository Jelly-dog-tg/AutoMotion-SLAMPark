"""Replay timing and PyBullet rendering; no physics integration or control."""
import math
import numpy as np
import pybullet as p
from .trajectory import wrap
from .hud import EdgeStatusBars


class ReplayClock:
    def __init__(self,duration,rate=1.,preview_hold=1.5,transition_duration=3.):
        self.duration, self.rate = duration,rate
        self.preview_hold, self.transition_duration = preview_hold,transition_duration
        self.elapsed, self.paused = 0.,False

    def restart(self):
        self.elapsed, self.paused = 0.,False

    def advance(self,wall_dt):
        if not self.paused:
            # Presentation transition has wall-clock duration; replay alone is sped up.
            lead = self.preview_hold+self.transition_duration
            if self.elapsed < lead:
                used = min(wall_dt,lead-self.elapsed)
                self.elapsed += used
                wall_dt -= used
            self.elapsed = min(lead+self.duration,self.elapsed+wall_dt*self.rate)

    @property
    def time(self):
        return max(0.,self.elapsed-self.preview_hold-self.transition_duration)

    @property
    def finished(self):
        return self.time >= self.duration

    def sample(self,data,preview):
        if self.elapsed < self.preview_hold:
            return np.r_[preview,0.,0.,data.samples[0,5],0.], 'PREVIEW (display only)'
        if self.elapsed < self.preview_hold+self.transition_duration:
            u = (self.elapsed-self.preview_hold)/self.transition_duration
            blend = u*u*(3-2*u)
            value = data.samples[0].copy()
            value[:2] = preview[:2]+blend*(value[:2]-preview[:2])
            value[2] = wrap(preview[2]+blend*wrap(value[2]-preview[2]))
            value[3:5] = 0.
            return value,'DISPLAY TRANSITION (not execution)'
        value = data.at(self.time)
        if self.finished:
            stage = 'COMPLETE'
        elif abs(value[4]) < 1e-6:
            stage = 'STOP / GEAR HOLD (recorded)'
        else:
            stage = 'FORWARD' if value[4]>0 else 'REVERSE'
        return value,stage


class ReplayView:
    def __init__(self,scene,vehicle,data):
        self.scene,self.vehicle,self.data = scene,vehicle,data
        self.cid = scene.client_id
        self.lines = []
        self.next_edge = 1
        self.hud = EdgeStatusBars(self.cid)
        for a,b in zip(data.path[:-1],data.path[1:]):
            self.line(a,b,[.08,.30,1.],.035,2)

    def line(self,a,b,color,z,width):
        return p.addUserDebugLine([float(a[0]),float(a[1]),z],[float(b[0]),float(b[1]),z],
                                   lineColorRGB=color,lineWidth=width,physicsClientId=self.cid)

    def reset(self):
        for item in self.lines:
            p.removeUserDebugItem(item,physicsClientId=self.cid)
        self.lines.clear()
        self.next_edge = 1

    def update(self,sample,stage,time_s,rate,paused=False,execution=True):
        self.vehicle.set_pose(*sample[:4])
        if execution:
            index = int(np.searchsorted(self.data.samples[:,6],time_s,side='right')-1)
            while self.next_edge <= index:
                i = self.next_edge
                self.lines.append(self.line(self.data.samples[i-1],self.data.samples[i],[.05,.8,.3],.055,3))
                self.next_edge += 1
        self.hud.update(
            f'{"PAUSED | " if paused else ""}{stage} | {time_s:.2f}/{self.data.duration:.2f}s | {100*time_s/self.data.duration:.1f}% | {rate:.2f}x',
            f'v={sample[4]:+.3f} m/s | steer={math.degrees(sample[3]):+.1f} deg | yaw={math.degrees(sample[2]):+.1f} deg',
            'SPACE pause | R/F5 restart | +/- rate | Q/ESC exit')

    def screenshot(self,path,caption):
        """Capture current OpenGL GUI camera, including debug paths and HUD."""
        camera = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        image = p.getCameraImage(1100,800,viewMatrix=camera[2],projectionMatrix=camera[3],
                                 renderer=p.ER_BULLET_HARDWARE_OPENGL,physicsClientId=self.cid)
        pixels = np.asarray(image[2],dtype=np.uint8).reshape(800,1100,4)
        # getCameraImage does not consistently render GUI debug overlays on all
        # drivers; add a truthful caption and exact source path overlay to PNG.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(figsize=(11,8),dpi=100)
        ax.imshow(pixels)
        view,projection = np.array(camera[2]).reshape(4,4,order='F'),np.array(camera[3]).reshape(4,4,order='F')
        def projected(points,z):
            homogeneous = np.column_stack((points[:,:2],np.full(len(points),z),np.ones(len(points))))
            clip = homogeneous @ (projection @ view).T
            ndc = clip[:,:2]/clip[:,3,None]
            return np.column_stack(((ndc[:,0]+1)*550,(1-ndc[:,1])*400))
        xy = projected(self.data.path,.035)
        ax.plot(xy[:,0],xy[:,1],color='royalblue',lw=1,label='Hybrid A* (original)')
        if self.next_edge>1:
            xy = projected(self.data.samples[:self.next_edge],.055)
            ax.plot(xy[:,0],xy[:,1],color='#16b953',lw=1,label='Recorded execution')
        from matplotlib.patches import Rectangle
        ax.set(xlim=(0,1100),ylim=(800,0))
        for y,height in [(0,65),(710,90)]:
            ax.add_patch(Rectangle((0,y),1100,height,facecolor='#070b12',alpha=.82,zorder=10))
        main,telemetry,hints,legend=self.hud.last_content
        ax.text(18,32,main,color='white',fontsize=12,va='center',zorder=11)
        ax.text(18,740,telemetry,color='white',fontsize=10,va='center',zorder=11)
        ax.text(18,778,hints,color='#d3d9e5',fontsize=8,va='center',zorder=11)
        ax.axis('off')
        ax.legend(loc='lower right',bbox_to_anchor=(.99,.12),fontsize=8,facecolor='#101722',labelcolor='white',framealpha=.82)
        fig.tight_layout(pad=.3)
        fig.savefig(path)
        plt.close(fig)
