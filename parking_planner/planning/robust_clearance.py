"""Read-only exact rectangle clearance and conservative tracking-error envelope."""
import math
import numpy as np
from parking_planner.planning.hybrid_astar import OccupancyGrid


class CellClearance:
    def __init__(self,mask,resolution,origin,geometry):
        self.rows,self.cols=np.where(mask)
        self.resolution=resolution;self.origin=origin;self.geometry=geometry
        x=origin[0]+self.cols*resolution;y=origin[1]+self.rows*resolution
        self.boxes=np.column_stack((x,y,x+resolution,y+resolution))
        self.radius=math.hypot(max(geometry.front_extent,geometry.rear_overhang),geometry.width/2)

    def polygon(self,pose):
        g=self.geometry;x,y,yaw=pose;c,s=math.cos(yaw),math.sin(yaw)
        local=np.array([[-g.rear_overhang,-g.width/2],[g.front_extent,-g.width/2],
                        [g.front_extent,g.width/2],[-g.rear_overhang,g.width/2]])
        return local @ np.array([[c,s],[-s,c]])+np.array([x,y])

    def exact_distances(self,pose,boxes):
        vertices=self.polygon(pose);g=self.geometry
        # Vehicle vertex -> axis-aligned cell rectangle.
        dx=np.maximum(np.maximum(boxes[:,None,0]-vertices[None,:,0],vertices[None,:,0]-boxes[:,None,2]),0)
        dy=np.maximum(np.maximum(boxes[:,None,1]-vertices[None,:,1],vertices[None,:,1]-boxes[:,None,3]),0)
        first=np.min(dx*dx+dy*dy,axis=1)
        corners=np.stack((boxes[:,[0,1]],boxes[:,[2,1]],boxes[:,[2,3]],boxes[:,[0,3]]),axis=1)
        c,s=math.cos(pose[2]),math.sin(pose[2]);delta=corners-np.array(pose[:2])
        lx=c*delta[:,:,0]+s*delta[:,:,1];ly=-s*delta[:,:,0]+c*delta[:,:,1]
        dx=np.maximum(np.maximum(-g.rear_overhang-lx,lx-g.front_extent),0)
        dy=np.maximum(np.abs(ly)-g.width/2,0)
        second=np.min(dx*dx+dy*dy,axis=1)
        # Four separating axes catch edge crossings without interior vertices.
        intersect=((boxes[:,0]<=vertices[:,0].max()+1e-12)&(boxes[:,2]>=vertices[:,0].min()-1e-12)&
                   (boxes[:,1]<=vertices[:,1].max()+1e-12)&(boxes[:,3]>=vertices[:,1].min()-1e-12)&
                   (lx.min(axis=1)<=g.front_extent+1e-12)&(lx.max(axis=1)>=-g.rear_overhang-1e-12)&
                   (ly.min(axis=1)<=g.width/2+1e-12)&(ly.max(axis=1)>=-g.width/2-1e-12))
        distance=np.sqrt(np.minimum(first,second));distance[intersect]=0.
        return distance

    def nearest(self,pose):
        if not len(self.boxes):return float('inf'),None
        poly=self.polygon(pose);lo=poly.min(axis=0);hi=poly.max(axis=0)
        dx=np.maximum(np.maximum(self.boxes[:,0]-hi[0],lo[0]-self.boxes[:,2]),0)
        dy=np.maximum(np.maximum(self.boxes[:,1]-hi[1],lo[1]-self.boxes[:,3]),0)
        bounds=dx*dx+dy*dy
        radius=.20
        while True:
            selected=np.flatnonzero(bounds<=radius*radius)
            if len(selected):
                distance=self.exact_distances(pose,self.boxes[selected]);j=int(np.argmin(distance))
                if distance[j]<=radius+1e-12:
                    index=int(selected[j])
                    return float(distance[j]),dict(row=int(self.rows[index]),col=int(self.cols[index]),
                        center=[float(self.boxes[index,0]+self.resolution/2),float(self.boxes[index,1]+self.resolution/2)],
                        bounds=self.boxes[index].tolist())
            radius*=2

    def within(self,pose,margin):
        if not len(self.boxes):return False
        poly=self.polygon(pose);lo=poly.min(axis=0);hi=poly.max(axis=0)
        b=self.boxes
        selected=(b[:,0]<=hi[0]+margin)&(b[:,2]>=lo[0]-margin)&(b[:,1]<=hi[1]+margin)&(b[:,3]>=lo[1]-margin)
        if not selected.any():return False
        return bool(np.any(self.exact_distances(pose,b[selected])<=margin+1e-12))


def pose_between(a,b,u):
    angle=math.atan2(math.sin(b[2]-a[2]),math.cos(b[2]-a[2]))
    return np.array([a[0]+u*(b[0]-a[0]),a[1]+u*(b[1]-a[1]),a[2]+u*angle])


def movement_bound(a,b,radius):
    angle=abs(math.atan2(math.sin(b[2]-a[2]),math.cos(b[2]-a[2])))
    return float(np.linalg.norm(np.array(b[:2])-a[:2])+2*radius*math.sin(angle/2))


class RobustCollisionGrid(OccupancyGrid):
    """Same occupancy cells and legacy check, plus a full-body distance guard.

    Only collision semantics become stricter. Vehicle dimensions and map labels
    remain untouched; the margin bounds EVERY position/yaw error combination.
    """
    def __init__(self,base,geometry,margin):
        super().__init__(base.width,base.height,base.resolution,base.origin_x,base.origin_y)
        self.occupied[:]=base.occupied
        self.base=base;self.margin=margin
        self.clearance=CellClearance(base.occupied,base.resolution,[base.origin_x,base.origin_y],geometry)

    def is_collision(self,state,local_x,local_y):
        if self.base.is_collision(state,local_x,local_y):return True
        pose=[state.x,state.y,state.yaw];poly=self.clearance.polygon(pose)
        if (poly[:,0].min()<=self.origin_x+self.margin or poly[:,0].max()>=self.origin_x+self.width-self.margin or
            poly[:,1].min()<=self.origin_y+self.margin or poly[:,1].max()>=self.origin_y+self.height-self.margin):return True
        return self.clearance.within(pose,self.margin)

class CertifiedRobustGrid(RobustCollisionGrid):
    """Fast conservative collision-layer mask with explicit covering bounds.

    The saved 0.20 m occupancy array remains unchanged. Its blocked rectangles
    are represented on an internal finer mask and ONLY dilated. A .1 m filled
    footprint sample cover and primitive between-state motion cover are added
    to the empirical .13 m geometric error envelope, with upward quantization.
    """
    def __init__(self,base,geometry,margin,planner_config,mask_resolution=.05):
        super().__init__(base,geometry,margin)
        h=base.resolution*.5
        sx=geometry.length/math.ceil(geometry.length/h)
        sy=geometry.width/math.ceil(geometry.width/h)
        self.footprint_sample_cover=math.hypot(sx,sy)/2
        substeps=max(1,math.ceil(planner_config.step_size/(base.resolution*.5)))
        substep=planner_config.step_size/substeps
        yaw_step=substep/geometry.wheelbase*math.tan(planner_config.max_steer)
        self.primitive_half_cover=substep/2+2*self.clearance.radius*math.sin(yaw_step/4)
        needed=margin+self.footprint_sample_cover+self.primitive_half_cover
        self.mask_resolution=mask_resolution
        self.dilation_cells=math.ceil(needed/mask_resolution)
        self.dilation_m=self.dilation_cells*mask_resolution
        ratio=round(base.resolution/mask_resolution)
        if not math.isclose(ratio*mask_resolution,base.resolution,abs_tol=1e-12):raise ValueError('Invalid collision-mask ratio')
        original=np.repeat(np.repeat(base.occupied,ratio,axis=0),ratio,axis=1)
        n=self.dilation_cells;pad=np.pad(original,n,constant_values=True)
        self.robust_mask=np.zeros_like(original)
        for dr in range(2*n+1):
            for dc in range(2*n+1):self.robust_mask |= pad[dr:dr+original.shape[0],dc:dc+original.shape[1]]
        self.audit=dict(geometry_error_margin_m=margin,footprint_sample_cover_m=self.footprint_sample_cover,
            primitive_step_m=substep,primitive_yaw_step_rad=yaw_step,primitive_half_motion_cover_m=self.primitive_half_cover,
            required_collision_layer_padding_m=needed,mask_resolution_m=mask_resolution,
            upward_rounded_square_padding_m=self.dilation_m,dilation_cells=n,
            saved_map_changed=False,original_planner_occupied_changed=False,
            guarantee='Every point of a perturbed full body is covered. Sampling-cover + between-primitive-state-cover + full pose-error envelope are enclosed by the dilated collision mask. This is stricter than the original checker, not a smaller physical vehicle.')

    def is_collision(self,state,local_x,local_y):
        if self.base.is_collision(state,local_x,local_y):return True
        c,s=math.cos(state.yaw),math.sin(state.yaw)
        x=state.x+c*local_x-s*local_y;y=state.y+s*local_x+c*local_y
        ix=np.floor((x-self.origin_x)/self.mask_resolution).astype(int)
        iy=np.floor((y-self.origin_y)/self.mask_resolution).astype(int)
        if np.any(ix<0) or np.any(iy<0) or np.any(ix>=self.robust_mask.shape[1]) or np.any(iy>=self.robust_mask.shape[0]):return True
        return bool(np.any(self.robust_mask[iy,ix]))
