"""Map-only parallel-wall bay recognition and exact conservative sweep layer."""
import math
import numpy as np
from parking_planner.planning.hybrid_astar import OccupancyGrid
from parking_planner.planning.robust_clearance import CellClearance


def recognize_bay(mapper,current_pose,geometry):
    r,c=np.where(mapper.states==1)
    points=np.column_stack((mapper.origin_x+(c+.5)*mapper.resolution,mapper.origin_y+(r+.5)*mapper.resolution))
    if len(points)<30:raise ValueError('Not enough occupied surface evidence; continue exploration')
    rng=np.random.default_rng(20261119);lines=[]
    # Deterministic RANSAC long surfaces; no scene rectangles/world coordinates.
    for _ in range(800):
        a,b=points[rng.choice(len(points),2,replace=False)];edge=b-a
        if np.linalg.norm(edge)<2.:continue
        d=edge/np.linalg.norm(edge)
        if d[0]<0:d=-d
        n=np.array([-d[1],d[0]]);offset=float(a@n);mask=np.abs(points@n-offset)<.06
        support=points[mask]
        if len(support)<16:continue
        vals=support@d;lo,hi=np.quantile(vals,[.02,.98])
        if hi-lo<3.3:continue
        # Refine orientation on line inliers only, not an entire L-shaped car.
        cov=np.cov(support.T);_,v=np.linalg.eigh(cov);d=v[:,-1]
        if d[0]<0:d=-d
        n=np.array([-d[1],d[0]]);offset=float(np.median(support@n));vals=support@d;lo,hi=np.quantile(vals,[.01,.99])
        if all(abs(offset-x['offset'])>.15 or abs(float(d@x['d']))<.995 for x in lines):
            lines.append(dict(d=d,n=n,offset=offset,lo=float(lo),hi=float(hi),count=len(support)))
    candidates=[]
    for i,a in enumerate(lines):
        for b in lines[i+1:]:
            if float(a['d']@b['d'])<.999:continue
            d=a['d']+b['d'];d/=np.linalg.norm(d);n=np.array([-d[1],d[0]])
            offs=sorted([a['offset'],b['offset']]);width=offs[1]-offs[0]
            if not geometry.width+.50<width<geometry.width+1.6:continue
            lo=max(a['lo'],b['lo']);hi=min(a['hi'],b['hi'])
            if hi-lo<geometry.length-.45:continue
            center=d*((lo+hi)/2)+n*np.mean(offs)
            # Car faces out toward the current aisle position; reverse enters bay.
            if float(d@(current_pose[:2]-center))<0:d=-d
            yaw=math.atan2(d[1],d[0]);rear=center-(geometry.length/2-geometry.rear_overhang)*d
            # Conservative recognized slot lies within observed long-face extent,
            # with a small measured-free end reserve, verified again by planner.
            axis=a['d'];normal=a['n'];padding=.30
            polygon=np.array([axis*s+normal*q for s,q in [(lo-padding,offs[0]),(hi+padding,offs[0]),(hi+padding,offs[1]),(lo-padding,offs[1])]])
            candidates.append(dict(goal=[*rear,yaw],center=center.tolist(),slot_polygon=polygon.tolist(),
                width_m=width,observed_depth_m=hi-lo,support_counts=[a['count'],b['count']],
                source='Fresh occupied long-line pair only; no historical target/scene obstacle coordinates.',score=abs(width-(geometry.width+.8))))
    if not candidates:raise ValueError('No unambiguous usable parallel-wall bay; explore more or click center then heading')
    candidates.sort(key=lambda x:x['score'])
    return candidates[0]


class LocalSweepGrid(OccupancyGrid):
    def __init__(self,mapper,geometry,margin,primitive_half_cover):
        super().__init__(mapper.width,mapper.height,mapper.resolution,mapper.origin_x,mapper.origin_y)
        self.occupied[:]=mapper.states!=0
        self.geometry=geometry;self.margin=margin+primitive_half_cover
        self.index=CellClearance(np.zeros((1,1),bool),self.resolution,[self.origin_x,self.origin_y],geometry)

    def is_collision(self,state,local_x,local_y):
        pose=[state.x,state.y,state.yaw];poly=self.index.polygon(pose);m=self.margin
        lo=poly.min(axis=0)-m;hi=poly.max(axis=0)+m
        x0,y0=np.floor((lo-[self.origin_x,self.origin_y])/self.resolution).astype(int)
        x1,y1=np.floor((hi-[self.origin_x,self.origin_y])/self.resolution).astype(int)
        if min(x0,y0)<0 or x1>=self.cols or y1>=self.rows:return True
        r,c=np.where(self.occupied[y0:y1+1,x0:x1+1])
        if not len(r):return False
        x=self.origin_x+(c+x0)*self.resolution;y=self.origin_y+(r+y0)*self.resolution
        boxes=np.column_stack((x,y,x+self.resolution,y+self.resolution))
        return bool(np.any(self.index.exact_distances(pose,boxes)<=m+1e-12))
