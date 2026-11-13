"""Bounded-window correlative coarse search; ICP and its gates remain unchanged.

The occupancy distance score proposes initial poses, never error certificates.
No physical reference geometry/poses are accepted.
"""
import math
import numpy as np


def distance_field(occupied,resolution,limit=.6):
    d=np.where(occupied,0.,float(limit))
    for _ in range(math.ceil(limit/resolution)+2):
        old=d.copy();pad=np.pad(old,1,constant_values=limit)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            neighbor=pad[1+dr:1+dr+d.shape[0],1+dc:1+dc+d.shape[1]]
            d=np.minimum(d,neighbor+resolution*math.hypot(dr,dc))
        if np.array_equal(d,old):break
    return d


def correlative_match(points,prediction,states,config,translation_window=1.5,yaw_window_deg=10.):
    points=np.asarray(points,float);prediction=np.asarray(prediction,float)
    if len(points)<30:return dict(success=False,reason='insufficient_dual_returns',candidates=[])
    field=distance_field(states==1,config['resolution']);origin=np.array([config['origin_x'],config['origin_y']]);res=config['resolution']
    def search(center,offsets,angles,keep):
        xy=np.array([[center[0]+dx,center[1]+dy]for dx in offsets for dy in offsets]);winners=[]
        for da in angles:
            yaw=center[2]+da;c,s=math.cos(yaw),math.sin(yaw)
            rotated=points@np.array([[c,s],[-s,c]])
            world=xy[:,None,:]+rotated
            cell=np.floor((world-origin)/res).astype(int)
            inside=(cell[:,:,0]>=0)&(cell[:,:,0]<field.shape[1])&(cell[:,:,1]>=0)&(cell[:,:,1]<field.shape[0])
            distance=np.full(inside.shape,.6)
            distance[inside]=field[cell[:,:,1][inside],cell[:,:,0][inside]]
            # Mean Huber-style bounded distance plus support loss penalty.
            support=np.mean(distance<=.15,axis=1)
            score=np.mean(np.minimum(distance,.35)**2,axis=1)+.04*(1-support)
            for j in np.argsort(score)[:keep]:winners.append(dict(pose=[*xy[j],yaw],score=float(score[j]),support_ratio=float(support[j])))
        return sorted(winners,key=lambda x:x['score'])[:keep]
    coarse=search(prediction,np.arange(-translation_window,translation_window+.001,.1),
        np.deg2rad(np.arange(-yaw_window_deg,yaw_window_deg+.001,1.)),20)
    # Refine spatially diverse basins, not twenty nearly identical neighbors.
    basins=[]
    for candidate in coarse:
        if all(np.linalg.norm(np.array(candidate['pose'][:2])-b['pose'][:2])>.15 or abs(candidate['pose'][2]-b['pose'][2])>math.radians(2)for b in basins):basins.append(candidate)
        if len(basins)>=4:break
    refined=[]
    for b in basins:
        refined+=search(np.array(b['pose']),np.arange(-.1,.101,.025),np.deg2rad(np.arange(-1.,1.001,.25)),3)
    refined=sorted(refined,key=lambda x:x['score'])
    return dict(success=bool(refined),pose=refined[0]['pose']if refined else prediction.tolist(),
        candidates=refined,coarse_basins=basins,translation_window_m=translation_window,yaw_window_deg=yaw_window_deg,
        coarse_xy_step_m=.1,coarse_yaw_step_deg=1.,fine_xy_step_m=.025,fine_yaw_step_deg=.25,
        score_definition='mean min(chamfer_distance,.35)^2 + .04*(1-support<=.15m)',
        no_pose_or_map_bound_implied=True)
