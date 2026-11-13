"""Fresh, rear-axle-local dual-LiDAR frontend; no historical/world pose input."""
import math
from dataclasses import replace
import numpy as np
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper
from parking_planner.localization.point_to_line_icp import PointToLineConfig,point_to_line_icp
from parking_planner.slam.online_slam import wrap_angle


class FreshLocalSLAM:
    def __init__(self,dual,initial_imu_heading):
        self.dual=dual;self.heading_zero=initial_imu_heading;self.pose=np.zeros(3)
        self.mapper=OccupancyGridMapper(width=24.,height=24.,resolution=.05,origin_x=-12.,origin_y=-12.)
        self.config=PointToLineConfig();self.history=[];self.distance_since_scan=0.;self.travel=0.;self.lost=0;self.dead_reckoning_distance=0.;self.degenerate_streak=0

    def predict(self,m):
        yaw=wrap_angle(m['heading_rad']-self.heading_zero)
        # Gyro-propagated heading plus a weak absolute relative-IMU correction.
        predicted=wrap_angle(self.pose[2]+m['gyro_rad_s']*m['dt'])
        self.pose[2]=wrap_angle(predicted+.03*wrap_angle(yaw-predicted))
        ds=m['forward_increment_m'];self.pose[:2]+=ds*np.array([math.cos(self.pose[2]),math.sin(self.pose[2])])
        self.distance_since_scan+=abs(ds);self.travel+=abs(ds)

    def observe(self,packet,t):
        prior=self.pose.copy();match=None;accepted=False
        rc=np.column_stack(np.where(self.mapper.states==1))
        if len(rc)>=40 and len(packet['points_local'])>=self.config.min_correspondences:
            points=np.column_stack((self.mapper.origin_x+(rc[:,1]+.5)*self.mapper.resolution,self.mapper.origin_y+(rc[:,0]+.5)*self.mapper.resolution))
            # Bounded local submap; original PCA, ICP and acceptance gates unchanged.
            keep=np.linalg.norm(points-self.pose[:2],axis=1)<12.1;points=points[keep]
            # Encoder increments are only ~2 cm per scan, not an 8 cm random
            # pose jump. The prior scale includes quantization and a 2% rolling
            # uncertainty model; acceptance thresholds remain EXACTLY unchanged.
            match_config=replace(self.config,prior_translation_std_m=.003+.02*self.distance_since_scan,
                prior_heading_std_rad=math.radians(.1))
            match=point_to_line_icp(packet['points_local'],self.pose,points,match_config)
            correction=np.asarray(match['pose'])-self.pose;correction[2]=wrap_angle(correction[2])
            accepted=bool(match['success']and np.linalg.norm(correction[:2])<.06 and abs(correction[2])<math.radians(.5))
            if accepted:
                # A stationary scan cannot move the local gauge: noisy grid
                # centroids otherwise create fictitious translation over time.
                if self.distance_since_scan>.005:
                    self.pose[:2]+=.25*correction[:2]
                    self.pose[2]=wrap_angle(self.pose[2]+.10*correction[2])
                self.lost=0;self.dead_reckoning_distance=0.;self.degenerate_streak=0
            else:
                # Parallel walls cannot constrain along-wall translation. Do
                # not mark rejected ICP as accepted or use its candidate pose.
                # Good surface support + encoder/IMU permits LIMITED propagation.
                partial=bool(match['degenerate']and match['correspondences']>=self.config.min_correspondences
                    and match['fitness']>=self.config.min_match_ratio and match['line_rmse_m']is not None
                    and match['line_rmse_m']<=self.config.max_line_rmse_m)
                self.dead_reckoning_distance+=self.distance_since_scan
                if partial:self.degenerate_streak+=1;self.lost=0
                else:self.lost+=1
        # Initial/fallback updates use the fresh relative encoder/IMU frame;
        # prolonged loss or large odometry distance is stopped by the supervisor.
        self.dual.update_map(self.mapper,self.pose,packet)
        self.history.append(dict(time_s=t,pose=self.pose.tolist(),prediction=prior.tolist(),accepted=accepted,
            bootstrapped=len(self.history)==0,match=match,odometry_distance_since_scan_m=self.distance_since_scan,
            registration_loss_streak=self.lost,degenerate_streak=self.degenerate_streak,dead_reckoning_distance_m=self.dead_reckoning_distance))
        self.distance_since_scan=0.
        return self.history[-1]
