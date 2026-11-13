"""P11.4.1 first continuous forward segment: real Bullet feedback, no replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pybullet as p
from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.simulation.pybullet.scene import ParkingScene
from parking_planner.simulation.pybullet.physics_vehicle import PhysicsVehicle
from parking_planner.simulation.pybullet.trajectory import load_execution,wrap
from parking_planner.simulation.pybullet.forward_tracker import ForwardTracker,first_forward_segment
from parking_planner.simulation.pybullet.hud import EdgeStatusBars
from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS

# Declared before running: no reference edits or post-hoc threshold relaxation.
LIMITS=dict(rms_cross_track_m=.08,max_cross_track_m=.15,mean_abs_heading_deg=3.,max_abs_heading_deg=6.,
            final_position_m=.10,final_heading_deg=3.,min_completion_percent=98.,max_roll_pitch_deg=5.,
            max_abs_speed_mps=.35,max_collision_episodes=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui',action='store_true')
    parser.add_argument('--hold',action='store_true')
    args=parser.parse_args()
    data=load_execution(ROOT/'results/p10_4_2_execution_trajectory.npz')
    ref,start,stop=first_forward_segment(data)
    scenario=build_perpendicular_parking_scenario()
    protected=[ROOT/'parking_planner/simulation/pybullet/physics_vehicle.py',
               ROOT/'parking_planner/control/stanley_controller.py',ROOT/'parking_planner/control/speed_controller.py']
    protected+=list((ROOT/'results').glob('p10*'))+list((ROOT/'results').glob('p11_3_basic*'))
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected if f.is_file()}
    cid=p.connect(p.GUI if args.gui else p.DIRECT,options=WINDOW_OPTIONS if args.gui else '')
    dt=1/240;out=ROOT/'results';rows=[];image=None
    try:
        p.setTimeStep(dt,physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=cid)
        scene=ParkingScene(cid,scenario)
        car=PhysicsVehicle(cid,scenario['geometry'],ref[0,:3])
        tracker=ForwardTracker(ref,scenario['geometry'].wheelbase)
        hud=None;trail_position=None
        if args.gui:
            scene.camera_target=[11.8,11.4,0.];scene.configure_gui();hud=EdgeStatusBars(cid)
            for a,b in zip(ref[:-1],ref[1:]):
                p.addUserDebugLine([*a[:2],.035],[*b[:2],.035],[.08,.3,1.],3,physicsClientId=cid)
        # Settle and pre-steer with real contact/braking; never reset pose/joints.
        for _ in range(480):
            car.command(0.,float(ref[0,3]),dt);p.stepSimulation(physicsClientId=cid)
        initial=car.state();collision_episodes=0;previous_contacts=set();completed=False;failure=None
        for step in range(30*240):
            wall=time.monotonic();state=car.state()
            motor,steering,errors,target=tracker.update(state,dt)
            car.command(motor,steering,dt)
            p.stepSimulation(physicsClientId=cid)
            actual=car.state();err=tracker.errors(actual)
            contacts={c[2] for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid) if c[2] in scene.obstacle_ids}
            collision_episodes+=len(contacts-previous_contacts);previous_contacts=contacts
            t=(step+1)*dt
            rows.append([t,actual['x'],actual['y'],actual['yaw'],actual['speed'],actual['steering'],motor,steering,target,
                         err['cross_track'],err['heading_error'],err['position_error'],err['arc'],actual['roll'],actual['pitch'],len(contacts)])
            if args.gui:
                if step%4==0:
                    point=[actual['x'],actual['y'],.055]
                    if trail_position is not None: p.addUserDebugLine(trail_position,point,[.05,.85,.3],3,physicsClientId=cid)
                    trail_position=point
                    hud.update(f'P11.4.1 | {tracker.stage} | {t:.2f}s | {100*tracker.progress/tracker.length:.1f}%',
                        f'v={actual["speed"]:+.3f} m/s | steer={math.degrees(actual["steering"]):+.1f} deg | yaw={math.degrees(actual["yaw"]):+.1f} deg | CTE={err["cross_track"]:+.3f}m | HE={math.degrees(err["heading_error"]):+.2f}deg',
                        'Forward-only physical tracking | ESC stop','BLUE reference / GREEN physical trace')
                if p.getKeyboardEvents(physicsClientId=cid).get(27,0)&p.KEY_WAS_TRIGGERED:
                    failure='USER_STOP';break
                time.sleep(max(0,dt-(time.monotonic()-wall)))
            if contacts: failure='OBSTACLE_COLLISION';break
            if max(abs(actual['roll']),abs(actual['pitch']))>math.radians(10): failure='UNSTABLE';break
            if tracker.stage=='COMPLETE': completed=True;break
        if not completed and failure is None: failure='TIMEOUT'
        # Real actuator braking only; final pose is still a physics result.
        for _ in range(240):
            car.command(0.,tracker.last_steering,dt);p.stepSimulation(physicsClientId=cid)
            actual=car.state()
            contacts={c[2] for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid) if c[2] in scene.obstacle_ids}
            collision_episodes+=len(contacts-previous_contacts);previous_contacts=contacts
            err=tracker.errors(actual);t+=dt
            rows.append([t,actual['x'],actual['y'],actual['yaw'],actual['speed'],actual['steering'],0.,0.,0.,
                         err['cross_track'],err['heading_error'],err['position_error'],err['arc'],actual['roll'],actual['pitch'],len(contacts)])
        trace=np.array(rows);final=car.state()
        cte=np.abs(trace[:,9]);head=np.abs(np.degrees(trace[:,10]))
        endpoint=float(np.linalg.norm(np.array([final['x'],final['y']])-ref[-1,:2]))
        endyaw=abs(math.degrees(float(wrap(final['yaw']-ref[-1,2]))))
        completion=float(100*max(trace[:,12])/tracker.length)
        metrics=dict(mean_abs_cross_track_m=float(np.mean(cte)),rms_cross_track_m=float(np.sqrt(np.mean(cte**2))),
            max_cross_track_m=float(np.max(cte)),mean_abs_heading_deg=float(np.mean(head)),max_abs_heading_deg=float(np.max(head)),
            mean_position_error_m=float(np.mean(trace[:,11])),max_position_error_m=float(np.max(trace[:,11])),
            final_position_error_m=endpoint,final_heading_error_deg=endyaw,completion_percent=float(completion),
            obstacle_collision_episodes=collision_episodes,max_abs_roll_deg=float(np.degrees(np.max(np.abs(trace[:,13])))),
            max_abs_pitch_deg=float(np.degrees(np.max(np.abs(trace[:,14])))),max_abs_speed_mps=float(np.max(np.abs(trace[:,4]))),
            final_signed_speed_mps=final['speed'],minimum_signed_speed_mps=float(np.min(trace[:,4])))
        checks=dict(completed=completed,rms_cross_track=metrics['rms_cross_track_m']<=LIMITS['rms_cross_track_m'],
            max_cross_track=metrics['max_cross_track_m']<=LIMITS['max_cross_track_m'],
            mean_heading=metrics['mean_abs_heading_deg']<=LIMITS['mean_abs_heading_deg'],
            max_heading=metrics['max_abs_heading_deg']<=LIMITS['max_abs_heading_deg'],
            endpoint=endpoint<=LIMITS['final_position_m'],end_heading=endyaw<=LIMITS['final_heading_deg'],
            completion=completion>=LIMITS['min_completion_percent'],no_collision=collision_episodes==0,
            stable=max(metrics['max_abs_roll_deg'],metrics['max_abs_pitch_deg'])<=LIMITS['max_roll_pitch_deg'],
            low_speed=metrics['max_abs_speed_mps']<=LIMITS['max_abs_speed_mps'],
            stopped=abs(final['speed'])<.025,forward_only=bool(np.min(trace[:,4])>-.01),
            protected_unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items()))
        summary=dict(stage='P11.4.1 forward-only physical tracking',status='PASS' if all(checks.values()) else 'REVIEW',
            metrics=metrics,checks=checks,limits=LIMITS,failure_reason=failure,physics_hz=240,control_hz=240,
            reference_source=data.metadata,reference_indices_inclusive=[start,stop],reference_samples=len(ref),
            reference_start=ref[0].tolist(),reference_end=ref[-1].tolist(),reference_length_m=tracker.length,
            reference_original_duration_s=float(ref[-1,6]-ref[0,6]),reference_unchanged=bool(np.array_equal(ref,data.samples[start:stop+1])),
            extraction='First direction=+1 and speed>0 frame through FIRST zero-speed frame; no subsequent switch/reverse samples.',
            controller=dict(lateral='Existing StanleyController, physical rear-axle feedback and front-axle tracking',gain=.65,softening_speed=.45,
                longitudinal='Existing SpeedController acceleration PID integrated into bounded motor-speed command',kp=2.5,ki=.35,kd=.02,
                max_motor_speed_mps=.30,speed_profile='Recorded speed shape normalized to 0.28 m/s peak plus distance-to-end braking envelope',
                note='New closed-loop run; original timestamps retained in reference but not used as a replay clock.'),
            physics_parameters=car.parameters(),initial_after_settle=initial,final=final,
            prior_attempt=dict(metrics='p11_4_1_attempt01_metrics.json',trace='p11_4_1_attempt01_trace.npz',
                finding='Immediate steering unwind while braking caused backward tire scrub; hold final steering through standstill. Physics, reference and acceptance limits unchanged.'),
            vehicle_pose_reset_after_spawn=False,external_motion_forces=False,protected_sha256=hashes,gui_run=args.gui,gui_camera_distance=scene.camera_distance if args.gui else None)
        np.savez_compressed(out/'p11_4_1_forward_tracking_trace.npz',trace=trace,reference=ref,
            columns=np.array(['time_s','rear_x','rear_y','yaw_rad','signed_speed_mps','mean_front_steer_rad',
                'motor_speed_mps','steer_command_rad','target_speed_mps','signed_cte_m','heading_error_rad',
                'position_error_m','nearest_arc_m','roll_rad','pitch_rad','obstacle_contacts']))
        if args.gui:
            hud.update(f'P11.4.1 | {summary["status"]} / STOPPED | {t:.2f}s | {completion:.1f}%',
                f'v={final["speed"]:+.3f}m/s | steer={math.degrees(final["steering"]):+.1f}deg | CTE={cte[-1]:.3f}m | HE={head[-1]:.2f}deg',
                'Forward segment complete; no reverse or parking | ESC exit','BLUE reference / GREEN physical trace')
            camera=p.getDebugVisualizerCamera(physicsClientId=cid)
            rgba=p.getCameraImage(1100,800,viewMatrix=camera[2],projectionMatrix=camera[3],renderer=p.ER_BULLET_HARDWARE_OPENGL,physicsClientId=cid)[2]
            image=np.array(rgba,dtype=np.uint8).reshape(800,1100,4)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
        if image is not None: axes[0,0].imshow(image);axes[0,0].axis('off');axes[0,0].set_title('Physical car after forward segment')
        else:
            axes[0,0].plot(trace[:,0],trace[:,4],label='Actual speed');axes[0,0].plot(trace[:,0],trace[:,8],label='Target speed');axes[0,0].legend();axes[0,0].set(xlabel='Time (s)',ylabel='Speed (m/s)')
        axes[0,1].plot(ref[:,0],ref[:,1],'b-',label='Original forward reference')
        axes[0,1].plot(trace[:,1],trace[:,2],color='green',label='Physical rear axle')
        axes[0,1].set(xlabel='X (m)',ylabel='Y (m)',title='Unmodified reference vs physics',aspect='equal');axes[0,1].legend()
        axes[1,0].plot(trace[:,0],trace[:,9]);axes[1,0].set(xlabel='Time (s)',ylabel='Signed CTE (m)',title='Cross-track error')
        axes[1,1].plot(trace[:,0],np.degrees(trace[:,10]));axes[1,1].set(xlabel='Time (s)',ylabel='Heading error (deg)',title='Heading error')
        for ax in (axes[0,1],axes[1,0],axes[1,1]): ax.grid(alpha=.25)
        fig.suptitle(f'P11.4.1 forward-only closed-loop physical tracking | {summary["status"]}')
        fig.savefig(out/'p11_4_1_forward_tracking.png',dpi=150);plt.close(fig)
        (out/'p11_4_1_forward_tracking_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        report=f'''# P11.4.1 前进段闭环物理跟踪\n\n结果：{summary['status']}；失败原因：{failure}。\n\n从原执行记录提取第 {start}–{stop} 帧（含首次停车），{len(ref)} 点，长度 {tracker.length:.4f} m；保存原后轴位姿、航向、转角、速度、方向、时间，未修改 P10 数据。物理车从首帧创建，静置及预转向两秒后闭环启动；其后没有车身/关节状态重置，没有额外运动外力。\n\n复用原 Stanley（gain=.65, softening=.45）及 PID（Kp=2.5, Ki=.35, Kd=.02）。PID 加速度输出积分为电机速度命令，仅通过 PhysicsVehicle.command 驱动。速度轮廓使用原记录相对形状，峰值配置 0.28 m/s，命令上限 0.30 m/s；起步/终点允许低于巡航速度。控制和物理步进均 240 Hz。\n\n误差针对原后轴折线最近投影与插值车身航向计算；Stanley 前进控制本身使用前轴误差。完成率按原路径弧长投影计算，不按仿真时间虚报。通过标准在运行前声明于脚本 LIMITS；失败样本未删除。\n\n|指标|实际值|\n|---|---:|\n|平均绝对横向误差 m|{metrics['mean_abs_cross_track_m']:.6f}|\n|RMS 横向误差 m|{metrics['rms_cross_track_m']:.6f}|\n|最大横向误差 m|{metrics['max_cross_track_m']:.6f}|\n|平均绝对航向误差 °|{metrics['mean_abs_heading_deg']:.6f}|\n|最大绝对航向误差 °|{metrics['max_abs_heading_deg']:.6f}|\n|最终位置误差 m|{endpoint:.6f}|\n|最终航向误差 °|{endyaw:.6f}|\n|完成率 %|{completion:.4f}|\n|障碍物碰撞事件|{collision_episodes}|\n|最大滚转/俯仰 °|{max(metrics['max_abs_roll_deg'],metrics['max_abs_pitch_deg']):.6f}|\n\n首轮未通过：停车回正前轮引起轮胎接触力后挪，最低速度 -0.03463 m/s。原始 JSON/NPZ 保留为 p11_4_1_attempt01_*。修正为制动及停车期间保持末次转角；未修改动力学、参考轨迹和验收阈值。

完整通过条件与所有检查结果见 JSON，完整真实状态及原参考片段见 NPZ。参考长度很短且主要为单向恒曲率转弯，本次通过不能证明倒车或完整泊车性能。\n\n运行：`conda run --no-capture-output -n parkingplanner python scripts/run_p11_4_1_forward_tracking.py --gui --hold`。\n\n本阶段在前进段结束后停止，未实现倒车、完整自主泊车或后续阶段。\n'''
        (out/'p11_4_1_forward_tracking_report.md').write_text(report,encoding='utf-8')
        print(json.dumps({k:summary[k] for k in ('status','metrics','checks','failure_reason')},indent=2),flush=True)
        if args.gui and args.hold:
            while p.isConnected(cid):
                if p.getKeyboardEvents(physicsClientId=cid).get(27,0)&p.KEY_WAS_TRIGGERED: break
                time.sleep(.05)
    finally:
        if p.isConnected(cid): p.disconnect(cid)


if __name__=='__main__': main()
