"""P11.3 foundation only: physical wheel-drive tests, no parking controller."""
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
from parking_planner.simulation.pybullet.hud import EdgeStatusBars
from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS
from parking_planner.simulation.pybullet.trajectory import load_execution,wrap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui',action='store_true',help='Run the same physical tests in real time with snapshots')
    parser.add_argument('--hold',action='store_true',help='Hold GUI after tests until ESC or window close')
    parser.add_argument('--output-prefix',default='p11_3_basic_motion',help='Independent result prefix for environment regression')
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_]+',args.output_prefix): parser.error('Invalid result prefix')
    scenario=build_perpendicular_parking_scenario()
    reference=load_execution(ROOT/'results/p10_4_2_execution_trajectory.npz')
    # Audit accepted inputs/code. ReplayView is intentionally display-only and
    # is excluded: its HUD layout is the one explicitly authorized change.
    protected=list((ROOT/'results').glob('p10*'))+[ROOT/'scripts/run_p11_2_pybullet_replay.py',
        ROOT/'parking_planner/simulation/pybullet/vehicle.py',ROOT/'parking_planner/simulation/pybullet/trajectory.py']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected if f.is_file()}
    cid=p.connect(p.GUI if args.gui else p.DIRECT,options=WINDOW_OPTIONS if args.gui else '')
    if cid<0: raise RuntimeError('Cannot connect to PyBullet')
    out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[];snapshots=[];summary={}
    try:
        dt=1/240
        p.setTimeStep(dt,physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=cid)
        scene=ParkingScene(cid,scenario)
        car=PhysicsVehicle(cid,scenario['geometry'],(4.,4.,0.))
        hud=None
        if args.gui:
            scene.camera_target=[8.,7.,0.];scene.configure_gui()
            hud=EdgeStatusBars(cid)
        phases=[('SETTLE',2.,0.,0.),('FORWARD_STRAIGHT',4.,.6,0.),('BRAKE',1.5,0.,0.),
                ('FORWARD_LEFT',3.,.6,.3),('REVERSE_LEFT_INTERLOCK',4.,-.6,.3),('FINAL_STOP',2.5,0.,0.)]
        total=sum(item[1] for item in phases)
        summaries=[];elapsed=0.;contact_episodes=0;previous_contacts=set();max_tilt=0.;shift_entry_speeds=[]
        captured_phases={'FORWARD_STRAIGHT','FORWARD_LEFT','REVERSE_LEFT_INTERLOCK','FINAL_STOP'}
        previous_direction=car.direction
        def screenshot(label):
            camera=p.getDebugVisualizerCamera(physicsClientId=cid)
            image=p.getCameraImage(1100,800,viewMatrix=camera[2],projectionMatrix=camera[3],
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL,physicsClientId=cid)
            pixels=np.asarray(image[2],dtype=np.uint8).reshape(800,1100,4)
            # Mirror actual HUD text, which getCameraImage omits on some drivers.
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig=Figure(figsize=(11,8),dpi=100)
            canvas=FigureCanvasAgg(fig)
            ax=fig.add_axes([0,0,1,1]);ax.imshow(pixels);ax.axis('off')
            main,telemetry,hints,legend=hud.last_content
            for text,x,y,size in [(main,.02,.955,11),(telemetry,.02,.075,10),
                                  (hints,.02,.022,8),(legend,.66,.135,8)]:
                fig.text(x,y,text,color='white',fontsize=size)
            canvas.draw()
            return np.asarray(canvas.buffer_rgba()).copy()
        for phase,duration,speed,steer in phases:
            begin=car.state();min_speed=1e6;max_speed=-1e6;seen=set();entry_speed=[]
            for step in range(round(duration/dt)):
                wall=time.monotonic()
                before=car.state()
                car.command(speed,steer,dt)
                if car.direction!=previous_direction:
                    shift_entry_speeds.append(abs(before['speed']))
                    previous_direction=car.direction
                p.stepSimulation(physicsClientId=cid)
                state=car.state();elapsed+=dt
                min_speed=min(min_speed,state['speed']);max_speed=max(max_speed,state['speed']);seen.add(state['phase'])
                max_tilt=max(max_tilt,abs(state['roll']),abs(state['pitch']))
                contacts=p.getContactPoints(bodyA=car.body,physicsClientId=cid)
                obstacles={c[2] for c in contacts if c[2] in scene.obstacle_ids}
                contact_episodes+=len(obstacles-previous_contacts);previous_contacts=obstacles
                rows.append([elapsed,state['x'],state['y'],state['yaw'],state['speed'],state['steering'],
                             state['roll'],state['pitch'],state['direction'],car.command_speed,steer,len(obstacles)])
                if args.gui:
                    keys=p.getKeyboardEvents(physicsClientId=cid)
                    if keys.get(27,0)&p.KEY_WAS_TRIGGERED: raise KeyboardInterrupt
                    if step%8==0 or step+1==round(duration/dt):
                        hud.update(f'P11.3 BASIC TEST | {phase} | {elapsed:.2f}/{total:.2f}s | {100*elapsed/total:.1f}%',
                                   f'v={state["speed"]:+.3f} m/s | steer={math.degrees(state["steering"]):+.1f} deg | yaw={math.degrees(state["yaw"]):+.1f} deg',
                                   'Physical wheel motors + stepSimulation | ESC exit | no trajectory controller',
                                   'BLUE chassis / RED rear lights')
                    time.sleep(max(0,dt-(time.monotonic()-wall)))
            end=car.state()
            summaries.append(dict(phase=phase,initial=begin,final=end,min_signed_speed_mps=min_speed,max_signed_speed_mps=max_speed,
                                  actuator_phases=sorted(seen),displacement_m=math.hypot(end['x']-begin['x'],end['y']-begin['y']),
                                  yaw_change_rad=float(wrap(end['yaw']-begin['yaw']))))
            if args.gui and phase in captured_phases:
                snapshots.append((phase,screenshot(phase)))
        by_name={item['phase']:item for item in summaries}
        contact_links={c[3] for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid) if c[2] not in scene.obstacle_ids and c[9]>0}
        inertia=car.parameters()['chassis_inertia_kg_m2']
        checks=dict(positive_mass_and_inertia=car.parameters()['total_mass_kg']>0 and min(inertia)>0,
            four_wheels_grounded=all(j in contact_links for j in car.wheel_joints),
            stable_roll_pitch=max_tilt<math.radians(5),
            forward_translation=by_name['FORWARD_STRAIGHT']['displacement_m']>1.,
            forward_speed_positive=by_name['FORWARD_STRAIGHT']['max_signed_speed_mps']>.3,
            straight_yaw_error=abs(by_name['FORWARD_STRAIGHT']['yaw_change_rad'])<.03,
            braking_stops=abs(by_name['BRAKE']['final']['speed'])<.035,
            front_steering_tracks=abs(by_name['FORWARD_LEFT']['final']['steering']-.3)<.06,
            forward_left_turn=by_name['FORWARD_LEFT']['yaw_change_rad']>.07,
            reverse_speed_negative=by_name['REVERSE_LEFT_INTERLOCK']['min_signed_speed_mps']<-.3,
            reverse_left_yaw_correct=by_name['REVERSE_LEFT_INTERLOCK']['yaw_change_rad']<-.07,
            shift_stops_before_reversing=bool(shift_entry_speeds) and max(shift_entry_speeds)<=car.config.shift_stop_speed_mps,
            reverse_interlock_exercised='STOP_BEFORE_SHIFT' in by_name['REVERSE_LEFT_INTERLOCK']['actuator_phases'],
            final_stop=abs(car.state()['speed'])<.035,no_obstacle_collision=contact_episodes==0)
        trace=np.asarray(rows)
        protected_unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
        checks['protected_sources_unchanged']=protected_unchanged
        summary=dict(stage='P11.3 physical model and basic motion ONLY',status='PASS' if all(checks.values()) else 'REVIEW',
            physics=dict(timestep_s=dt,solver_iterations=100,gravity_mps2=9.81,pose_reset_after_creation=False,
                         actuation='front steering position motors; rear wheel velocity motors; four-wheel braking; no external motion forces'),
            parameters=car.parameters(),reference_inspection=dict(samples=len(reference.samples),duration_s=reference.duration,
                body_reference='rear axle center',steering_source='steering_deg converted to radians',
                actual_speed_source='speed_mps (signed)',direction_values=[-1,0,1],
                used_for_tracking=False),checks=checks,phases=summaries,
            max_roll_pitch_deg=math.degrees(max_tilt),collision_episodes=contact_episodes,
            shift_entry_abs_speeds_mps=shift_entry_speeds,final_state=car.state(),
            trajectory_tracking_position_error=None,trajectory_tracking_heading_error=None,lateral_tracking_error=None,
            max_tracking_error=None,final_parking_position_error=None,final_parking_heading_error=None,
            parking_success=None,not_run_reason='User requested a pause after physical-model and basic-motion validation; closed-loop parking not implemented.',
            protected_sha256=hashes,gui_test=args.gui)
        np.savez_compressed(out/f'{args.output_prefix}_trace.npz',trace=trace,
            columns=np.array(['time_s','rear_x','rear_y','yaw_rad','signed_speed_mps','mean_front_steer_rad',
                              'roll_rad','pitch_rad','gear','applied_motor_speed_mps','requested_steer_rad','obstacle_contacts']))
        (out/f'{args.output_prefix}_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        if snapshots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
            for ax,(label,pixels) in zip(axes.flat,snapshots): ax.imshow(pixels);ax.set_title(label);ax.axis('off')
            fig.suptitle('P11.3 physical wheel-drive basic tests (NOT autonomous parking)')
            fig.savefig(out/f'{args.output_prefix}.png',dpi=130);plt.close(fig)
        report=f'''# P11.3 基础物理模型阶段\n\n结果：{summary['status']}。本阶段不包含闭环轨迹跟踪或完整泊车。\n\n车辆外廓 4.5×1.8 m，轴距 2.7 m；车身质量 1100 kg，轮胎各 18 kg，转向节各 2 kg，总质量 {car.parameters()['total_mass_kg']:.0f} kg。轮胎半径 0.32 m，轮距 {car.track:.2f} m。车身惯性由 Bullet 碰撞形状和质量计算，见 JSON。\n\n前轮为 Z 轴转向关节与 Y 轴滚动关节串联，后轮 Y 轴电机驱动。使用几何 Ackermann 内外轮角；转角指令限幅 ±35°。轮胎摩擦系数 1.2，驱动转矩上限 160 N·m/后轮，制动 300 N·m/轮。倒挡前必须减速至 0.035 m/s 以下并保持 0.35 秒。此为刚性底盘、无悬架、简化库仑摩擦的低速基础模型，尚未标定真实轮胎与悬架特性。\n\n固定步长 1/240 s，100 次求解迭代，重力 9.81 m/s²。车辆仅在创建时指定位置，运动期间不重置车身或关节状态，不施加人为平移/偏航力。\n\n实验依次执行：落地稳定、直行、制动、左转前进、制动换挡并倒车左转、最终制动。障碍物碰撞事件：{contact_episodes}；最大滚转/俯仰：{summary['max_roll_pitch_deg']:.4f}°。完整阶段数据、通过条件及真实物理状态见 JSON/NPZ。\n\nP10 数据只检查格式，未作为运动目标。轨迹误差、最终停车误差和入库成功率均未测量，JSON 保留 null，不宣称完成自主泊车。\n\n运行：`conda run --no-capture-output -n parkingplanner python scripts/run_p11_3_physics_basic.py --gui --hold`。\n\n基础运动测试完成后暂停，等待人工检查。\n'''
        (out/f'{args.output_prefix}_report.md').write_text(report,encoding='utf-8')
        print(json.dumps({k:summary[k] for k in ('status','checks','max_roll_pitch_deg','collision_episodes','shift_entry_abs_speeds_mps','final_state')},indent=2),flush=True)
        if args.gui and args.hold:
            while p.isConnected(cid):
                if p.getKeyboardEvents(physicsClientId=cid).get(27,0)&p.KEY_WAS_TRIGGERED: break
                time.sleep(.05)
    finally:
        if p.isConnected(cid): p.disconnect(cid)


if __name__=='__main__': main()
