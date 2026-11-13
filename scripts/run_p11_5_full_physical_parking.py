"""P11.5 unified complete wheel-driven physical parking, with read-only P10 input."""
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
from parking_planner.simulation.pybullet.parking_state_machine import ParkingStateMachine,split_recorded_segments
from parking_planner.simulation.pybullet.full_parking_executor import LIMITS,body_and_bay,reverse_sign_checks
from parking_planner.simulation.pybullet.hud import EdgeStatusBars
from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS

PREFIX='p11_5_full_physical_parking'
COLUMNS=['time_s','rear_x','rear_y','body_yaw_rad','actual_signed_speed_mps','target_signed_speed_mps',
    'requested_motor_speed_mps','applied_motor_speed_mps','actual_mean_front_steer_rad','command_bicycle_steer_rad',
    'gear','segment','signed_cte_m','heading_error_rad','position_error_m','segment_arc_m','roll_rad','pitch_rad',
    'effective_collision_count','min_body_bay_margin_m','painted_side_line_crossing']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui',action='store_true')
    parser.add_argument('--hold',action='store_true')
    args=parser.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    data=load_execution(out/'p10_4_2_execution_trajectory.npz')
    segments=split_recorded_segments(data)
    scenario=build_perpendicular_parking_scenario()
    protocol_file=out/f'{PREFIX}_protocol.json'
    protected=list((ROOT/'results').glob('p10*'))+list((ROOT/'results').glob('p11_4_1*'))
    protected += [ROOT/'parking_planner/simulation/pybullet/physics_vehicle.py',ROOT/'parking_planner/simulation/pybullet/forward_tracker.py',
                  ROOT/'parking_planner/control/stanley_controller.py',ROOT/'parking_planner/control/speed_controller.py']
    if protocol_file.exists():
        protocol=json.loads(protocol_file.read_text(encoding='utf-8'))
        if protocol['limits']!=LIMITS: raise RuntimeError('Thresholds differ from predeclared protocol')
        for f,h in protocol['protected_sha256'].items():
            if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h: raise RuntimeError(f'Protected baseline changed: {f}')
    else:
        protocol=dict(limits=LIMITS,protected_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected if f.is_file()},
            test_sequence='three isolated reverse checks, then one full closed-loop run',
            bay_policy='Painted bay lines may be crossed during approach; final full-body containment and clearance are mandatory. Peripheral scenery has no collision shapes and is excluded.')
        protocol_file.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    checks_file=out/'p11_5_reverse_checks.json'
    if checks_file.exists(): diagnostic=json.loads(checks_file.read_text(encoding='utf-8'))
    else:
        cases=reverse_sign_checks(scenario);diagnostic=dict(cases=cases,all_passed=all(c['passed'] for c in cases))
        checks_file.write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    if not diagnostic['all_passed']: raise RuntimeError('Reverse diagnostics failed; full experiment refused')
    print('PREDECLARED LIMITS:',json.dumps(LIMITS),flush=True)
    print('REFERENCE SEGMENTS:',[(s.direction,s.start,s.stop,s.hold_after_s) for s in segments],flush=True)
    cid=p.connect(p.GUI if args.gui else p.DIRECT,options=WINDOW_OPTIONS if args.gui else '')
    if cid<0:raise RuntimeError('Cannot create PyBullet client')
    originals={};forbidden=[];snapshots={};rows=[];phases=[];collision_events=[];gear_events=[];interactive=[]
    try:
        dt=1/240
        p.setTimeStep(dt,physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=cid)
        scene=ParkingScene(cid,scenario)
        car=PhysicsVehicle(cid,scenario['geometry'],data.samples[0,:3])
        machine=ParkingStateMachine(segments,scenario['geometry'].wheelbase)
        # Runtime guard is scoped to the PHYSICAL vehicle. HUD panels may move.
        for name in ('resetBasePositionAndOrientation','resetBaseVelocity','resetJointState','applyExternalForce','applyExternalTorque'):
            original=getattr(p,name);originals[name]=original
            def guarded(*a,_name=name,_fn=original,**kw):
                body=a[0] if a else kw.get('bodyUniqueId',kw.get('objectUniqueId'))
                if body==car.body:
                    forbidden.append(_name)
                    raise RuntimeError(f'Forbidden vehicle motion API: {_name}')
                return _fn(*a,**kw)
            setattr(p,name,guarded)
        hud=None;last_point=None;camera=None
        if args.gui:
            scene.camera_target=[11.8,11.4,0.];scene.configure_gui();hud=EdgeStatusBars(cid)
            for segment in segments:
                color=[.08,.3,1.] if segment.direction>0 else [1.,.48,.08]
                for a,b in zip(segment.reference[:-1],segment.reference[1:]):
                    p.addUserDebugLine([*a[:2],.035],[*b[:2],.035],color,2,physicsClientId=cid)
        previous_contacts=set();last_gear=car.direction;t=0.;paused=False;quit_after=False
        last_announced=None;last_print=-1.;max_tire_margin_min=float('inf')
        def record(state,phase,index,motor,target,steer):
            nonlocal previous_contacts,t,last_gear,max_tire_margin_min
            k=max(0,index)
            err=machine.trackers[k].errors(state)
            polygon,margins,sat,outside,paint=body_and_bay(state,scenario)
            bullet={scene.obstacle_ids.index(c[2]) for c in p.getContactPoints(bodyA=car.body,physicsClientId=cid)
                    if c[2] in scene.obstacle_ids and (c[8]<=0 or c[9]>0)}
            contacts=set(sat)|bullet
            if outside:contacts.add(-1)
            for obstacle in contacts-previous_contacts:
                collision_events.append(dict(time_s=t,obstacle_index=obstacle,bullet_contact=obstacle in bullet,
                    full_rectangle_overlap=obstacle in sat,outside_test_domain=outside,
                    phase=phase,pose=[state['x'],state['y'],state['yaw']]))
            previous_contacts=contacts
            rows.append([t,state['x'],state['y'],state['yaw'],state['speed'],target,motor,car.command_speed,
                         state['steering'],steer,state['direction'],index,err['cross_track'],err['heading_error'],
                         err['position_error'],err['arc'],state['roll'],state['pitch'],len(contacts),min(margins),int(paint)])
            phases.append(phase)
            return err,contacts
        # Initial settling is recorded too. Placement occurred only at creation.
        state=car.state();record(state,'SETTLING',0,0.,0.,machine.steering)
        for _ in range(480):
            car.command(0.,machine.steering,dt);p.stepSimulation(physicsClientId=cid);t+=dt
            state=car.state();record(state,'SETTLING',0,0.,0.,machine.steering)
        def capture(label,state):
            if not args.gui or label in snapshots:return
            cam=p.getDebugVisualizerCamera(physicsClientId=cid)
            image=p.getCameraImage(1100,800,viewMatrix=cam[2],projectionMatrix=cam[3],renderer=p.ER_BULLET_HARDWARE_OPENGL,physicsClientId=cid)[2]
            # Native debug text is not part of getCameraImage on some drivers.
            # Composite the exact current HUD content onto this actual frame.
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig=Figure(figsize=(11,8),dpi=100);canvas=FigureCanvasAgg(fig)
            ax=fig.add_axes([0,0,1,1]);ax.imshow(np.array(image,dtype=np.uint8).reshape(800,1100,4));ax.axis('off')
            if hud and hud.last_content:
                for text,x,y,size in [(hud.last_content[0],.02,.955,10),(hud.last_content[1],.02,.075,8),
                                      (hud.last_content[2],.02,.022,8),(hud.last_content[3],.66,.135,7)]:
                    fig.text(x,y,text,color='white',fontsize=size)
            canvas.draw();snapshots[label]=np.asarray(canvas.buffer_rgba()).copy()
        while not machine.finished:
            wall=time.monotonic()
            if args.gui:
                keys=p.getKeyboardEvents(physicsClientId=cid)
                key=lambda k:bool(keys.get(k,0)&p.KEY_WAS_TRIGGERED)
                if key(32):paused=not paused;interactive.append(dict(time_s=t,action='PAUSE' if paused else 'RESUME'))
                if key(ord('e')) or key(ord('E')) or key(p.B3G_F6):
                    paused=False;machine.emergency('OPERATOR_EMERGENCY_STOP',t,car.state())
                if key(ord('q')) or key(ord('Q')) or key(27):
                    paused=False;quit_after=True;machine.emergency('OPERATOR_EXIT',t,car.state())
            state=car.state()
            if paused:
                if hud:hud.update(f'PAUSED | {machine.phase} | {t:.2f}s | {machine.progress:.1f}%',
                    f'gear={state["direction"]:+d} | v={state["speed"]:+.3f} m/s | target=0 | steer={math.degrees(state["steering"]):+.1f}deg | yaw={math.degrees(state["yaw"]):+.1f}deg',
                    'SPACE resume | E/F6 emergency brake | Q/ESC exit','BLUE forward / ORANGE reverse / GREEN actual')
                time.sleep(.02);continue
            motor,steer,err,target=machine.update(state,dt,t)
            if machine.phase!=last_announced:
                print(f't={t:.3f} state={machine.phase} segment={machine.index+1} v={state["speed"]:+.4f}',flush=True)
                last_announced=machine.phase
            old_gear=state['direction']
            car.command(motor,steer,dt)
            # The motor/gear API runs its own untouched standstill interlock.
            engaged_state=car.state()
            if car.direction!=old_gear:
                gear_events.append(dict(time_s=t,from_gear=old_gear,to_gear=car.direction,
                    actual_speed_before_mps=state['speed'],actual_speed_after_command_mps=engaged_state['speed'],
                    requested_motor_speed_mps=motor,target_speed_mps=target,state=machine.phase))
            p.stepSimulation(physicsClientId=cid);t+=dt;state=car.state()
            err,contacts=record(state,machine.phase,machine.index,motor,target,steer)
            if contacts and machine.phase!='EMERGENCY_STOP':machine.emergency('EFFECTIVE_COLLISION',t,state)
            if max(abs(state['roll']),abs(state['pitch']))>math.radians(LIMITS['max_roll_pitch_deg']) and machine.phase!='EMERGENCY_STOP':
                machine.emergency('EXCESSIVE_TILT',t,state)
            if t>LIMITS['max_execution_s'] and machine.phase not in ('EMERGENCY_STOP','COMPLETE'):
                machine.emergency('TIMEOUT',t,state)
            if args.gui:
                if len(rows)%4==0:
                    point=[state['x'],state['y'],.055]
                    if last_point is not None:p.addUserDebugLine(last_point,point,[.05,.85,.3],3,physicsClientId=cid)
                    last_point=point
                    hud.update(f'{machine.phase} | seg {max(0,machine.index)+1}/{len(segments)} | gear {state["direction"]:+d} | {t:.2f}s | {machine.progress:.1f}% | collision={len(contacts)}',
                        f'v={state["speed"]:+.3f} target={target:+.3f}m/s | steer={math.degrees(state["steering"]):+.1f}deg | yaw={math.degrees(state["yaw"]):+.1f}deg | CTE={err["cross_track"]:+.3f}m',
                        'SPACE pause/resume | E/F6 emergency brake | Q/ESC controlled exit','BLUE forward / ORANGE reverse / GREEN actual')
                if machine.phase.startswith('SHIFT_TO_') and machine.index>=0 and machine.phase_time>.4:capture('SHIFT',state)
                if machine.index==1 and machine.trackers[1].progress>.5*machine.trackers[1].length:capture('FORWARD',state)
                if machine.index==2 and machine.trackers[2].progress>.65*machine.trackers[2].length:capture('REVERSE_PARKING',state)
                time.sleep(max(0,dt-(time.monotonic()-wall)))
            if int(t/10)!=last_print:
                last_print=int(t/10);print(f'Progress t={t:.1f}s {machine.progress:.1f}% CTE={err["cross_track"]:+.4f}',flush=True)
        final=car.state();trace=np.array(rows);phase_array=np.array(phases)
        polygon,margins,sat,outside,paint=body_and_bay(final,scenario)
        x0,y0,x1,y1=scenario['parking_slot']
        tire_bounds=[];tires_inside=True
        for joint in car.wheel_joints:
            lo,hi=p.getAABB(car.body,joint,physicsClientId=cid)
            tire_bounds.append(dict(link=joint,min=list(lo),max=list(hi)))
            tires_inside &= bool(lo[0]>=x0+LIMITS['bay_body_margin_m'] and hi[0]<=x1-LIMITS['bay_body_margin_m'] and
                                 lo[1]>=y0+LIMITS['bay_body_margin_m'] and hi[1]<=y1-LIMITS['bay_body_margin_m'])
        reference_end=data.samples[-1,:3];goal=scenario['goal']
        final_xy=np.array([final['x'],final['y']])
        reference_position_error=float(np.linalg.norm(final_xy-reference_end[:2]))
        reference_heading_error=abs(math.degrees(float(wrap(final['yaw']-reference_end[2]))))
        goal_position_error=float(np.linalg.norm(final_xy-[goal.x,goal.y]))
        goal_heading_error=abs(math.degrees(float(wrap(final['yaw']-goal.yaw))))
        segment_metrics=[]
        for i,segment in enumerate(segments):
            selected=(trace[:,11]==i)&np.isin(phase_array,['FORWARD','REVERSE','DECELERATING'])
            values=trace[selected]
            event=next((e for e in machine.segment_events if e['segment']==i),None)
            segment_metrics.append(dict(segment=i,direction=segment.direction,reference_indices=[segment.start,segment.stop],
                reference_length_m=machine.trackers[i].length,completed=event is not None,
                completion_percent=float(100*machine.trackers[i].progress/machine.trackers[i].length),
                rms_cte_m=float(np.sqrt(np.mean(values[:,12]**2))) if len(values) else None,
                max_abs_cte_m=float(np.max(np.abs(values[:,12]))) if len(values) else None,
                max_abs_heading_deg=float(np.max(np.abs(np.degrees(values[:,13])))) if len(values) else None,
                minimum_signed_speed_mps=float(np.min(values[:,4])) if len(values) else None,
                maximum_signed_speed_mps=float(np.max(values[:,4])) if len(values) else None,
                end_event=event))
        moving=np.isin(phase_array,['FORWARD','REVERSE','DECELERATING'])
        max_cte=float(np.max(np.abs(trace[moving,12]))) if np.any(moving) else None
        max_heading=float(np.max(np.abs(np.degrees(trace[moving,13])))) if np.any(moving) else None
        hold=trace[-round(LIMITS['final_hold_s']/dt):]
        hold_valid=bool(len(hold)==round(LIMITS['final_hold_s']/dt) and np.all(np.abs(hold[:,4])<=LIMITS['final_abs_speed_mps']) and
                        np.all(hold[:,19]>=LIMITS['bay_body_margin_m']))
        protected_ok=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protocol['protected_sha256'].items())
        checks=dict(all_segments_executed=len(machine.segment_events)==len(segments),
            all_segments_completed=all(s['completion_percent']>=LIMITS['min_segment_completion_percent'] for s in segment_metrics),
            both_shifts_completed=len(machine.shifts)==len(segments)-1,
            shifts_stopped=all(abs(e['actual_speed_before_mps'])<=LIMITS['gear_engagement_abs_speed_mps'] for e in gear_events),
            shifts_have_hold=all(e['duration_s']>=e['original_recorded_hold_s'] for e in machine.shifts),
            no_collision=not collision_events,body_inside_with_margin=min(margins)>=LIMITS['bay_body_margin_m'],
            all_four_tires_inside=tires_inside,final_pose_matches_reference=reference_position_error<=LIMITS['reference_final_position_m'],
            final_heading_matches_reference=reference_heading_error<=LIMITS['reference_final_heading_deg'],
            final_pose_matches_bay_goal=goal_position_error<=LIMITS['requested_goal_position_m'],
            final_heading_matches_bay=goal_heading_error<=LIMITS['requested_goal_heading_deg'],
            final_speed_zero_within_tolerance=abs(final['speed'])<=LIMITS['final_abs_speed_mps'],held_inside_bay_for_2s=hold_valid,
            tracking_cross_track=max_cte is not None and max_cte<=LIMITS['max_cross_track_m'],
            tracking_heading=max_heading is not None and max_heading<=LIMITS['max_heading_error_deg'],
            no_pose_reset_or_external_motion=not forbidden,baseline_preserved=protected_ok,
            state_machine_complete=machine.phase=='COMPLETE' and machine.failure is None)
        status='PASS' if all(checks.values()) else 'REVIEW'
        if hud:
            hud.update(f'{status} | {machine.phase} | gear {final["direction"]:+d} | {t:.2f}s | {machine.progress:.1f}%',
                f'v={final["speed"]:+.5f} target=0m/s | steer={math.degrees(final["steering"]):+.1f}deg | yaw={math.degrees(final["yaw"]):+.1f}deg | collisions={len(collision_events)}',
                'Experiment finished and held | Q/ESC exit','BLUE forward / ORANGE reverse / GREEN actual')
            capture('FINAL',final)
        summary=dict(stage='P11.4.2-P11.5 complete physical autonomous parking',status=status,checks=checks,limits=LIMITS,
            failure_reason=machine.failure,total_physics_time_s=t,physics_hz=240,control_hz=240,
            segment_results=segment_metrics,gear_engagement_events=gear_events,shift_events=machine.shifts,
            state_transitions=machine.events,interactive_events=interactive,collision_events=collision_events,
            reference_final_position_error_m=reference_position_error,reference_final_heading_error_deg=reference_heading_error,
            bay_goal_position_error_m=goal_position_error,bay_goal_heading_error_deg=goal_heading_error,
            final_state=final,body_corners_xy=polygon.tolist(),body_bay_margins_m=dict(zip(['left','right','entrance','back'],margins)),
            final_tire_collision_aabbs=tire_bounds,final_hold_valid=hold_valid,max_abs_cte_m=max_cte,max_abs_heading_error_deg=max_heading,
            max_abs_roll_deg=float(np.degrees(np.max(np.abs(trace[:,16])))),max_abs_pitch_deg=float(np.degrees(np.max(np.abs(trace[:,17])))),
            painted_line_crossing_samples=int(np.sum(trace[:,20])),
            collision_policy=protocol['bay_policy'],effective_obstacles=[list(r)for r in scenario['parked_vehicles']],
            background_collision_enabled=False,forbidden_vehicle_motion_api_calls=forbidden,
            reference_source=data.metadata,reference_samples=len(data.samples),reference_unchanged=True,
            vehicle_parameters=car.parameters(),reverse_sign_checks=diagnostic,protocol_sha256=hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
            protected_sha256=protocol['protected_sha256'],gui_run=args.gui)
        np.savez_compressed(out/f'{PREFIX}_trace.npz',trace=trace,columns=np.array(COLUMNS),state=phase_array,
            reference=data.samples,original_raw_trajectory=data.raw_trajectory,original_planned_path=data.path,
            original_segment_labels=data.segments,controller_segment_bounds=np.array([[s.start,s.stop]for s in segments]))
        (out/f'{PREFIX}_metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,2,figsize=(15,16),layout='constrained')
        for ax,label in zip(axes[:2].flat,['FORWARD','SHIFT','REVERSE_PARKING','FINAL']):
            if label in snapshots:ax.imshow(snapshots[label]);ax.axis('off');ax.set_title(label)
            else:ax.axis('off');ax.text(.5,.5,f'{label}: not reached / GUI not enabled',ha='center')
        for segment in segments:
            axes[2,0].plot(segment.reference[:,0],segment.reference[:,1],color='royalblue' if segment.direction>0 else 'darkorange',lw=2)
        axes[2,0].plot(trace[:,1],trace[:,2],color='green',lw=1,label='This run: physical rear axle')
        for a,b,c,d in scenario['parked_vehicles']:
            axes[2,0].add_patch(plt.Rectangle((a,b),c-a,d-b,facecolor='gray',alpha=.4))
        axes[2,0].plot(*np.vstack((polygon,polygon[0])).T,color='blue',lw=1)
        axes[2,0].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='gold',lw=2))
        axes[2,0].set(xlabel='X (m)',ylabel='Y (m)',title='Original reference and measured physical trajectory',aspect='equal');axes[2,0].legend(fontsize=8)
        axes[2,1].plot(trace[:,0],trace[:,4],label='Measured speed')
        axes[2,1].plot(trace[:,0],trace[:,5],'--',label='Target speed')
        for event in machine.shifts:
            axes[2,1].axvspan(event['start_time_s'],event['end_time_s'],alpha=.25,color='orange')
        axes[2,1].set(xlabel='Physics time (s)',ylabel='Signed speed (m/s)',title='Stop-before-shift (orange bands)');axes[2,1].legend(fontsize=8)
        for ax in axes[2]:ax.grid(alpha=.2)
        fig.suptitle(f'P11.5 full physical parking | {status} | original P10 reference, new wheel-driven execution')
        fig.savefig(out/f'{PREFIX}.png',dpi=140);plt.close(fig)
        segment_table='\n'.join(f'|{s["segment"]+1}|{s["direction"]:+d}|{s["completed"]}|{s["completion_percent"]:.3f}|{s["rms_cte_m"] if s["rms_cte_m"] is not None else "not reached"}|'for s in segment_metrics)
        report=f'''# P11.4–P11.5 完整物理自主泊车实验\n\n结果：{status}。控制状态：{machine.phase}；失败原因：{machine.failure}。\n\n## 输入与控制\n\n只使用原 P10.4.2 的 {len(data.samples)} 帧完整保存轨迹，按真实方向 -1/+1/-1 提取三段。原始记录、时间、速度、航向、方向和规划路径均保存在 NPZ；没有生成替代倒车路径。三个单独的倒车符号检查通过后，仅执行一次完整端到端实验。\n\n前进复用未修改的 ForwardTracker（Stanley gain=.65，softening=.45；PID 2.5/.35/.02）；倒车使用独立 ReverseTracker，通过已有 Stanley 的 direction=-1 选择真实后轴、倒车运动坐标系和转向反馈符号，不给车身 yaw 加 π。前进峰值轮廓 .28 m/s，倒车 .20 m/s。物理与反馈控制均 240 Hz，所有驱动/制动/转向只经 PhysicsVehicle.command。\n\n## 换挡与运动真实性\n\n记录初始接合及所有方向切换，实际完成 {len(machine.shifts)} 次方向切换。STOPPED 状态确认 |v|≤.015 m/s 后，SHIFT 状态在制动下预转向，连续静止等待不少于原记录停车时长；随后仍需通过原物理车内部 .35 s 联锁。停车制动保持末次转角，避免已发现的轮胎回正后挪。最终在车位内保持 2 秒。\n\n车辆只在创建时定位。运行期间对物理车辆的 resetBasePositionAndOrientation、resetBaseVelocity、resetJointState、applyExternalForce、applyExternalTorque 均有运行时拒绝保护；HUD 背景的视觉定位不属于车辆运动。禁用调用次数 {len(forbidden)}。\n\n## 预先声明的验收\n\n协议文件在运行前保存，SHA-256 为 {summary['protocol_sha256']}。终点相对原记录 ≤.10 m/3°，相对请求车位目标 ≤.25 m/5°；车身及四轮碰撞 AABB 完整入库，车身边界余量 ≥.03 m；|v|≤.01 m/s 并连续保持 2 秒；每段完成率 ≥98%；零有效碰撞；横向误差上限 .15 m，航向误差上限 6°。没有事后放宽阈值。\n\n|段|方向|执行完成|完成率 %|横向 RMS m|\n|---|---:|---|---:|---:|\n{segment_table}\n\n## 最终结果\n\n总物理时间 {t:.3f} s。相对原记录终点位置误差 {reference_position_error:.6f} m，航向误差 {reference_heading_error:.6f}°；相对目标车位中心 {goal_position_error:.6f} m，方向误差 {goal_heading_error:.6f}°。最终真实速度 {final['speed']:.8f} m/s。车身四边余量（左、右、入口、后侧）：{margins} m。有效碰撞事件 {len(collision_events)}。\n\n碰撞逐物理步检查所有车身/车轮链接的 Bullet 接触，并独立以真实 4.5×1.8 m 矩形做 SAT 复核；装饰隔离墩、护栏无碰撞体且未列入有效障碍物。车位线是地面油漆：进入过程可跨越，单独记录了 {int(np.sum(trace[:,20]))} 个侧线交叉采样；最终全车必须保持在边界内。\n\n所有检查结果、真实每步轨迹、换挡/状态转换时间、实际/目标速度、转角、误差及碰撞事件见 JSON/NPZ。原始 P10 与 P11.4.1 文件哈希校验：{protected_ok}。\n\n## 限制与运行\n\n此结果仅针对保存的单一停车场/轨迹，不能代表其他布局的鲁棒性。P11.3 刚性底盘、简化摩擦模型未改动；传感器定位、动态障碍物、悬架及真实轮胎标定不在本实验范围内。\n\n`conda run --no-capture-output -n parkingplanner python scripts/run_p11_5_full_physical_parking.py --gui --hold`\n\nSPACE 暂停/继续（暂停冻结物理时间），E/F6 紧急制动，Q/ESC 受控制动并退出；完成后保持最终姿态。\n'''
        (out/f'{PREFIX}_report.md').write_text(report,encoding='utf-8')
        print(json.dumps({k:summary[k] for k in ('status','checks','reference_final_position_error_m','reference_final_heading_error_deg',
            'bay_goal_position_error_m','bay_goal_heading_error_deg','body_bay_margins_m','total_physics_time_s','shift_events','failure_reason')},indent=2),flush=True)
        if args.gui and args.hold and not quit_after:
            while p.isConnected(cid):
                keys=p.getKeyboardEvents(physicsClientId=cid)
                if any(keys.get(k,0)&p.KEY_WAS_TRIGGERED for k in (27,ord('q'),ord('Q'))):break
                time.sleep(.05)
    finally:
        for name,original in originals.items():setattr(p,name,original)
        if p.isConnected(cid):p.disconnect(cid)


if __name__=='__main__':main()
