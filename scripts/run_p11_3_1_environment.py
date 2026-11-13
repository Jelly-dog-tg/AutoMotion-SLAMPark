"""P11.3.1 scenery validation and snapshot; original experiments are read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pybullet as p
from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.simulation.pybullet.scene import ParkingScene
from parking_planner.simulation.pybullet.physics_vehicle import PhysicsVehicle
from parking_planner.simulation.pybullet.trajectory import load_execution
from parking_planner.simulation.pybullet.hud import EdgeStatusBars
from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS


def main():
    protected=list((ROOT/'results').glob('p10*'))+list((ROOT/'results').glob('p11_3_basic*'))
    protected+=[ROOT/'parking_planner/simulation/pybullet/physics_vehicle.py']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected if f.is_file()}
    out=ROOT/'results'
    with np.load(out/'p11_3_basic_motion_trace.npz') as old,np.load(out/'p11_3_1_basic_motion_trace.npz') as new:
        equal=bool(np.array_equal(old['trace'],new['trace']))
        maximum=float(np.max(np.abs(old['trace']-new['trace'])))
    basic=json.loads((out/'p11_3_1_basic_motion_metrics.json').read_text(encoding='utf-8'))
    cid=p.connect(p.GUI,options=WINDOW_OPTIONS)
    try:
        scenario=build_perpendicular_parking_scenario();scene=ParkingScene(cid,scenario)
        reference=load_execution(out/'p10_4_2_execution_trajectory.npz')
        car=PhysicsVehicle(cid,scenario['geometry'],reference.samples[0,:3])
        p.setTimeStep(1/240,physicsClientId=cid)
        for _ in range(480): car.command(0.,0.,1/240);p.stepSimulation(physicsClientId=cid)
        scene.camera_target=[11.8,11.4,0.];scene.configure_gui()
        hud=EdgeStatusBars(cid);state=car.state()
        hud.update('P11.3.1 | ENVIRONMENT REVIEW | 100%',
                   f'v={state["speed"]:+.3f} m/s | steer={np.degrees(state["steering"]):+.1f} deg | yaw={np.degrees(state["yaw"]):+.1f} deg',
                   'Visual-only perimeter scenery | original physical floor and obstacles unchanged',
                   'BLUE experimental vehicle / peripheral background')
        visual_only=all(not p.getCollisionShapeData(body,-1,physicsClientId=cid) for body in scene.background_ids)
        physical_count=len(scene.obstacle_ids)
        for body,rect in zip(scene.obstacle_ids,scenario['parked_vehicles']):
            lo,hi=p.getAABB(body,physicsClientId=cid)
            assert np.allclose([lo[0],lo[1],hi[0],hi[1]],rect)
        camera=p.getDebugVisualizerCamera(physicsClientId=cid)
        view=np.array(camera[2]).reshape(4,4,order='F');proj=np.array(camera[3]).reshape(4,4,order='F')
        inverse=np.linalg.inv(proj @ view);covered=True;ground_points=[]
        for x,y in [(-1,-1),(-1,1),(1,-1),(1,1),(0,0)]:
            near=inverse@np.array([x,y,-1,1]);near=near[:3]/near[3]
            far=inverse@np.array([x,y,1,1]);far=far[:3]/far[3]
            d=far-near;u=(-.005-near[2])/d[2]
            point=near+u*d
            covered &= bool(u>=0 and np.all(np.abs(point[:2]-11)<1000))
            ground_points.append(point.tolist())
        image=p.getCameraImage(1100,800,viewMatrix=camera[2],projectionMatrix=camera[3],
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,physicsClientId=cid)[2]
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(11,8),dpi=120);canvas=FigureCanvasAgg(fig)
        ax=fig.add_axes([0,0,1,1]);ax.imshow(np.array(image,dtype=np.uint8).reshape(800,1100,4));ax.axis('off')
        for text,x,y,size in [(hud.last_content[0],.02,.955,11),(hud.last_content[1],.02,.075,9),
                              (hud.last_content[2],.02,.022,8),(hud.last_content[3],.66,.135,7)]:
            fig.text(x,y,text,color='white',fontsize=size)
        fig.savefig(out/'p11_3_1_environment.png')
        unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
        metrics=dict(status='PASS' if equal and basic['status']=='PASS' and visual_only and covered and unchanged else 'REVIEW',
            original_physics_trace_bit_identical=equal,max_basic_trace_difference=maximum,basic_test_status=basic['status'],
            background_visual_only=visual_only,background_body_count=len(scene.background_ids),physical_obstacle_count=physical_count,
            apron_size_m=[2000,2000],apron_top_z_m=-.005,original_physical_ground_top_z_m=0.,
            frustum_covered_by_asphalt=covered,frustum_ground_points=ground_points,protected_unchanged=unchanged,protected_sha256=hashes,
            note='Original 22x22 physical floor retained; exterior apron and all scenery are visual-only. No usable driving area or planning obstacle set is extended.')
        (out/'p11_3_1_environment_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        report=f'''# P11.3.1 停车场环境扩展\n\n结果：{metrics['status']}。\n\n外围新增 2000×2000 m 同色深灰沥青展示层，顶面 Z=-0.005 m；原实验区 22×22 m 地面的高度 Z=0、碰撞体、摩擦和接触关系完全保留。外部视觉地面不表示新增可行驶物理区域。摄像机保持 35°/-62° 俯视斜角，目标 (11.8,11.4)，自动适配原区域边界。视锥四角及中心与展示地面的交点均在覆盖范围内。\n\n实验区外放置 5 辆停放车辆、3 个隔离墩及稀疏护栏，附带独立车位线。背景模型全部无碰撞体，不加入原有两辆障碍物集合；原车位、标线、箭头、物理车参数、车辆起点和 HUD 未改变。\n\n重新运行基础物理测试：{basic['status']}。新旧真实状态轨迹逐元素完全一致：{equal}；最大差值 {maximum}。障碍物碰撞仍为 {basic['collision_episodes']}。原 P10 和 P11.3 成果 SHA-256 检查通过：{unchanged}。\n\n新基础实验结果使用 p11_3_1_basic_motion 前缀，未覆盖旧结果。\n\n复现：`python scripts/run_p11_3_physics_basic.py --output-prefix p11_3_1_basic_motion`，然后 `python scripts/run_p11_3_1_environment.py`。\n'''
        (out/'p11_3_1_environment_report.md').write_text(report,encoding='utf-8')
        print(json.dumps({k:v for k,v in metrics.items() if k not in ('protected_sha256','frustum_ground_points')},indent=2))
    finally:
        p.disconnect(cid)


if __name__=='__main__': main()
