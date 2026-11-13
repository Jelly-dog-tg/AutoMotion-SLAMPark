"""P11.1 only: static PyBullet lot and pose-controlled visual vehicle.

Run: conda run -n parkingplanner python scripts/run_p11_1_pybullet_scene.py
Keys: Left/Right steering, Home reset; mouse uses standard PyBullet orbit/zoom.
No planner, controller, sensors, or vehicle dynamics run in this entry point.
"""
import argparse
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pybullet as p
from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.vehicle.ackermann import VehicleState
from parking_planner.simulation.pybullet.scene import ParkingScene
from parking_planner.simulation.pybullet.vehicle import VehicleVisual
from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS, preview_pose


def check_scene(scene, vehicle):
    """Check coordinate geometry against the existing scenario, without P1-P10 runs."""
    cid, scenario = scene.client_id, scene.scenario
    geometry, start = scenario['geometry'], scenario['start']
    for body, rect in zip(scene.obstacle_ids,scenario['parked_vehicles'],strict=True):
        lower,upper = p.getAABB(body,physicsClientId=cid)
        assert np.allclose([lower[0],lower[1],upper[0],upper[1]],rect,atol=1e-9)
    assert all(not p.getCollisionShapeData(body,-1,physicsClientId=cid) for body in scene.marking_ids)
    for x,y,yaw,steer in [(start.x,start.y,start.yaw,0.),(10.,8.,math.pi/2,.35),(8.,9.,-.7,-.4)]:
        vehicle.set_pose(x,y,yaw,steer)
        c,s = math.cos(yaw),math.sin(yaw)
        for axle,lx in [('rear',0.),('front',geometry.wheelbase)]:
            for side,ly in [('left',vehicle.track_width/2),('right',-vehicle.track_width/2)]:
                position,q = p.getBasePositionAndOrientation(vehicle.wheels[axle+'_'+side],physicsClientId=cid)
                assert np.allclose(position,[x+c*lx-s*ly,y+s*lx+c*ly,vehicle.wheel_radius],atol=1e-9)
                rotation = np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)
                wheel_forward = rotation[:,0]
                angle = yaw+(steer if axle=='front' else 0)
                assert np.allclose(wheel_forward,[math.cos(angle),math.sin(angle),0],atol=1e-9)
        center,q = p.getBasePositionAndOrientation(vehicle.body_id,physicsClientId=cid)
        local_center = geometry.length/2-geometry.rear_overhang
        assert np.allclose(center[:2],[x+c*local_center,y+s*local_center],atol=1e-9)
        assert np.allclose(p.getVisualShapeData(vehicle.body_id,physicsClientId=cid)[0][3],
                           [geometry.length,geometry.width,.6])
    # Failure is atomic: a rejected update must not move any visual part.
    before = vehicle.pose
    try:
        vehicle.set_pose(float('nan'),0,0,0)
        raise AssertionError('Expected ValueError')
    except ValueError:
        assert vehicle.pose == before
    vehicle.set_pose(start.x,start.y,start.yaw,0.)
    print('PASS: exact obstacle footprints, visual-only paint, 4 wheel offsets, yaw/steering, body dimensions, invalid-input rejection, initial-pose restoration.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true',help='Use DIRECT for automated geometry checks')
    parser.add_argument('--check',action='store_true',help='Verify geometry and set_pose; restore initial pose')
    parser.add_argument('--duration',type=float,default=0.,help='Auto-close after N seconds; 0 keeps GUI open')
    parser.add_argument('--steering-deg',type=float,default=0.,help='Initial front-wheel display angle')
    parser.add_argument('--screenshot',type=Path,help='Save rendered scene PNG using the initial GUI camera')
    args = parser.parse_args()
    if args.duration < 0 or not math.isfinite(args.duration) or not math.isfinite(args.steering_deg):
        parser.error('Duration must be finite and nonnegative; steering must be finite')
    cid = p.connect(p.DIRECT if args.headless else p.GUI,options='' if args.headless else WINDOW_OPTIONS)
    if cid < 0:
        raise RuntimeError('Unable to connect to PyBullet')
    try:
        scenario = build_perpendicular_parking_scenario()
        scene = ParkingScene(cid,scenario)
        vehicle = VehicleVisual(cid,scenario['geometry'])
        if args.check:
            check_scene(scene,vehicle)
        start = scenario['start']
        preview = preview_pose(scenario)
        from scripts.run_p9_5_3_keyboard_exploration import collision
        if collision(VehicleState(*preview),scenario):
            raise RuntimeError("Preview footprint is in collision")
        steering = math.radians(args.steering_deg)
        vehicle.set_pose(*preview,steering)
        if not args.headless:
            scene.configure_gui()
            p.addUserDebugText('P11.1 | LEFT/RIGHT: steer | HOME: reset',
                              [2,4,.1],textColorRGB=[1,1,1],textSize=.8,physicsClientId=cid)
            p.addUserDebugText('TARGET BAY',[9.3,16.6,.03],textColorRGB=[1,.8,.2],textSize=1.1,physicsClientId=cid)
        print(f'P11.1 ready: {p.getConnectionInfo(cid)}; preview rear axle=({preview[0]}, {preview[1]}), yaw={preview[2]}; steering={args.steering_deg} deg',flush=True)
        if args.screenshot:
            scene.save_screenshot(args.screenshot,gui=not args.headless)
            print(f'Screenshot: {args.screenshot.resolve()}',flush=True)
        if args.headless:
            return
        begin = time.monotonic()
        while p.isConnected(cid):
            if args.duration and time.monotonic()-begin >= args.duration:
                break
            keys = p.getKeyboardEvents(physicsClientId=cid)
            for key,increment in [(p.B3G_LEFT_ARROW,5.),(p.B3G_RIGHT_ARROW,-5.)]:
                if keys.get(key,0) & p.KEY_WAS_TRIGGERED:
                    steering = max(-scenario['planner_config'].max_steer,
                                   min(scenario['planner_config'].max_steer,steering+math.radians(increment)))
                    vehicle.set_pose(*vehicle.pose[:3],steering)
                    print(f'Steering: {math.degrees(steering):.1f} deg',flush=True)
            if keys.get(p.B3G_HOME,0) & p.KEY_WAS_TRIGGERED:
                steering = 0.
                vehicle.set_pose(*preview,steering)
            time.sleep(1/60)
    except p.error:
        if p.isConnected(cid):
            raise
    finally:
        if p.isConnected(cid):
            p.disconnect(cid)


if __name__ == '__main__':
    main()
