
"""P11.6.1: Manual physical driving with live PyBullet LiDAR.

Controls:
W/S: speed
A/D: steering
C: center steering
Space: brake
P: pause/resume
M: save
Q/Esc: save and exit
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)

from parking_planner.simulation.pybullet.scene import ParkingScene

from parking_planner.simulation.pybullet.physics_vehicle import (
    PhysicsVehicle,
)

from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS

from parking_planner.simulation.pybullet.hud import EdgeStatusBars

from parking_planner.sensing.lidar_2d import LidarConfig

from parking_planner.sensing.pybullet_lidar import PyBulletLidar2D


DT = 1.0 / 240.0

SCAN_STEPS = 96

SPEED_INCREMENT = 0.10
STEER_INCREMENT = math.radians(5.0)

MAX_MANUAL_SPEED = 0.45

OUT = ROOT / "results"


def triggered(events, character):
    return any(
        events.get(ord(ch), 0) & p.KEY_WAS_TRIGGERED
        for ch in {character.lower(), character.upper()}
    )


def draw_scan(cid, scan, old_items, previous, current):
    """Update a subset of LiDAR rays for 3D visualization."""

    for item in old_items:
        p.removeUserDebugItem(
            item,
            physicsClientId=cid,
        )

    new_items = []

    indices = np.linspace(
        0,
        len(scan["angles_rad"]) - 1,
        36,
        dtype=int,
    )

    origin = scan["origin_xyz"].tolist()

    for index in indices:
        if not scan["valid"][index]:
            continue

        distance = float(scan["ranges_m"][index])

        direction = scan["angles_rad"][index] + current[2]

        end = [
            current[0] + distance * math.cos(direction),
            current[1] + distance * math.sin(direction),
            origin[2],
        ]

        new_items.append(
            p.addUserDebugLine(
                origin,
                end,
                lineColorRGB=[0.3, 0.82, 0.9],
                lineWidth=1.0,
                physicsClientId=cid,
            )
        )

        new_items.append(
            p.addUserDebugLine(
                [end[0], end[1], end[2] - 0.07],
                [end[0], end[1], end[2] + 0.07],
                lineColorRGB=[1.0, 0.55, 0.1],
                lineWidth=2.5,
                physicsClientId=cid,
            )
        )

    if previous is not None:
        p.addUserDebugLine(
            [previous[0], previous[1], 0.05],
            [current[0], current[1], 0.05],
            lineColorRGB=[0.1, 0.9, 0.35],
            lineWidth=3,
            physicsClientId=cid,
        )

    return new_items


def main():

    scenario = build_perpendicular_parking_scenario()

    cid = p.connect(
        p.GUI,
        options=WINDOW_OPTIONS,
    )

    if cid < 0:
        raise RuntimeError("PyBullet GUI failed to initialize")

    records = []
    scans = []
    poses_at_scan = []
    scan_timestamps = []

    scan_visuals = []
    previous_scan_pose = None

    t_sim = 0.0

    paused = True

    target_speed = 0.0
    target_steer = 0.0

    collision_count = 0
    self_hit_total = 0

    last_scan = None

    def save():
        OUT.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            OUT / "p11_6_1_physical_lidar_scans.npz",

            angles_rad=lidar.angles,

            ranges_m=np.stack([
                s["ranges_m"]
                for s in scans
            ]),

            valid=np.stack([
                s["valid"]
                for s in scans
            ]),

            scan_timestamps_s=np.asarray(
                scan_timestamps,
                dtype=float,
            ),

            true_rear_axle_poses=np.asarray(
                poses_at_scan,
                dtype=float,
            ),

            physical_trace=np.asarray(
                records,
                dtype=float,
            ).reshape(-1, 6),

            trace_columns=np.asarray([
                "time_s",
                "x",
                "y",
                "yaw",
                "speed",
                "steering",
            ]),
        )

        summary = {
            "stage": "P11.6.1 real PyBullet LiDAR + manual physical exploration",

            "status": (
                "SCAN_CAPTURED"
                if scans
                else "NO_SCAN"
            ),

            "source": "live PyBullet rayTestBatch; no P10 files read",

            "slam_enabled": False,

            "num_scans": len(scans),

            "beams_per_scan": lidar.config.num_beams,

            "total_hits": int(
                sum(
                    int(s["valid"].sum())
                    for s in scans
                )
            ),

            "last_hit_count": (
                int(last_scan["valid"].sum())
                if last_scan
                else 0
            ),

            "self_hit_total": self_hit_total,

            "collision_samples": collision_count,

            "simulation_time_s": t_sim,

            "last_true_pose": (
                poses_at_scan[-1]
                if poses_at_scan
                else None
            ),

            "sensor_height_m": lidar.height_m,

            "sensor_reference": (
                "rear axle center in XY; "
                "horizontal plane at fixed Z"
            ),

            "background_models_collidable": False,
        }

        metrics_path = (
            OUT / "p11_6_1_physical_lidar_metrics.json"
        )

        metrics_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )

    try:

        p.setTimeStep(
            DT,
            physicsClientId=cid,
        )

        p.setPhysicsEngineParameter(
            numSolverIterations=100,
            deterministicOverlappingPairs=1,
            physicsClientId=cid,
        )

        scene = ParkingScene(
            cid,
            scenario,
        )

        start = scenario["start"]

        car = PhysicsVehicle(
            cid,
            scenario["geometry"],
            (start.x, start.y, start.yaw),
        )

        lidar = PyBulletLidar2D(
            cid,
            car.body,

            config=LidarConfig(
                num_beams=180,
                fov_deg=270.0,
                min_range_m=0.12,
                max_range_m=12.0,
                range_std_m=0.015,
            ),

            height_m=1.25,
        )

        # Allow vehicle to settle under gravity.
        for _ in range(240):
            car.command(
                0.0,
                0.0,
                DT,
            )

            p.stepSimulation(
                physicsClientId=cid,
            )

        scene.camera_target = [10.6, 11.3, 0.0]
        scene.camera_distance = 21.0

        scene.configure_gui()

        hud = EdgeStatusBars(cid)

        def capture():
            nonlocal last_scan
            nonlocal self_hit_total
            nonlocal scan_visuals
            nonlocal previous_scan_pose

            state = car.state()

            scan = lidar.scan(state)

            pose = [
                state["x"],
                state["y"],
                state["yaw"],
            ]

            scans.append(scan)
            poses_at_scan.append(pose)
            scan_timestamps.append(t_sim)

            self_hit_total += scan["self_hits"]

            scan_visuals = draw_scan(
                cid,
                scan,
                scan_visuals,
                previous_scan_pose,
                pose,
            )

            previous_scan_pose = pose

            last_scan = scan

        # Initial fresh scan.
        capture()

        step = 0
        ui_frame = 0

        while p.isConnected(cid):

            ui_frame += 1
            frame_start = time.monotonic()

            keys = p.getKeyboardEvents(
                physicsClientId=cid,
            )

            if (
                triggered(keys, "q")
                or (keys.get(27, 0) & p.KEY_WAS_TRIGGERED)
            ):
                break

            if triggered(keys, "p"):
                paused = not paused

            if triggered(keys, "w"):
                target_speed = min(
                    MAX_MANUAL_SPEED,
                    target_speed + SPEED_INCREMENT,
                )

            if triggered(keys, "s"):
                target_speed = max(
                    -MAX_MANUAL_SPEED,
                    target_speed - SPEED_INCREMENT,
                )

            if triggered(keys, "a"):
                target_steer = min(
                    car.config.max_steering_rad,
                    target_steer + STEER_INCREMENT,
                )

            if triggered(keys, "d"):
                target_steer = max(
                    -car.config.max_steering_rad,
                    target_steer - STEER_INCREMENT,
                )

            if triggered(keys, "c"):
                target_steer = 0.0

            if keys.get(32, 0) & p.KEY_WAS_TRIGGERED:
                target_speed = 0.0

            if triggered(keys, "m"):
                save()

            if not paused:

                car.command(
                    target_speed,
                    target_steer,
                    DT,
                )

                p.stepSimulation(
                    physicsClientId=cid,
                )

                t_sim += DT
                step += 1

                state = car.state()

                records.append([
                    t_sim,
                    state["x"],
                    state["y"],
                    state["yaw"],
                    state["speed"],
                    state["steering"],
                ])

                contacts = p.getContactPoints(
                    bodyA=car.body,
                    physicsClientId=cid,
                )

                if any(
                    c[2] in scene.obstacle_ids
                    for c in contacts
                ):
                    collision_count += 1
                    target_speed = 0.0

                if step % SCAN_STEPS == 0:
                    capture()

            if ui_frame % 12 == 0:

                state = car.state()

                hud.update(
                    f"P11.6.1 | {'PAUSED' if paused else 'MANUAL'} "
                    f"| {t_sim:.1f}s "
                    f"| LiDAR scans: {len(scans)} "
                    f"| hits: {int(last_scan['valid'].sum())}",

                    f"v={state['speed']:+.2f} m/s "
                    f"| target={target_speed:+.2f} m/s "
                    f"| steer={math.degrees(state['steering']):+.1f}deg "
                    f"| yaw={math.degrees(state['yaw']):+.1f}deg",

                    "P run/pause | W/S speed | A/D steer "
                    "| C center | SPACE brake | M save | Q exit",

                    "CYAN LiDAR rays / ORANGE hits / GREEN driven trace",
                )

            time.sleep(
                max(
                    0.0,
                    DT - (time.monotonic() - frame_start),
                )
            )

        save()

    finally:

        if p.isConnected(cid):
            p.disconnect(cid)


if __name__ == "__main__":
    main()
