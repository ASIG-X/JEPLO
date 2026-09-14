import argparse
import os
import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from terrain_switcher import TerrainSwitcher

import config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lidar-backend", type=str, default=config.LIDAR_BACKEND,
    choices=["jax", "cpu"], help=f"LiDAR backend (default: {config.LIDAR_BACKEND})",
)
args = parser.parse_args()

if config.ENABLE_LIDAR:
    import sys
    import numpy as np
    import zmq

    if args.lidar_backend == "jax":
        # JAX otherwise reserves 75% of VRAM on its first operation. Respect any
        # allocator setting supplied by the caller; when none is supplied, grow
        # the pool on demand instead of monopolizing the GPU used by the viewer.
        jax_allocator_vars = (
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
        )
        if not any(name in os.environ for name in jax_allocator_vars):
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # The selected backend is imported only after configuring JAX's allocator.
    sys.path.insert(0, "MuJoCo-LiDAR/src")
    from mujoco_lidar import MjLidarWrapper, scan_gen


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)
terrain_switcher = TerrainSwitcher(mj_model, mj_data)


def viewer_key_callback(key):
    terrain_switcher.key_callback(key)
    if config.ENABLE_ELASTIC_BAND:
        elastic_band.MujuocoKeyCallback(key)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=viewer_key_callback
    )
else:
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=viewer_key_callback
    )

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

# LiDAR setup
if config.ENABLE_LIDAR:
    lidar_context = zmq.Context()
    lidar_publisher = lidar_context.socket(zmq.PUB)
    lidar_publisher.bind(f"tcp://*:{config.LIDAR_ZMQ_PORT}")

    if config.LIDAR_TYPE == "mid360":
        livox_generator = scan_gen.LivoxGenerator(config.LIDAR_TYPE)
        rays_theta, rays_phi = livox_generator.sample_ray_angles()
        lidar_dynamic = True
    elif config.LIDAR_TYPE == "airy":
        rays_theta, rays_phi = scan_gen.generate_airy96()
        lidar_dynamic = False

    rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
    rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)

    geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
    geomgroup[3:] = 0  # exclude robot visual geoms
    lidar_wrapper = MjLidarWrapper(
        mj_model,
        site_name="lidar",
        backend=args.lidar_backend,
        args={"bodyexclude": mj_model.body("base_link").id, "geomgroup": geomgroup},
    )

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    lidar_dt = 1.0 / config.LIDAR_HZ if config.ENABLE_LIDAR else None
    lidar_counter = 0
    lidar_substeps = int(round(lidar_dt / config.SIMULATE_DT)) if config.ENABLE_LIDAR else 0

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        terrain_switcher.apply_pending()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        # LiDAR ray tracing and ZMQ publish
        if config.ENABLE_LIDAR:
            lidar_counter += 1
            if lidar_counter % lidar_substeps == 0:
                global rays_theta, rays_phi
                if lidar_dynamic:
                    rays_theta, rays_phi = livox_generator.sample_ray_angles()
                    rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
                    rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)
                lidar_wrapper.trace_rays(mj_data, rays_theta, rays_phi)
                local_points = lidar_wrapper.get_hit_points()
                sensor_pos = lidar_wrapper.sensor_position.astype(np.float32)
                sensor_rot = lidar_wrapper.sensor_rotation.astype(np.float32)
                # Message: [3 floats pos][9 floats rot][N*3 floats local points]
                msg = sensor_pos.tobytes() + sensor_rot.tobytes() + local_points.astype(np.float32).tobytes()
                lidar_publisher.send(msg)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
