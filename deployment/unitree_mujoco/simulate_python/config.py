ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

# LiDAR settings
ENABLE_LIDAR = True
LIDAR_TYPE = "mid360"  # "mid360" or "airy"
# JAX is fastest for the full 24k-ray scan. The launcher disables JAX's default
# 75% VRAM preallocation unless the environment supplies an allocator setting.
# Use "cpu" to avoid GPU use entirely (it calls MuJoCo's native mj_multiRay).
LIDAR_BACKEND = "jax"  # "cpu" or "jax"
LIDAR_HZ = 20  # LiDAR update rate in Hz
LIDAR_ZMQ_PORT = 5590  # ZMQ PUB port for point cloud
