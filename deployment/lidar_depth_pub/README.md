# lidar_depth_pub — Livox Mid360 LiDAR Depth Image Publisher

Subscribes to `/livox/lidar` (Livox `CustomMsg` from `livox_ros_driver2`) and
publishes preprocessed spherical depth images over **ZMQ PUB**, matching
the simulation LiDAR preprocessing pipeline.

The required `--fov` argument selects one of three layouts:

- `--fov 25x60`: **25×60**, horizontal FOV **−60° to +60°**
- `--fov 25x90`: **25×90**, horizontal FOV **−90° to +90°**
- `--fov 25x120`: **25×120**, horizontal FOV **−120° to +120°**

In the 25×120 layout, the two image-bottom corner triangles are forced to far
distance (`1.0`) to suppress real Mid360 edge artifacts. This cleanup is not
applied in the 25×60 or 25×90 layouts.

## Pipeline

| Stage | Description |
|-------|-------------|
| 1. Receive | Raw 3D points from Livox Mid360 via ROS2 `CustomMsg` |
| 2. Filter | Reject near-zero, too-close (<0.1 m), and too-far (>2.0 m) points |
| 3. Project | Selected 25×60, 25×90, or 25×120 spherical grid — min distance per bin |
| 4. Normalize | Clamp [0, 2.0], divide by 2.0 → [0, 1] |
| 5. Buffer | Push into a configurable frame ring buffer (default: 5; init = 1.0) |
| 6. Aggregate | Element-wise min across the buffered frames |
| 7. Publish | Float32 depth image over ZMQ |

## Spherical Grid Parameters

| Parameter | Value |
|-----------|-------|
| Vertical FOV | −4.5° to +45.5° (50° total) |
| Horizontal FOV | −60° to +60°, −90° to +90°, or −120° to +120° |
| Vertical bins | 25 |
| Horizontal bins | 60, 90, or 120 (required `--fov`) |
| Max distance | 2.0 m |
| Min distance | 0.1 m |
| Normalization | ÷ 2.0 → [0, 1] |
| Accumulation | Configurable with `--stacked-frames` (default: 5), element-wise min |

## ZMQ Message Format

```
Offset  Size     Type       Description
──────  ───────  ─────────  ────────────────────────────────
0       4 bytes  uint32     width  = 60, 90, or 120 (columns)
4       4 bytes  uint32     height = 25 (rows)
8       variable float32[]  width × height depth values
──────  ───────  ─────────  ────────────────────────────────
Total: 6008, 9008, or 12008 bytes
```

Values in `[0.0, 1.0]`:
- **0.0** = at sensor (0 m)
- **1.0** = at or beyond max range (2.0 m)

Layout is **row-major** (row 0 = lowest elevation angle −4.5°, row 24 = highest
elevation angle 45.5°). Columns cover the symmetric horizontal range selected
by `--fov`.

Default port: **5560** (tcp).

## Dependencies

- ROS2 (Humble or later)
- [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)
- libzmq (v4.3+)
- CMake 3.16+

## Build

```bash
# From workspace root (where this folder lives)
colcon build --packages-select lidar_depth_pub
source install/setup.bash
```

Or standalone:

```bash
cd lidar_depth_pub
mkdir build && cd build
cmake .. && make
```

## Run

### ROS2 Livox source (default)

Make sure `livox_ros_driver2` is running and publishing on `/livox/lidar`.

```bash
ros2 run lidar_depth_pub lidar_depth_pub --fov 25x60
ros2 run lidar_depth_pub lidar_depth_pub --fov 25x90 --port 5561
ros2 run lidar_depth_pub lidar_depth_pub --fov 25x120 --stacked-frames 8
```

### Simulation source (unitree_mujoco)

Receives raw point clouds from `unitree_mujoco` via ZMQ instead of ROS2.
No ROS2 dependencies needed in this mode.

```bash
./build/lidar_depth_pub --sim --fov 25x60
./build/lidar_depth_pub --sim --fov 25x90 --sim-port 5590 --port 5560
./build/lidar_depth_pub --sim --fov 25x120 --stacked-frames 8
```

`--stacked-frames` accepts any positive integer. If omitted, the program uses
the existing default of 5 frames. The option is supported by both
`lidar_depth_pub` and `lidar_depth_pub_occ`.

### Optional curved-band occlusion

The `lidar_depth_pub` executable can add up to two fixed synthetic occlusion
bands. No curved band is added by default. Repeat `--occlusion-pattern` once per
band, choosing from `top`, `bottom`, `left`, `right`, `diagonal-down`, and
`diagonal-up`:

```bash
ros2 run lidar_depth_pub lidar_depth_pub --fov 25x90 \
  --occlusion-pattern top

./build/lidar_depth_pub --sim --fov 25x120 \
  --occlusion-pattern left --occlusion-pattern diagonal-up
```

Each pattern is randomized once at startup and then kept fixed. It is rendered
as a slightly curved quadratic Bezier band, kept away from the image border,
with width near 40% of the depth-image height (about 10 pixels for these
25-row layouts). Masked pixels use the same far-depth value (`1.0`) as the cage
occlusion. Use `--no-cage-mask` if the selected band(s) should be applied
without the normal cage pillars.

## Quick Python Receiver Test

```python
import zmq, struct, numpy as np

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5560")
sub.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    data = sub.recv()
    w, h = struct.unpack_from("<II", data, 0)
    depth = np.frombuffer(data, dtype=np.float32, offset=8).reshape(h, w)
    print(f"{w}x{h}  min={depth.min():.3f}  max={depth.max():.3f}  mean={depth.mean():.3f}")
```
