"""
LiDAR Point Cloud Visualizer

Receives point cloud data via ZMQ SUB and renders with PyOpenGL.
Controls:
  Left mouse drag   - rotate view
  Right mouse drag  - pan view
  Scroll wheel      - zoom in/out
  R                 - reset view
  Q / ESC           - quit
"""

import sys
import time
import numpy as np
import zmq

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import config


# --- Global state ---
points = np.zeros((0, 3), dtype=np.float32)
colors = np.zeros((0, 3), dtype=np.float32)

# Camera
cam_distance = 5.0
cam_yaw = 45.0
cam_pitch = 30.0
cam_target = [0.0, 0.0, 0.3]

# Mouse interaction
mouse_last = [0, 0]
mouse_button = None

# Stats
stats_num_points = 0
stats_freq_hz = 0.0
stats_frame_count = 0
stats_last_time = time.time()
stats_z_min = 0.0
stats_z_max = 0.0
stats_bytes_per_frame = 0
stats_total_frames = 0

# ZMQ
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(f"tcp://localhost:{config.LIDAR_ZMQ_PORT}")
sub.subscribe(b"")
sub.setsockopt(zmq.RCVTIMEO, 0)  # non-blocking recv
sub.setsockopt(zmq.CONFLATE, 1)  # keep only latest message

# Window size (tracked for 2D text overlay)
win_width = 1280
win_height = 720


def height_to_color(z_values):
    """Map height values to RGB colors using a simple HSV-like ramp."""
    if len(z_values) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z_min, z_max = z_values.min(), z_values.max()
    if z_max - z_min < 1e-6:
        norm = np.full_like(z_values, 0.5)
    else:
        norm = (z_values - z_min) / (z_max - z_min)

    # HSV hue ramp (0=red at bottom, 0.66=blue at top)
    hue = norm * 0.66
    rgb = np.zeros((len(z_values), 3), dtype=np.float32)
    for i, h in enumerate(hue):
        # Simplified HSV to RGB (S=1, V=1)
        c = 1.0
        x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
        hi = int(h * 6.0) % 6
        if hi == 0:
            rgb[i] = [c, x, 0]
        elif hi == 1:
            rgb[i] = [x, c, 0]
        elif hi == 2:
            rgb[i] = [0, c, x]
        elif hi == 3:
            rgb[i] = [0, x, c]
        elif hi == 4:
            rgb[i] = [x, 0, c]
        else:
            rgb[i] = [c, 0, x]
    return rgb


def init_gl():
    glClearColor(0.05, 0.05, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    glPointSize(2.0)


def draw_text(x, y, text):
    """Draw 2D text at pixel position (x, y) from top-left."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, win_width, 0, win_height, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    glRasterPos2f(x, win_height - y)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(ch))

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def draw_hud():
    """Draw heads-up display with stats."""
    glColor3f(0.0, 1.0, 0.0)
    lines = [
        f"Freq: {stats_freq_hz:.1f} Hz",
        f"Points: {stats_num_points}",
        f"Z range: [{stats_z_min:.2f}, {stats_z_max:.2f}] m",
        f"Frame size: {stats_bytes_per_frame / 1024:.1f} KB",
        f"Total frames: {stats_total_frames}",
    ]
    for i, line in enumerate(lines):
        draw_text(10, 20 + i * 18, line)


def draw_grid():
    """Draw a ground grid for reference."""
    glColor3f(0.25, 0.25, 0.25)
    glBegin(GL_LINES)
    grid_size = 10
    step = 1.0
    for i in range(-grid_size, grid_size + 1):
        glVertex3f(i * step, -grid_size * step, 0)
        glVertex3f(i * step, grid_size * step, 0)
        glVertex3f(-grid_size * step, i * step, 0)
        glVertex3f(grid_size * step, i * step, 0)
    glEnd()


def draw_axes():
    """Draw XYZ axes at origin."""
    glLineWidth(2.0)
    glBegin(GL_LINES)
    # X - red
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(1, 0, 0)
    # Y - green
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 1, 0)
    # Z - blue
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 1)
    glEnd()
    glLineWidth(1.0)


def display():
    global points, colors
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Camera transform
    eye_x = cam_target[0] + cam_distance * np.cos(np.radians(cam_pitch)) * np.cos(np.radians(cam_yaw))
    eye_y = cam_target[1] + cam_distance * np.cos(np.radians(cam_pitch)) * np.sin(np.radians(cam_yaw))
    eye_z = cam_target[2] + cam_distance * np.sin(np.radians(cam_pitch))
    gluLookAt(eye_x, eye_y, eye_z,
              cam_target[0], cam_target[1], cam_target[2],
              0, 0, 1)

    draw_grid()
    draw_axes()

    # Draw points
    if len(points) > 0:
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, points)
        glColorPointer(3, GL_FLOAT, 0, colors)
        glDrawArrays(GL_POINTS, 0, len(points))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    draw_hud()

    glutSwapBuffers()


def reshape(w, h):
    global win_width, win_height
    if h == 0:
        h = 1
    win_width = w
    win_height = h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, w / h, 0.1, 200.0)


def keyboard(key, x, y):
    if key in (b'q', b'\x1b'):
        sys.exit(0)
    elif key == b'r':
        global cam_distance, cam_yaw, cam_pitch, cam_target
        cam_distance = 5.0
        cam_yaw = 45.0
        cam_pitch = 30.0
        cam_target = [0.0, 0.0, 0.3]


def mouse(button, state, x, y):
    global mouse_button, mouse_last
    if state == GLUT_DOWN:
        mouse_button = button
        mouse_last = [x, y]
    else:
        mouse_button = None


def motion(x, y):
    global cam_yaw, cam_pitch, cam_target, mouse_last
    dx = x - mouse_last[0]
    dy = y - mouse_last[1]
    mouse_last = [x, y]

    if mouse_button == GLUT_LEFT_BUTTON:
        cam_yaw -= dx * 0.3
        cam_pitch += dy * 0.3
        cam_pitch = max(-89, min(89, cam_pitch))
    elif mouse_button == GLUT_RIGHT_BUTTON:
        # Pan in camera-local XY
        scale = cam_distance * 0.002
        right = np.array([-np.sin(np.radians(cam_yaw)), np.cos(np.radians(cam_yaw)), 0])
        up = np.array([0, 0, 1])
        cam_target[0] -= (dx * right[0] + dy * up[0]) * scale
        cam_target[1] -= (dx * right[1] + dy * up[1]) * scale
        cam_target[2] -= (dx * right[2] + dy * up[2]) * scale


def mouse_wheel(button, direction, x, y):
    global cam_distance
    if direction > 0:
        cam_distance *= 0.9
    else:
        cam_distance *= 1.1
    cam_distance = max(0.5, min(50.0, cam_distance))


def idle():
    global points, colors
    global stats_num_points, stats_freq_hz, stats_frame_count, stats_last_time
    global stats_z_min, stats_z_max, stats_bytes_per_frame, stats_total_frames
    try:
        data = sub.recv()
        buf = np.frombuffer(data, dtype=np.float32)
        # Message: [3 floats pos][9 floats rot][N*3 floats local points]
        sensor_pos = buf[:3]
        sensor_rot = buf[3:12].reshape(3, 3)
        local_pts = buf[12:].reshape(-1, 3)
        pts = local_pts @ sensor_rot.T + sensor_pos
        points = pts
        colors = height_to_color(pts[:, 2])

        # Update stats
        stats_num_points = len(pts)
        stats_bytes_per_frame = len(data)
        stats_total_frames += 1
        if len(pts) > 0:
            stats_z_min = pts[:, 2].min()
            stats_z_max = pts[:, 2].max()

        stats_frame_count += 1
        now = time.time()
        elapsed = now - stats_last_time
        if elapsed >= 1.0:
            stats_freq_hz = stats_frame_count / elapsed
            stats_frame_count = 0
            stats_last_time = now
    except zmq.Again:
        pass
    glutPostRedisplay()


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1280, 720)
    glutCreateWindow(b"LiDAR Point Cloud Visualizer")
    init_gl()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutMouseWheelFunc(mouse_wheel)
    glutIdleFunc(idle)
    print(f"Connecting to ZMQ on tcp://localhost:{config.LIDAR_ZMQ_PORT}")
    print("Waiting for point cloud data...")
    glutMainLoop()


if __name__ == "__main__":
    main()
