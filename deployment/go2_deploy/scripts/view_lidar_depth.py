#!/usr/bin/env python3
# This file is part of JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion
#
# Copyright (c) 2026 Qihao Yuan
#
# Developer: Qihao Yuan <qihao.yuan@rug.nl>
#
# For commercial use, please contact me at <qihao.yuan@rug.nl> or Kailai Li at <kailai.li@liu.se>.
#
# This file is subject to the terms and conditions outlined in the 'LICENSE' file,
# which is included as part of this source code package.

"""
Real-time LiDAR depth image viewer via ZMQ.

Subscribes to the lidar depth publisher on tcp://localhost:5560,
displays the 25×60 spherical depth image in a resizable pygame window
using nearest-neighbor upscaling while maintaining aspect ratio.

Usage:
  python3 view_lidar_depth.py
  python3 view_lidar_depth.py --host 192.168.1.10 --port 5560
"""

import argparse
import struct
import sys
import time

import numpy as np

try:
    import zmq
except ImportError:
    print("ERROR: pyzmq required. Install with: pip install pyzmq")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("ERROR: pygame required. Install with: pip install pygame")
    sys.exit(1)

# Max depth distance (must match C++ kMaxDistance)
kMaxDist = 2.0


# Orange (near, t=0) → Blue (far, t=1)
_ORANGE = np.array([255, 140, 0], dtype=np.float32)
_BLUE = np.array([30, 60, 255], dtype=np.float32)


def depth_color_scalar(t: float):
    """Return (R, G, B) ints in [0,255] for a single float t in [0,1]."""
    t = max(0.0, min(t, 1.0))
    r = int(_ORANGE[0] + t * (_BLUE[0] - _ORANGE[0]))
    g = int(_ORANGE[1] + t * (_BLUE[1] - _ORANGE[1]))
    b = int(_ORANGE[2] + t * (_BLUE[2] - _ORANGE[2]))
    return (r, g, b)


def depth_colormap(depth: np.ndarray) -> np.ndarray:
    """Orange→Blue colormap for a 2D [0,1] array. Returns (H,W,3) uint8."""
    t = np.clip(depth, 0, 1).astype(np.float32)
    r = _ORANGE[0] + t * (_BLUE[0] - _ORANGE[0])
    g = _ORANGE[1] + t * (_BLUE[1] - _ORANGE[1])
    b = _ORANGE[2] + t * (_BLUE[2] - _ORANGE[2])
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="ZMQ lidar depth image viewer")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5560)
    args = parser.parse_args()

    endpoint = f"tcp://{args.host}:{args.port}"
    print(f"Connecting to {endpoint} ...")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 50)  # 50ms timeout for responsive UI
    sub.setsockopt(zmq.CONFLATE, 1)  # only keep latest message
    sub.connect(endpoint)

    pygame.init()
    pygame.display.set_caption("LiDAR Depth (25×60)")
    screen = pygame.display.set_mode((720, 360), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    frame_count = 0
    t0 = time.time()
    last_depth = None
    running = True

    print("Waiting for lidar depth frames... (close window or press ESC to quit)")

    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # Receive depth frame
        try:
            msg = sub.recv()
            if len(msg) >= 8:
                w, h = struct.unpack("II", msg[:8])
                expected = 8 + w * h * 4
                if len(msg) == expected:
                    depth = np.frombuffer(msg[8:], dtype=np.float32).reshape(h, w)
                    last_depth = depth
                    frame_count += 1
        except zmq.Again:
            pass

        # Render
        screen.fill((0, 0, 0))
        win_w, win_h = screen.get_size()

        if last_depth is not None:
            # Apply orange→blue colormap: near (0) = orange, far (1) = blue
            rgb = depth_colormap(last_depth)
            img_h, img_w = last_depth.shape

            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

            # Layout: depth image + color bar + labels
            bar_w_frac = 0.06
            label_w_frac = 0.15
            total_w_ratio = 1.0 + bar_w_frac + label_w_frac

            # Scale to fit window, maintaining aspect ratio
            usable_w = win_w
            usable_h = win_h
            scale = min(usable_w / (img_w * total_w_ratio), usable_h / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            bar_w = max(1, int(img_w * bar_w_frac * scale))
            label_w = max(1, int(img_w * label_w_frac * scale))

            scaled = pygame.transform.scale(surface, (new_w, new_h))

            # Total composite width
            composite_w = new_w + bar_w + label_w
            x_off = (win_w - composite_w) // 2
            y_off = (win_h - new_h) // 2

            # Draw depth image
            screen.blit(scaled, (x_off, y_off))

            # Draw color bar (vertical gradient: top=far/1, bottom=near/0)
            for py in range(new_h):
                t = py / max(new_h - 1, 1)
                t_val = 1.0 - t  # top=1 (far), bottom=0 (near)
                r, g, b = depth_color_scalar(t_val)
                pygame.draw.line(
                    screen,
                    (r, g, b),
                    (x_off + new_w, y_off + py),
                    (x_off + new_w + bar_w - 1, y_off + py),
                )

            # Draw labels
            small_font = pygame.font.SysFont("monospace", max(12, new_h // 10))
            far_label = small_font.render(f"Far ({kMaxDist:.1f}m)", True, (200, 200, 200))
            near_label = small_font.render("Near (0m)", True, (200, 200, 200))
            lx = x_off + new_w + bar_w + 4
            screen.blit(far_label, (lx, y_off + 2))
            screen.blit(near_label, (lx, y_off + new_h - near_label.get_height() - 2))

            # Axis labels
            axis_font = pygame.font.SysFont("monospace", max(11, new_h // 12))
            # Horizontal: azimuth
            left_lbl = axis_font.render(f"{-60}\u00b0", True, (160, 160, 160))
            right_lbl = axis_font.render(f"+60\u00b0", True, (160, 160, 160))
            screen.blit(left_lbl, (x_off, y_off + new_h + 2))
            screen.blit(right_lbl, (x_off + new_w - right_lbl.get_width(), y_off + new_h + 2))
            # Vertical: elevation
            top_lbl = axis_font.render(f"+45.5\u00b0", True, (160, 160, 160))
            bot_lbl = axis_font.render(f"-4.5\u00b0", True, (160, 160, 160))
            screen.blit(top_lbl, (x_off - top_lbl.get_width() - 4, y_off))
            screen.blit(bot_lbl, (x_off - bot_lbl.get_width() - 4, y_off + new_h - bot_lbl.get_height()))

            # FPS + size text
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            info = f"{fps:.0f} fps  |  {img_w}x{img_h}"
            fps_surf = font.render(info, True, (200, 200, 200))
            screen.blit(fps_surf, (5, 5))
        else:
            wait_surf = font.render("Waiting for lidar depth...", True, (150, 150, 150))
            rect = wait_surf.get_rect(center=(win_w // 2, win_h // 2))
            screen.blit(wait_surf, rect)

        pygame.display.flip()
        clock.tick(60)

    print(f"\nReceived {frame_count} frames.")
    sub.close()
    ctx.term()
    pygame.quit()


if __name__ == "__main__":
    main()
