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
ZMQ pygame controller for Go2 robot (locomotion_node_wm).

Publishes velocity commands as 3x float32 (vx, vy, wz) via ZMQ PUB on port 5562.

Uses pygame for true simultaneous key detection (multiple keys at once).

Controls:
  W / Up   : Forward
  S / Down : Backward
  A        : Strafe left
  D        : Strafe right
  Left / J : Turn left
  Right / K: Turn right
  1        : Slow mode
  2        : Fast mode
  SPACE    : Emergency stop (zero all)
  Q / ESC  : Quit
"""

import argparse
import struct
import time

import pygame
import zmq

# Speed presets
SPEED_MODES = {
    "slow": {"forward": 0.75, "lateral": 0.3, "angular": 1.0},
    "fast": {"forward": 1.0, "lateral": 1.0, "angular": 2.0},
}

PUBLISH_HZ = 50
DEFAULT_CMD_PORT = 5562

# Colors
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)
GREEN = (80, 220, 80)
RED = (220, 80, 80)
YELLOW = (220, 220, 80)
CYAN = (80, 220, 220)
DIM = (100, 100, 100)
WHITE = (255, 255, 255)


def main():
    parser = argparse.ArgumentParser(description="ZMQ pygame controller for Go2 WM locomotion (simultaneous keys)")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CMD_PORT,
        help=f"ZMQ PUB port (default: {DEFAULT_CMD_PORT})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="*",
        help="Bind address (default: * for all interfaces). Use a specific IP to restrict.",
    )
    args = parser.parse_args()

    # Initialize ZMQ publisher
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    endpoint = f"tcp://{args.host}:{args.port}"
    pub.bind(endpoint)
    print(f"ZMQ command publisher bound to {endpoint}")

    def send_cmd(vx: float, vy: float, wz: float):
        pub.send(struct.pack("fff", vx, vy, wz))

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((520, 340))
    pygame.display.set_caption("Go2 WM Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)
    font_bold = pygame.font.SysFont("monospace", 18, bold=True)
    font_large = pygame.font.SysFont("monospace", 22, bold=True)

    speed_mode = "slow"
    vx, vy, wz = 0.0, 0.0, 0.0
    running = True

    try:
        while running:
            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_1:
                        speed_mode = "slow"
                    elif event.key == pygame.K_2:
                        speed_mode = "fast"
                    elif event.key == pygame.K_SPACE:
                        vx, vy, wz = 0.0, 0.0, 0.0
                        send_cmd(0.0, 0.0, 0.0)

            if not running:
                break

            # --- Simultaneous key state polling ---
            keys = pygame.key.get_pressed()
            limits = SPEED_MODES[speed_mode]

            new_vx = 0.0
            new_vy = 0.0
            new_wz = 0.0

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                new_vx += limits["forward"]
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                new_vx -= limits["forward"]
            if keys[pygame.K_a]:
                new_vy += limits["lateral"]
            if keys[pygame.K_d]:
                new_vy -= limits["lateral"]
            if keys[pygame.K_LEFT] or keys[pygame.K_j]:
                new_wz += limits["angular"]
            if keys[pygame.K_RIGHT] or keys[pygame.K_k]:
                new_wz -= limits["angular"]

            vx, vy, wz = new_vx, new_vy, new_wz

            # --- Publish ---
            send_cmd(vx, vy, wz)

            # --- Draw ---
            screen.fill(BG_COLOR)

            y = 12
            screen.blit(font_large.render("Go2 WM Control (pygame)", True, WHITE), (14, y))
            y += 32

            controls = [
                "W/Up S/Dn : Forward / Backward",
                "A    D     : Strafe left / right",
                "Left/J Rt/K: Turn left / right",
                "1 / 2      : Slow / Fast mode",
                "SPACE: E-stop    Q/ESC: Quit",
            ]
            for line in controls:
                screen.blit(font.render(line, True, DIM), (14, y))
                y += 22

            y += 10
            mode_color = YELLOW if speed_mode == "fast" else GREEN
            screen.blit(font_bold.render(f"Mode: {speed_mode.upper()}", True, mode_color), (14, y))
            y += 24
            screen.blit(
                font.render(
                    f"Limits: vx=+/-{limits['forward']:.1f}  vy=+/-{limits['lateral']:.1f}  wz=+/-{limits['angular']:.1f}",
                    True,
                    TEXT_COLOR,
                ),
                (14, y),
            )
            y += 30

            pygame.draw.line(screen, DIM, (14, y), (506, y))
            y += 10

            vx_color = GREEN if abs(vx) > 0.01 else DIM
            vy_color = CYAN if abs(vy) > 0.01 else DIM
            wz_color = YELLOW if abs(wz) > 0.01 else DIM
            screen.blit(font.render(f"Forward : {vx:+.2f} m/s", True, vx_color), (14, y))
            y += 22
            screen.blit(font.render(f"Lateral : {vy:+.2f} m/s", True, vy_color), (14, y))
            y += 22
            screen.blit(font.render(f"Angular : {wz:+.2f} rad/s", True, wz_color), (14, y))
            y += 28

            moving = abs(vx) > 0.01 or abs(vy) > 0.01 or abs(wz) > 0.01
            status = "MOVING" if moving else "STOPPED"
            status_color = GREEN if moving else RED
            screen.blit(font_bold.render(f"[{status}]", True, status_color), (14, y))
            screen.blit(font.render(f"ZMQ port: {args.port}", True, DIM), (140, y))

            pygame.display.flip()
            clock.tick(PUBLISH_HZ)

    finally:
        # Send zero commands before exiting
        for _ in range(10):
            send_cmd(0.0, 0.0, 0.0)
            time.sleep(0.02)

        pygame.quit()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
