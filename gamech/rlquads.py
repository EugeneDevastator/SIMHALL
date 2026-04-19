"""
raylib transparent quads demo - rlgl batched
A = add 1000 quads
D = remove 1000 quads
ESC = quit

Requirements:
    pip install raylib numpy
"""

import time
import numpy as np
from pyray import *
from raylib import rl

W, H = 1920, 1080
init_window(W, H, "raylib quads")
set_target_fps(0)

MAX_QUADS  = 200_000
quad_count = 0

quads = np.zeros((MAX_QUADS, 10), dtype=np.float32)

COLORS = np.array([
    (0.95, 0.27, 0.27),
    (0.27, 0.62, 0.95),
    (0.35, 0.85, 0.55),
    (0.95, 0.75, 0.25),
    (0.75, 0.35, 0.95),
    (0.95, 0.55, 0.25),
    (0.25, 0.85, 0.85),
    (0.95, 0.35, 0.75),
], dtype=np.float32)


def add_quads(n=1000):
    global quad_count
    n = min(n, MAX_QUADS - quad_count)
    if n <= 0:
        return
    idx = np.random.randint(0, len(COLORS), n)
    s   = quad_count
    quads[s:s+n, 0] = np.random.uniform(0, W, n)
    quads[s:s+n, 1] = np.random.uniform(0, H, n)
    quads[s:s+n, 2] = np.random.uniform(10, 60, n)
    quads[s:s+n, 3] = np.random.uniform(10, 50, n)
    quads[s:s+n, 4:7] = COLORS[idx]
    quads[s:s+n, 7]   = np.random.uniform(0.1, 0.9, n)
    speed = np.random.uniform(60, 240, n)
    angle = np.random.uniform(0, 2 * np.pi, n)
    quads[s:s+n, 8] = np.cos(angle) * speed
    quads[s:s+n, 9] = np.sin(angle) * speed
    quad_count += n
    print(f"  quads: {quad_count:,}")


def remove_quads(n=1000):
    global quad_count
    quad_count = max(0, quad_count - n)
    print(f"  quads: {quad_count:,}")


def update_quads(dt):
    if quad_count == 0:
        return
    q = quads[:quad_count]
    q[:, 0] += q[:, 8] * dt
    q[:, 1] += q[:, 9] * dt
    over_r = q[:, 0] + q[:, 2] > W
    over_l = q[:, 0]            < 0
    q[over_r | over_l, 8] *= -1
    q[:, 0] = np.clip(q[:, 0], 0, W - q[:, 2])
    over_b = q[:, 1] + q[:, 3] > H
    over_t = q[:, 1]            < 0
    q[over_t | over_b, 9] *= -1
    q[:, 1] = np.clip(q[:, 1], 0, H - q[:, 3])


def draw_quads_batched():
    if quad_count == 0:
        return

    q = quads[:quad_count]

    # build vertex arrays in numpy - 6 verts per quad (2 triangles)
    # each vert: x, y  (rlgl uses 2D screen coords here)
    n = quad_count
    x0 = q[:, 0]
    y0 = q[:, 1]
    x1 = q[:, 0] + q[:, 2]
    y1 = q[:, 1] + q[:, 3]

    r = (q[:, 4] * 255).astype(np.uint8)
    g = (q[:, 5] * 255).astype(np.uint8)
    b = (q[:, 6] * 255).astype(np.uint8)
    a = (q[:, 7] * 255).astype(np.uint8)

    # rlgl immediate mode - still per-quad but no Python Color object alloc
    # Use rl_begin / rl_vertex2f / rl_color4ub
    rl.rlBegin(4)
    for i in range(n):
        rl.rlColor4ub(int(r[i]), int(g[i]), int(b[i]), int(a[i]))
        rl.rlVertex2f(float(x0[i]), float(y0[i]))
        rl.rlVertex2f(float(x1[i]), float(y0[i]))
        rl.rlVertex2f(float(x1[i]), float(y1[i]))
        rl.rlVertex2f(float(x0[i]), float(y1[i]))
    rl.rlEnd()


fps_time   = time.perf_counter()
fps_frames = 0

print("Controls:  A = +1000 quads   D = -1000 quads   ESC = quit")
add_quads(1000)

while not window_should_close():
    dt = get_frame_time()

    if is_key_pressed(KEY_A):
        add_quads(1000)
    if is_key_pressed(KEY_D):
        remove_quads(1000)

    update_quads(dt)

    fps_frames += 1
    now = time.perf_counter()
    if now - fps_time >= 0.5:
        fps = fps_frames / (now - fps_time)
        set_window_title(
            f"raylib quads  |  {fps:.0f} FPS  |  {quad_count:,} quads  |  A=+1000  D=-1000  ESC=quit"
        )
        fps_time   = now
        fps_frames = 0

    begin_drawing()
    clear_background(Color(20, 20, 26, 255))

    rl.rlSetTexture(rl.rlGetTextureIdDefault())
    draw_quads_batched()
    rl.rlSetTexture(0)

    draw_text(f"Quads: {quad_count:,}", 20, 20, 32, WHITE)
    end_drawing()

close_window()
