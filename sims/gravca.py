"""
Gravity CA - scalar potential field, particles emit + feel gradient.
"""
import pyray as rl
from pyray import ffi
import numpy as np
import random
import math

SCREEN_W   = 1920
SCREEN_H   = 1080
GRID_SIZE  = 200
UI_PANEL_W = 280
SIM_W      = SCREEN_H
SIM_H      = SCREEN_H
CELL       = SIM_W / GRID_SIZE
SIM_OX     = UI_PANEL_W

# --- sim params ---
EMIT_RATE    = 2.0    # field emitted per unit mass per step
DIFFUSE_RATE = 0.24   # fraction that spreads to 4 neighbors each sub-step
DIFFUSE_STEPS= 4      # sub-steps of diffusion per sim step
DECAY_RATE   = 0.015  # field lost per step (keeps total bounded)
GRAD_ACCEL   = 0.12   # gradient -> acceleration multiplier
VEL_DAMPING  = 0.985
VEL_MAX      = 2.0
FIELD_CAP    = 200.0

field     = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
mass      = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
gvx       = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
gvy       = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

pixels    = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
is_paused = False
show_velocity  = True
vel_scale      = 8.0

_drag_states = [False] * 8


def add_particles(n=10):
    placed, attempts = 0, 0
    while placed < n and attempts < n * 20:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        if mass[y, x] == 0.0:
            mass[y, x] = 1.0 + random.random() * 2.0
            angle = random.uniform(0, 2 * math.pi)
            spd   = random.uniform(0.05, 0.3)
            gvx[y, x] = math.cos(angle) * spd
            gvy[y, x] = math.sin(angle) * spd
            placed += 1
        attempts += 1


def clear_all():
    field[:] = 0.0
    mass[:]  = 0.0
    gvx[:]   = 0.0
    gvy[:]   = 0.0


def step():
    global field, mass, gvx, gvy

    # 1. emit: mass deposits field at its location
    field += mass * EMIT_RATE

    # 2. diffuse: spread field to neighbors (multiple sub-steps)
    for _ in range(DIFFUSE_STEPS):
        neighbors = (
                np.roll(field,  1, axis=0) +
                np.roll(field, -1, axis=0) +
                np.roll(field,  1, axis=1) +
                np.roll(field, -1, axis=1)
        )
        field += DIFFUSE_RATE * (neighbors - 4.0 * field)

    # 3. decay
    field *= (1.0 - DECAY_RATE)
    np.clip(field, 0.0, FIELD_CAP, out=field)

    # 4. gradient -> force on particles
    # central difference, wrapping
    gx = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) * 0.5
    gy = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) * 0.5

    has_mass = mass > 0.0
    # gradient points away from mass, so we go UP gradient = toward mass
    gvx = np.where(has_mass, (gvx + gx * GRAD_ACCEL) * VEL_DAMPING, 0.0)
    gvy = np.where(has_mass, (gvy + gy * GRAD_ACCEL) * VEL_DAMPING, 0.0)

    # clamp velocity
    vmag = np.sqrt(gvx * gvx + gvy * gvy)
    over = vmag > VEL_MAX
    gvx  = np.where(over, gvx / vmag * VEL_MAX, gvx)
    gvy  = np.where(over, gvy / vmag * VEL_MAX, gvy)

    # 5. move particles
    avx   = np.abs(gvx)
    avy   = np.abs(gvy)
    total = avx + avy + 1e-9
    ph    = avx / total   # probability of moving horizontally

    vmag      = np.sqrt(gvx * gvx + gvy * gvy)
    move_prob = np.tanh(vmag * 3.0)
    move_mask = has_mass & (vmag > 0.001)

    r1 = np.random.random((GRID_SIZE, GRID_SIZE)).astype(np.float32)
    r2 = np.random.random((GRID_SIZE, GRID_SIZE)).astype(np.float32)

    choose_h = r1 < ph
    mdx = np.where(choose_h, np.sign(gvx).astype(np.int32), 0)
    mdy = np.where(choose_h, 0, np.sign(gvy).astype(np.int32))

    has_dir = (mdx != 0) | (mdy != 0)
    do_move = move_mask & has_dir & (r2 < move_prob)

    move_ys, move_xs = np.where(do_move)

    if len(move_xs) > 0:
        step_dx   = mdx[move_ys, move_xs]
        step_dy   = mdy[move_ys, move_xs]
        target_xs = (move_xs + step_dx) % GRID_SIZE
        target_ys = (move_ys + step_dy) % GRID_SIZE

        # deduplicate targets
        target_keys = target_ys * GRID_SIZE + target_xs
        _, uid = np.unique(target_keys, return_index=True)
        move_xs   = move_xs[uid];   move_ys   = move_ys[uid]
        target_xs = target_xs[uid]; target_ys = target_ys[uid]
        step_dx   = step_dx[uid];   step_dy   = step_dy[uid]

        occupied = mass[target_ys, target_xs] != 0.0

        # elastic collision: swap velocities
        cmx = move_xs[occupied]; cmy = move_ys[occupied]
        ctx = target_xs[occupied]; cty = target_ys[occupied]
        if len(cmx) > 0:
            tvx = gvx[cmy, cmx].copy(); tvy = gvy[cmy, cmx].copy()
            gvx[cmy, cmx] = gvx[cty, ctx]; gvy[cmy, cmx] = gvy[cty, ctx]
            gvx[cty, ctx] = tvx;           gvy[cty, ctx] = tvy

        free      = ~occupied
        move_xs   = move_xs[free];   move_ys   = move_ys[free]
        target_xs = target_xs[free]; target_ys = target_ys[free]

        if len(move_xs) > 0:
            nm = mass.copy(); nvx = gvx.copy(); nvy = gvy.copy()
            nm[target_ys,  target_xs]  = mass[move_ys, move_xs]
            nvx[target_ys, target_xs]  = gvx[move_ys,  move_xs]
            nvy[target_ys, target_xs]  = gvy[move_ys,  move_xs]
            nm[move_ys,  move_xs]  = 0.0
            nvx[move_ys, move_xs]  = 0.0
            nvy[move_ys, move_xs]  = 0.0
            mass = nm; gvx = nvx; gvy = nvy


def build_pixels():
    pixels[:, :, 3] = 255
    f_norm = np.clip(field / 40.0, 0.0, 1.0)
    r = (f_norm * 30).astype(np.uint8)
    g = (f_norm * 10).astype(np.uint8)
    b = (f_norm * 80).astype(np.uint8)
    pixels[:, :, 0] = r
    pixels[:, :, 1] = g
    pixels[:, :, 2] = b
    has_m = mass > 0.0
    brightness = np.clip(mass / 3.0, 0.3, 1.0)
    pixels[:, :, 0] = np.where(has_m, (brightness * 255).astype(np.uint8), pixels[:, :, 0])
    pixels[:, :, 1] = np.where(has_m, (brightness * 180).astype(np.uint8), pixels[:, :, 1])
    pixels[:, :, 2] = np.where(has_m, 60, pixels[:, :, 2])


def draw_velocity_vectors():
    pys, pxs = np.where(mass > 0.0)
    for i in range(len(pys)):
        py, px = int(pys[i]), int(pxs[i])
        vx = float(gvx[py, px]); vy = float(gvy[py, px])
        vm = math.sqrt(vx*vx + vy*vy)
        if vm < 0.005:
            continue
        cx = int(SIM_OX + (px + 0.5) * CELL)
        cy = int((py + 0.5) * CELL)
        ex = int(cx + vx * vel_scale * CELL)
        ey = int(cy + vy * vel_scale * CELL)
        rl.draw_line(cx, cy, ex, ey, rl.Color(255, 220, 0, 200))


def draw_slider(label, x, y, w, val, vmin, vmax, drag_idx, fmt="{:.3f}"):
    BAR_H   = 8
    KNOB_R  = 7
    LABEL_H = 18
    rl.draw_text(label, x, y, LABEL_H, rl.Color(40, 40, 40, 255))
    y += LABEL_H + 4
    ky = y + KNOB_R
    rl.draw_rectangle(x, ky - BAR_H // 2, w, BAR_H, rl.Color(180, 180, 180, 255))
    t  = (val - vmin) / (vmax - vmin)
    kx = int(x + t * w)
    mp  = rl.get_mouse_position()
    lmb = rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT)
    if lmb and not _drag_states[drag_idx]:
        if abs(mp.x - kx) <= KNOB_R + 6 and abs(mp.y - ky) <= KNOB_R + 6:
            _drag_states[drag_idx] = True
    if _drag_states[drag_idx]:
        if lmb:
            t   = max(0.0, min(1.0, (mp.x - x) / w))
            val = vmin + t * (vmax - vmin)
        else:
            _drag_states[drag_idx] = False
    t  = (val - vmin) / (vmax - vmin)
    kx = int(x + t * w)
    rl.draw_circle(kx, ky, KNOB_R, rl.Color(60, 120, 200, 255))
    rl.draw_text(fmt.format(val), x + w + 8, y, LABEL_H, rl.Color(40, 40, 40, 255))
    return val, ky + KNOB_R + 10


def draw_ui(mgx, mgy):
    global is_paused, show_velocity, vel_scale
    global EMIT_RATE, DIFFUSE_RATE, DIFFUSE_STEPS, DECAY_RATE, GRAD_ACCEL, VEL_DAMPING

    rl.draw_rectangle(0, 0, UI_PANEL_W, SCREEN_H, rl.Color(225, 225, 225, 255))
    rl.draw_line(UI_PANEL_W, 0, UI_PANEL_W, SCREEN_H, rl.Color(160, 160, 160, 255))

    yc = 16
    rl.draw_text("GRAVITY CA", 16, yc, 26, rl.BLACK); yc += 44

    bw = UI_PANEL_W - 32
    bh = 36

    def button(label, y, col, hcol):
        r   = rl.Rectangle(16, y, bw, bh)
        hov = rl.check_collision_point_rec(rl.get_mouse_position(), r)
        rl.draw_rectangle_rec(r, hcol if hov else col)
        rl.draw_text(label, 28, y + 9, 17, rl.WHITE)
        return hov and rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)

    if button("Add 10 Particles", yc, rl.Color(50,100,160,255), rl.Color(70,130,200,255)):
        add_particles(10)
    yc += bh + 6

    if button("Clear All", yc, rl.Color(160,50,50,255), rl.Color(200,70,70,255)):
        clear_all()
    yc += bh + 6

    lbl = "Resume [SPACE]" if is_paused else "Pause  [SPACE]"
    if button(lbl, yc, rl.Color(50,140,50,255), rl.Color(70,180,70,255)):
        is_paused = not is_paused
    yc += bh + 6

    tog_r = rl.Rectangle(16, yc, bw, bh)
    hov   = rl.check_collision_point_rec(rl.get_mouse_position(), tog_r)
    rl.draw_rectangle_rec(tog_r, rl.Color(50,160,50,255) if show_velocity else rl.Color(140,140,140,255))
    rl.draw_text("Velocity Vectors", 28, yc + 9, 17, rl.WHITE)
    if hov and rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
        show_velocity = not show_velocity
    yc += bh + 14

    SW = bw - 48
    EMIT_RATE,    yc = draw_slider("Emit Rate",       16, yc, SW, EMIT_RATE,    0.0, 10.0, 0)
    DIFFUSE_RATE, yc = draw_slider("Diffuse Rate",    16, yc, SW, DIFFUSE_RATE, 0.01, 0.24, 1)
    DECAY_RATE,   yc = draw_slider("Decay Rate",      16, yc, SW, DECAY_RATE,  0.001, 0.1, 2)
    GRAD_ACCEL,   yc = draw_slider("Grad Accel",      16, yc, SW, GRAD_ACCEL,  0.0, 0.5,  3)
    VEL_DAMPING,  yc = draw_slider("Vel Damping",     16, yc, SW, VEL_DAMPING, 0.9, 1.0,  4, "{:.4f}")
    vel_scale,    yc = draw_slider("Vel Display",     16, yc, SW, vel_scale,   1.0, 20.0, 5)
    yc += 4

    n_parts   = int(np.sum(mass > 0.0))
    avg_field = float(np.mean(field))
    max_field = float(np.max(field))
    rl.draw_text(f"Particles: {n_parts}",        16, yc, 17, rl.BLACK); yc += 22
    rl.draw_text(f"Avg field: {avg_field:.2f}",  16, yc, 17, rl.BLACK); yc += 22
    rl.draw_text(f"Max field: {max_field:.2f}",  16, yc, 17, rl.BLACK); yc += 22

    if 0 <= mgx < GRID_SIZE and 0 <= mgy < GRID_SIZE:
        gf  = float(field[mgy, mgx])
        m   = float(mass[mgy, mgx])
        vx  = float(gvx[mgy, mgx])
        vy  = float(gvy[mgy, mgx])
        vm  = math.sqrt(vx*vx + vy*vy)
        rl.draw_text(f"Cell ({mgx},{mgy})", 16, yc, 16, rl.Color(80,80,80,255)); yc += 20
        rl.draw_text(f"Field: {gf:.3f}",   16, yc, 16, rl.BLACK); yc += 20
        rl.draw_text(f"Mass:  {m:.2f}",    16, yc, 16, rl.BLACK); yc += 20
        rl.draw_text(f"Speed: {vm:.4f}",   16, yc, 16, rl.BLACK); yc += 20

    rl.draw_text(f"FPS: {rl.get_fps()}", 16, SCREEN_H - 24, 15, rl.Color(120,120,120,255))


def main():
    global is_paused

    rl.init_window(SCREEN_W, SCREEN_H, "Gravity CA")
    rl.set_target_fps(60)

    img     = rl.gen_image_color(GRID_SIZE, GRID_SIZE, rl.BLACK)
    sim_tex = rl.load_texture_from_image(img)
    rl.unload_image(img)
    rl.set_texture_filter(sim_tex, rl.TEXTURE_FILTER_POINT)

    add_particles(20)

    pixels_ptr = ffi.cast("void *", ffi.from_buffer(pixels))

    while not rl.window_should_close():
        if rl.is_key_pressed(rl.KEY_SPACE):
            is_paused = not is_paused

        mp  = rl.get_mouse_position()
        mgx = int((mp.x - SIM_OX) / CELL)
        mgy = int(mp.y / CELL)

        if rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT):
            if SIM_OX <= mp.x < SIM_OX + SIM_W and 0 <= mp.y < SIM_H:
                if 0 <= mgx < GRID_SIZE and 0 <= mgy < GRID_SIZE:
                    if mass[mgy, mgx] == 0.0:
                        mass[mgy, mgx] = 1.0 + random.random() * 2.0
                        angle = random.uniform(0, 2 * math.pi)
                        spd   = random.uniform(0.05, 0.3)
                        gvx[mgy, mgx] = math.cos(angle) * spd
                        gvy[mgy, mgx] = math.sin(angle) * spd

        if not is_paused:
            step()

        build_pixels()
        rl.update_texture(sim_tex, pixels_ptr)

        rl.begin_drawing()
        rl.clear_background(rl.BLACK)

        src = rl.Rectangle(0, 0, GRID_SIZE, GRID_SIZE)
        dst = rl.Rectangle(SIM_OX, 0, SIM_W, SIM_H)
        rl.draw_texture_pro(sim_tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

        if show_velocity:
            draw_velocity_vectors()

        draw_ui(mgx, mgy)
        rl.end_drawing()

    rl.unload_texture(sim_tex)
    rl.close_window()


if __name__ == "__main__":
    main()
