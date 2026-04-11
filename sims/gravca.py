"""
Proof of concept that gravity can be emergent from CA rules 
and probability based movement.
"""
import pyray as rl
from pyray import ffi
import numpy as np
import random
import math

SCREEN_W     = 1920
SCREEN_H     = 1080
GRID_SIZE    = 200
UI_PANEL_W   = 280
SIM_W        = SCREEN_H
SIM_H        = SCREEN_H
CELL         = SIM_W / GRID_SIZE
SIM_OX       = UI_PANEL_W

FIELD_EMIT     = 0.2
DIFFUSION_RATE = 0.3
DECAY_RATE     = 0.98
GRAV_ACCEL     = 0.15
VEL_DAMPING    = 0.97
VEL_THRESHOLD  = 0.001

field = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
mass  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
gvx   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
gvy   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

pixels    = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
is_paused = False

show_velocity   = True
vel_multiplier  = 8.0
field_multiplier = 1.0

# Only axis-aligned dirs for field diffusion
DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def _shift(arr, dy, dx):
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


def add_particles(n=10):
    placed, attempts = 0, 0
    while placed < n and attempts < n * 20:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        if mass[y, x] == 0.0:
            mass[y, x] = 1.0 + random.random() * 2.0
            # give random initial velocity so particles don't start static
            angle = random.uniform(0, 2 * math.pi)
            spd   = random.uniform(0.1, 0.4)
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

    has_mass = mass > 0.0

    # --- field emit + diffuse ---
    field += has_mass * mass * FIELD_EMIT

    neighbor_sum = np.zeros_like(field)
    for dy, dx in DIRS8:
        neighbor_sum += _shift(field, dy, dx)
    field = field * (1.0 - DIFFUSION_RATE) + neighbor_sum * (DIFFUSION_RATE / 8.0)
    field *= DECAY_RATE
    np.clip(field, 0.0, 10.0, out=field)

    # --- gravity force from field gradient (8 dirs) ---
    fx = np.zeros_like(field)
    fy = np.zeros_like(field)
    for dy, dx in DIRS8:
        s = _shift(field, dy, dx)
        fx += dx * s
        fy += dy * s

    accel = GRAV_ACCEL * field_multiplier
    gvx = np.where(has_mass, (gvx + fx * accel) * VEL_DAMPING, gvx)
    gvy = np.where(has_mass, (gvy + fy * accel) * VEL_DAMPING, gvy)

    # --- movement: H or V only, chosen by velocity magnitude ratio ---
    # abs velocity components determine probability of moving that axis
    avx = np.abs(gvx)
    avy = np.abs(gvy)
    total = avx + avy + 1e-9

    # probability of moving horizontally vs vertically
    ph = avx / total   # prob choose H axis
    pv = avy / total   # prob choose V axis  (ph + pv = 1)

    vmag     = np.sqrt(gvx * gvx + gvy * gvy)
    move_prob = np.tanh(vmag * vel_multiplier)
    move_mask = has_mass & (vmag > VEL_THRESHOLD)

    rand1 = np.random.random((GRID_SIZE, GRID_SIZE)).astype(np.float32)
    rand2 = np.random.random((GRID_SIZE, GRID_SIZE)).astype(np.float32)

    # choose axis: rand1 < ph → horizontal, else vertical
    choose_h = rand1 < ph

    # direction sign on chosen axis
    mdx = np.where(choose_h, np.sign(gvx).astype(np.int32), 0)
    mdy = np.where(choose_h, 0, np.sign(gvy).astype(np.int32))

    # zero-velocity edge case: sign(0)=0, skip those
    has_dir  = (mdx != 0) | (mdy != 0)
    do_move  = move_mask & has_dir & (rand2 < move_prob)

    move_ys, move_xs = np.where(do_move)

    if len(move_xs) > 0:
        step_dx = mdx[move_ys, move_xs]
        step_dy = mdy[move_ys, move_xs]

        target_xs = (move_xs + step_dx) % GRID_SIZE
        target_ys = (move_ys + step_dy) % GRID_SIZE

        # deduplicate targets (first-come wins)
        target_keys   = target_ys * GRID_SIZE + target_xs
        _, unique_idx = np.unique(target_keys, return_index=True)
        move_xs   = move_xs[unique_idx]
        move_ys   = move_ys[unique_idx]
        target_xs = target_xs[unique_idx]
        target_ys = target_ys[unique_idx]
        step_dx   = step_dx[unique_idx]
        step_dy   = step_dy[unique_idx]

        occupied = mass[target_ys, target_xs] != 0.0

        # --- collision: exchange velocity vectors ---
        col_mx = move_xs[occupied]
        col_my = move_ys[occupied]
        col_tx = target_xs[occupied]
        col_ty = target_ys[occupied]

        if len(col_mx) > 0:
            tmp_vx = gvx[col_my, col_mx].copy()
            tmp_vy = gvy[col_my, col_mx].copy()
            gvx[col_my, col_mx] = gvx[col_ty, col_tx]
            gvy[col_my, col_mx] = gvy[col_ty, col_tx]
            gvx[col_ty, col_tx] = tmp_vx
            gvy[col_ty, col_tx] = tmp_vy

        # --- free moves ---
        free = ~occupied
        move_xs   = move_xs[free]
        move_ys   = move_ys[free]
        target_xs = target_xs[free]
        target_ys = target_ys[free]

        new_mass = mass.copy()
        new_gvx  = gvx.copy()
        new_gvy  = gvy.copy()

        new_mass[target_ys, target_xs] = mass[move_ys, move_xs]
        new_gvx[target_ys, target_xs]  = gvx[move_ys, move_xs]
        new_gvy[target_ys, target_xs]  = gvy[move_ys, move_xs]
        new_mass[move_ys, move_xs]     = 0.0
        new_gvx[move_ys, move_xs]      = 0.0
        new_gvy[move_ys, move_xs]      = 0.0

        mass = new_mass
        gvx  = new_gvx
        gvy  = new_gvy


def build_pixels():
    pixels[:, :, 3] = 255
    b = np.clip(field * 60.0, 0, 255).astype(np.uint8)
    g = np.clip((field - 1.0) * 40.0, 0, 100).astype(np.uint8)
    pixels[:, :, 0] = 0
    pixels[:, :, 1] = g
    pixels[:, :, 2] = b
    has_m = mass > 0.0
    pixels[:, :, 0] = np.where(has_m, 255, pixels[:, :, 0])
    pixels[:, :, 1] = np.where(has_m, 60,  pixels[:, :, 1])
    pixels[:, :, 2] = np.where(has_m, 60,  pixels[:, :, 2])


def draw_velocity_vectors():
    pys, pxs = np.where(mass > 0.0)
    for i in range(len(pys)):
        py, px = int(pys[i]), int(pxs[i])
        cvx = float(gvx[py, px])
        cvy = float(gvy[py, px])
        vm  = math.sqrt(cvx*cvx + cvy*cvy)
        if vm < 0.005:
            continue
        cx = int(SIM_OX + (px + 0.5) * CELL)
        cy = int((py + 0.5) * CELL)
        ex = int(cx + cvx * 30)
        ey = int(cy + cvy * 30)
        rl.draw_line(cx, cy, ex, ey, rl.Color(255, 230, 0, 255))


def draw_slider(label, x, y, w, val, vmin, vmax, dragging, fmt="{:.2f}"):
    BAR_H  = 10
    KNOB_R = 8
    LABEL_H = 20

    rl.draw_text(label, x, y, LABEL_H, rl.Color(200, 220, 255, 255))
    y += LABEL_H + 4

    bar_rect = rl.Rectangle(x, y + KNOB_R - BAR_H // 2, w, BAR_H)
    rl.draw_rectangle_rec(bar_rect, rl.Color(60, 60, 60, 255))

    t     = (val - vmin) / (vmax - vmin)
    kx    = int(x + t * w)
    ky    = y + KNOB_R
    mp    = rl.get_mouse_position()
    lmb   = rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT)

    if lmb and not dragging:
        if abs(mp.x - kx) <= KNOB_R + 4 and abs(mp.y - ky) <= KNOB_R + 4:
            dragging = True

    if dragging:
        if lmb:
            t   = max(0.0, min(1.0, (mp.x - x) / w))
            val = vmin + t * (vmax - vmin)
        else:
            dragging = False

    t   = (val - vmin) / (vmax - vmin)
    kx  = int(x + t * w)
    rl.draw_circle(kx, ky, KNOB_R, rl.Color(100, 160, 220, 255))

    val_str = fmt.format(val)
    rl.draw_text(val_str, x + w + 8, y, LABEL_H, rl.Color(200, 220, 255, 255))

    return val, dragging, y + KNOB_R * 2 + 10


def draw_ui(mgx, mgy):
    global is_paused, show_velocity, vel_multiplier, field_multiplier
    global _drag_vel, _drag_field

    rl.draw_rectangle(0, 0, UI_PANEL_W, SCREEN_H, rl.Color(28, 28, 28, 255))
    rl.draw_line(UI_PANEL_W, 0, UI_PANEL_W, SCREEN_H, rl.Color(70, 70, 70, 255))

    yc = 20
    rl.draw_text("GRAVITY CA", 16, yc, 30, rl.WHITE)
    yc += 55

    bw = UI_PANEL_W - 32
    bh = 44

    def button(label, y, col, hcol):
        r   = rl.Rectangle(16, y, bw, bh)
        hov = rl.check_collision_point_rec(rl.get_mouse_position(), r)
        rl.draw_rectangle_rec(r, hcol if hov else col)
        rl.draw_text(label, 28, y + 12, 20, rl.WHITE)
        return hov and rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)

    if button("Add Particles", yc, rl.Color(50,100,140,255), rl.Color(70,130,180,255)):
        add_particles(10)
    yc += bh + 8

    if button("Clear All", yc, rl.Color(120,40,40,255), rl.Color(160,60,60,255)):
        clear_all()
    yc += bh + 8

    lbl = "Resume [SPACE]" if is_paused else "Pause  [SPACE]"
    if button(lbl, yc, rl.Color(55,120,55,255), rl.Color(80,160,80,255)):
        is_paused = not is_paused
    yc += bh + 16

    tog_r = rl.Rectangle(16, yc, bw, bh)
    hov   = rl.check_collision_point_rec(rl.get_mouse_position(), tog_r)
    tog_col = rl.Color(80,140,80,255) if show_velocity else rl.Color(80,80,80,255)
    rl.draw_rectangle_rec(tog_r, tog_col)
    rl.draw_text("Velocity Vectors", 28, yc + 12, 20, rl.WHITE)
    if hov and rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
        show_velocity = not show_velocity
    yc += bh + 20

    SW = bw - 40
    vel_multiplier, _drag_vel, yc = draw_slider(
        "Vel Multiplier", 16, yc, SW, vel_multiplier, 0.0, 10.0, _drag_vel)
    yc += 8

    field_multiplier, _drag_field, yc = draw_slider(
        "Field Multiplier", 16, yc, SW, field_multiplier, 1.0, 100.0, _drag_field, fmt="{:.1f}")
    yc += 16

    n_parts = int(np.sum(mass > 0.0))
    rl.draw_text(f"Particles: {n_parts}", 16, yc, 18, rl.Color(200,220,255,255))
    yc += 28

    rl.draw_text("--- Cell Info ---", 16, yc, 18, rl.Color(160,160,160,255))
    yc += 24
    if 0 <= mgx < GRID_SIZE and 0 <= mgy < GRID_SIZE:
        gf  = float(field[mgy, mgx])
        m   = float(mass[mgy, mgx])
        cvx = float(gvx[mgy, mgx])
        cvy = float(gvy[mgy, mgx])
        vm  = math.sqrt(cvx*cvx + cvy*cvy)
        for line in [
            f"Cell: ({mgx},{mgy})",
            f"Field: {gf:.4f}",
            f"Mass:  {m:.2f}",
            f"Vx:    {cvx:.4f}",
            f"Vy:    {cvy:.4f}",
            f"Speed: {vm:.4f}",
        ]:
            rl.draw_text(line, 16, yc, 18, rl.Color(200,220,255,255))
            yc += 22

    yc += 12
    rl.draw_text("--- Legend ---", 16, yc, 18, rl.Color(160,160,160,255))
    yc += 24
    rl.draw_rectangle(16, yc, 18, 18, rl.Color(0, 0, 180, 255))
    rl.draw_text("Gravity field", 42, yc + 2, 16, rl.Color(180,180,255,255))
    yc += 22
    rl.draw_rectangle(16, yc, 18, 18, rl.Color(255, 60, 60, 255))
    rl.draw_text("Particle", 42, yc + 2, 16, rl.Color(255,180,180,255))
    yc += 22
    rl.draw_rectangle(16, yc, 18, 4, rl.Color(255, 230, 0, 255))
    rl.draw_text("Velocity vector", 42, yc - 5, 16, rl.Color(255,240,150,255))

    rl.draw_text(f"FPS: {rl.get_fps()}", 16, SCREEN_H - 30, 18, rl.Color(130,130,130,255))


_drag_vel   = False
_drag_field = False


def main():
    global is_paused

    rl.init_window(SCREEN_W, SCREEN_H, "Gravity CA")
    rl.set_target_fps(60)

    img     = rl.gen_image_color(GRID_SIZE, GRID_SIZE, rl.BLACK)
    sim_tex = rl.load_texture_from_image(img)
    rl.unload_image(img)
    rl.set_texture_filter(sim_tex, rl.TEXTURE_FILTER_POINT)

    add_particles(10)

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
                        spd   = random.uniform(0.1, 0.4)
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
