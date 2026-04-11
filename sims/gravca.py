"""
Gravity Cellular automata proof of concept.
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
SIM_W      = SCREEN_W - UI_PANEL_W
SIM_H      = SCREEN_H
SIM_OX     = UI_PANEL_W

EMIT_RATE     = 2.0
DIFFUSE_RATE  = 0.24
DIFFUSE_STEPS = 4
DECAY_RATE    = 0.015
GRAD_ACCEL    = 0.12
VEL_DAMPING   = 0.985
VEL_MAX       = 2.0
FIELD_CAP     = 200.0

field = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
mass  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
gvx   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
gvy   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

render_w = SIM_W
render_h = SIM_H
pixels   = np.zeros((render_h, render_w, 4), dtype=np.uint8)

_neighbors = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
_r1        = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
_r2        = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
_vmag      = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

_gxi2d = np.zeros((render_h, render_w), dtype=np.int32)
_gyi2d = np.zeros((render_h, render_w), dtype=np.int32)
_view_dirty = True

is_paused     = False
show_velocity = True
vel_scale     = 8.0

view_cx = float(GRID_SIZE // 2)
view_cy = float(GRID_SIZE // 2)
zoom    = float(SIM_W) / float(GRID_SIZE)
ZOOM_MIN = 0.05
ZOOM_MAX = 40.0

_pan_last    = None
_drag_states = [False] * 8


def screen_to_grid(sx, sy):
    cx = (sx - SIM_OX - SIM_W * 0.5) / zoom + view_cx
    cy = (sy           - SIM_H * 0.5) / zoom + view_cy
    return cx, cy


def grid_to_screen(gx, gy):
    sx = (gx - view_cx) * zoom + SIM_W * 0.5 + SIM_OX
    sy = (gy - view_cy) * zoom + SIM_H * 0.5
    return sx, sy


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


def _roll_neighbors(src, out):
    out[:] = 0.0
    out[1:,  :]  += src[:-1, :]
    out[0,   :]  += src[-1,  :]
    out[:-1, :]  += src[1:,  :]
    out[-1,  :]  += src[0,   :]
    out[:,  1:]  += src[:, :-1]
    out[:,   0]  += src[:,  -1]
    out[:, :-1]  += src[:,  1:]
    out[:,  -1]  += src[:,   0]


def step():
    global field, mass, gvx, gvy, _vmag, _neighbors, _r1, _r2

    field += mass * EMIT_RATE

    for _ in range(DIFFUSE_STEPS):
        _roll_neighbors(field, _neighbors)
        field += DIFFUSE_RATE * (_neighbors - 4.0 * field)

    field *= (1.0 - DECAY_RATE)
    np.clip(field, 0.0, FIELD_CAP, out=field)

    gx = np.empty_like(field)
    gy = np.empty_like(field)
    gx[:, 1:-1] = (field[:, 2:]  - field[:, :-2]) * 0.5
    gx[:,  0]   = (field[:,  1]  - field[:,  -1])  * 0.5
    gx[:, -1]   = (field[:,  0]  - field[:, -2])   * 0.5
    gy[1:-1, :] = (field[2:,  :] - field[:-2, :])  * 0.5
    gy[0,    :] = (field[1,   :] - field[-1,  :])  * 0.5
    gy[-1,   :] = (field[0,   :] - field[-2,  :])  * 0.5

    has_mass = mass > 0.0
    gvx[has_mass]  = (gvx[has_mass] + gx[has_mass] * GRAD_ACCEL) * VEL_DAMPING
    gvy[has_mass]  = (gvy[has_mass] + gy[has_mass] * GRAD_ACCEL) * VEL_DAMPING
    gvx[~has_mass] = 0.0
    gvy[~has_mass] = 0.0

    np.multiply(gvx, gvx, out=_vmag)
    _vmag += gvy * gvy
    np.sqrt(_vmag, out=_vmag)

    over = _vmag > VEL_MAX
    if np.any(over):
        safe = np.where(over, _vmag, 1.0)
        gvx[over] = gvx[over] / safe[over] * VEL_MAX
        gvy[over] = gvy[over] / safe[over] * VEL_MAX

    avx   = np.abs(gvx)
    avy   = np.abs(gvy)
    total = avx + avy + 1e-9
    ph    = avx / total

    np.multiply(gvx, gvx, out=_vmag)
    _vmag += gvy * gvy
    np.sqrt(_vmag, out=_vmag)

    move_prob = np.tanh(_vmag * 3.0)
    move_mask = has_mass & (_vmag > 0.001)

    _r1[:] = np.random.random((GRID_SIZE, GRID_SIZE))
    _r2[:] = np.random.random((GRID_SIZE, GRID_SIZE))

    choose_h = _r1 < ph
    mdx = np.where(choose_h, np.sign(gvx).astype(np.int32), 0)
    mdy = np.where(choose_h, 0, np.sign(gvy).astype(np.int32))

    has_dir = (mdx != 0) | (mdy != 0)
    do_move = move_mask & has_dir & (_r2 < move_prob)

    move_ys, move_xs = np.where(do_move)

    if len(move_xs) > 0:
        step_dx   = mdx[move_ys, move_xs]
        step_dy   = mdy[move_ys, move_xs]
        target_xs = (move_xs + step_dx) % GRID_SIZE
        target_ys = (move_ys + step_dy) % GRID_SIZE

        target_keys = target_ys * GRID_SIZE + target_xs
        _, uid = np.unique(target_keys, return_index=True)
        move_xs   = move_xs[uid];   move_ys   = move_ys[uid]
        target_xs = target_xs[uid]; target_ys = target_ys[uid]

        occupied = mass[target_ys, target_xs] != 0.0

        cmx = move_xs[occupied]; cmy = move_ys[occupied]
        ctx = target_xs[occupied]; cty = target_ys[occupied]
        if len(cmx) > 0:
            tvx = gvx[cmy, cmx].copy(); tvy = gvy[cmy, cmx].copy()
            gvx[cmy, cmx] = gvx[cty, ctx]; gvy[cmy, cmx] = gvy[cty, ctx]
            gvx[cty, ctx] = tvx;           gvy[cty, ctx] = tvy

        free = ~occupied
        fmx = move_xs[free];   fmy = move_ys[free]
        ftx = target_xs[free]; fty = target_ys[free]

        if len(fmx) > 0:
            mass[fty, ftx] = mass[fmy, fmx]
            gvx[fty,  ftx] = gvx[fmy,  fmx]
            gvy[fty,  ftx] = gvy[fmy,  fmx]
            mass[fmy, fmx] = 0.0
            gvx[fmy,  fmx] = 0.0
            gvy[fmy,  fmx] = 0.0


def _rebuild_index_maps():
    global _gxi2d, _gyi2d
    px_xs = np.arange(render_w, dtype=np.float32)
    px_ys = np.arange(render_h, dtype=np.float32)
    gxf = (px_xs - SIM_W * 0.5) / zoom + view_cx
    gyf = (px_ys - SIM_H * 0.5) / zoom + view_cy
    gxi = np.clip(gxf.astype(np.int32), 0, GRID_SIZE - 1)
    gyi = np.clip(gyf.astype(np.int32), 0, GRID_SIZE - 1)
    _gyi2d = gyi[:, np.newaxis] * np.ones(render_w, dtype=np.int32)
    _gxi2d = np.ones(render_h, dtype=np.int32)[:, np.newaxis] * gxi[np.newaxis, :]


def build_pixels():
    f_vals = field[_gyi2d, _gxi2d]
    m_vals = mass[_gyi2d,  _gxi2d]

    f_norm = np.clip(f_vals / 40.0, 0.0, 1.0)
    pixels[:, :, 0] = (f_norm * 30).astype(np.uint8)
    pixels[:, :, 1] = (f_norm * 10).astype(np.uint8)
    pixels[:, :, 2] = (f_norm * 80).astype(np.uint8)
    pixels[:, :, 3] = 255

    has_m  = m_vals > 0.0
    bright = np.clip(m_vals / 3.0, 0.3, 1.0)
    b255   = (bright * 255).astype(np.uint8)
    b180   = (bright * 180).astype(np.uint8)
    pixels[:, :, 0][has_m] = b255[has_m]
    pixels[:, :, 1][has_m] = b180[has_m]
    pixels[:, :, 2][has_m] = 60


def draw_velocity_vectors():
    if zoom < 2.0:
        return
    pys, pxs = np.where(mass > 0.0)
    for i in range(len(pys)):
        py, px = int(pys[i]), int(pxs[i])
        vx = float(gvx[py, px]); vy = float(gvy[py, px])
        vm = math.sqrt(vx*vx + vy*vy)
        if vm < 0.005:
            continue
        sx, sy = grid_to_screen(px + 0.5, py + 0.5)
        if sx < SIM_OX or sx > SCREEN_W or sy < 0 or sy > SIM_H:
            continue
        ex = sx + vx * vel_scale * zoom
        ey = sy + vy * vel_scale * zoom
        rl.draw_line(int(sx), int(sy), int(ex), int(ey), rl.Color(255, 220, 0, 200))


def draw_slider(label, x, y, w, val, vmin, vmax, drag_idx, fmt="{:.3f}"):
    KNOB_R  = 7
    LABEL_H = 18
    rl.draw_text(label, x, y, LABEL_H, rl.Color(40, 40, 40, 255))
    y += LABEL_H + 4
    ky = y + KNOB_R
    rl.draw_rectangle(x, ky - 4, w, 8, rl.Color(180, 180, 180, 255))
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
    global EMIT_RATE, DIFFUSE_RATE, DECAY_RATE, GRAD_ACCEL, VEL_DAMPING

    rl.draw_rectangle(0, 0, UI_PANEL_W, SCREEN_H, rl.Color(225, 225, 225, 255))
    rl.draw_line(UI_PANEL_W, 0, UI_PANEL_W, SCREEN_H, rl.Color(160, 160, 160, 255))

    yc = 16
    rl.draw_text("GRAVITY CA", 16, yc, 26, rl.BLACK); yc += 44

    bw = UI_PANEL_W - 32
    bh = 34

    def button(label, y, col, hcol):
        r   = rl.Rectangle(16, y, bw, bh)
        hov = rl.check_collision_point_rec(rl.get_mouse_position(), r)
        rl.draw_rectangle_rec(r, hcol if hov else col)
        rl.draw_text(label, 28, y + 8, 16, rl.WHITE)
        return hov and rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)

    blue  = (rl.Color(50,100,160,255), rl.Color(70,130,200,255))
    red   = (rl.Color(160,50,50,255),  rl.Color(200,70,70,255))
    green = (rl.Color(50,140,50,255),  rl.Color(70,180,70,255))

    if button("Add 10 Particles",   yc, *blue):  add_particles(10)
    yc += bh + 4
    if button("Add 100 Particles",  yc, *blue):  add_particles(100)
    yc += bh + 4
    if button("Add 1000 Particles", yc, *blue):  add_particles(1000)
    yc += bh + 4
    if button("Clear All",          yc, *red):   clear_all()
    yc += bh + 4

    lbl = "Resume [SPACE]" if is_paused else "Pause  [SPACE]"
    if button(lbl, yc, *green): is_paused = not is_paused
    yc += bh + 4

    tog_r = rl.Rectangle(16, yc, bw, bh)
    hov   = rl.check_collision_point_rec(rl.get_mouse_position(), tog_r)
    rl.draw_rectangle_rec(tog_r, rl.Color(50,160,50,255) if show_velocity else rl.Color(140,140,140,255))
    rl.draw_text("Velocity Vectors", 28, yc + 8, 16, rl.WHITE)
    if hov and rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
        show_velocity = not show_velocity
    yc += bh + 12

    SW = bw - 52
    EMIT_RATE,    yc = draw_slider("Emit Rate",    16, yc, SW, EMIT_RATE,    0.0,  10.0, 0)
    DIFFUSE_RATE, yc = draw_slider("Diffuse Rate", 16, yc, SW, DIFFUSE_RATE, 0.01, 0.24, 1)
    DECAY_RATE,   yc = draw_slider("Decay Rate",   16, yc, SW, DECAY_RATE,   0.001,0.1,  2)
    GRAD_ACCEL,   yc = draw_slider("Grad Accel",   16, yc, SW, GRAD_ACCEL,   0.0,  0.5,  3)
    VEL_DAMPING,  yc = draw_slider("Vel Damping",  16, yc, SW, VEL_DAMPING,  0.9,  1.0,  4, "{:.4f}")
    vel_scale,    yc = draw_slider("Vel Display",  16, yc, SW, vel_scale,    1.0,  20.0, 5)
    yc += 6

    n_parts   = int(np.sum(mass > 0.0))
    avg_field = float(np.mean(field))
    max_field = float(np.max(field))
    rl.draw_text(f"Particles: {n_parts}",       16, yc, 16, rl.BLACK); yc += 20
    rl.draw_text(f"Avg field: {avg_field:.2f}", 16, yc, 16, rl.BLACK); yc += 20
    rl.draw_text(f"Max field: {max_field:.2f}", 16, yc, 16, rl.BLACK); yc += 20
    rl.draw_text(f"Zoom: {zoom:.3f}x",          16, yc, 16, rl.BLACK); yc += 20

    if 0 <= mgx < GRID_SIZE and 0 <= mgy < GRID_SIZE:
        gf  = float(field[mgy, mgx])
        m   = float(mass[mgy, mgx])
        vx  = float(gvx[mgy, mgx])
        vy  = float(gvy[mgy, mgx])
        vm  = math.sqrt(vx*vx + vy*vy)
        rl.draw_text(f"Cell ({mgx},{mgy})", 16, yc, 15, rl.Color(80,80,80,255)); yc += 18
        rl.draw_text(f"Field: {gf:.3f}",   16, yc, 15, rl.BLACK); yc += 18
        rl.draw_text(f"Mass:  {m:.2f}",    16, yc, 15, rl.BLACK); yc += 18
        rl.draw_text(f"Speed: {vm:.4f}",   16, yc, 15, rl.BLACK); yc += 18

    rl.draw_text("RMB drag: pan", 16, SCREEN_H - 52, 14, rl.Color(120,120,120,255))
    rl.draw_text("Scroll: zoom",  16, SCREEN_H - 34, 14, rl.Color(120,120,120,255))
    rl.draw_text(f"FPS: {rl.get_fps()}", 16, SCREEN_H - 16, 14, rl.Color(120,120,120,255))


def main():
    global is_paused, view_cx, view_cy, zoom, _pan_last, _view_dirty

    rl.init_window(SCREEN_W, SCREEN_H, "Gravity CA")
    rl.set_target_fps(60)

    img     = rl.gen_image_color(render_w, render_h, rl.BLACK)
    sim_tex = rl.load_texture_from_image(img)
    rl.unload_image(img)
    rl.set_texture_filter(sim_tex, rl.TEXTURE_FILTER_POINT)

    add_particles(20)
    _rebuild_index_maps()
    _view_dirty = False

    pixels_ptr = ffi.cast("void *", ffi.from_buffer(pixels))

    while not rl.window_should_close():
        if rl.is_key_pressed(rl.KEY_SPACE):
            is_paused = not is_paused

        mp     = rl.get_mouse_position()
        in_sim = SIM_OX <= mp.x < SCREEN_W and 0 <= mp.y < SIM_H

        if in_sim:
            wheel = rl.get_mouse_wheel_move()
            if wheel != 0.0:
                gx_before, gy_before = screen_to_grid(mp.x, mp.y)
                zoom *= (1.15 ** wheel)
                zoom  = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
                gx_after, gy_after = screen_to_grid(mp.x, mp.y)
                view_cx += gx_before - gx_after
                view_cy += gy_before - gy_after
                _view_dirty = True

        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_RIGHT) and in_sim:
            _pan_last = (mp.x, mp.y)
        if rl.is_mouse_button_down(rl.MOUSE_BUTTON_RIGHT) and _pan_last is not None:
            dx = mp.x - _pan_last[0]
            dy = mp.y - _pan_last[1]
            view_cx -= dx / zoom
            view_cy -= dy / zoom
            _pan_last   = (mp.x, mp.y)
            _view_dirty = True
        if rl.is_mouse_button_released(rl.MOUSE_BUTTON_RIGHT):
            _pan_last = None

        view_cx = max(0.0, min(float(GRID_SIZE), view_cx))
        view_cy = max(0.0, min(float(GRID_SIZE), view_cy))

        if _view_dirty:
            _rebuild_index_maps()
            _view_dirty = False

        mgx, mgy = screen_to_grid(mp.x, mp.y)
        mgx = int(mgx); mgy = int(mgy)

        if rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT) and in_sim:
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

        src = rl.Rectangle(0, 0, render_w, render_h)
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
