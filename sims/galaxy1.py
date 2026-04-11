"""
Simplified galaxy lifecycle simulation.
80 stars on Keplerian orbits, 7 spectral classes (O through M).
Watch stars age, go supernova, and collapse into remnants.
Supernova positions marked on orbit. Time speed slider spans
0.001 to 1e40 Gyr/s. Scroll=zoom, RMB/MMB=pan.
"""

import pyray as rl
import math
import random

SCREEN_W = 1400
SCREEN_H = 900
GALAXY_CX = 0.0
GALAXY_CY = 0.0
GALAXY_RADIUS = 380.0
STAR_COUNT = 80

STAR_CLASSES = [
    ("O", (100, 149, 237, 255),  0.003, 40.0,  150.0),
    ("B", (150, 180, 255, 255),  0.05,  3.0,   40.0),
    ("A", (200, 220, 255, 255),  0.5,   1.5,   3.0),
    ("F", (255, 255, 200, 255),  3.0,   1.0,   1.5),
    ("G", (255, 230, 100, 255),  9.0,   0.7,   1.0),
    ("K", (255, 160,  60, 255),  20.0,  0.4,   0.7),
    ("M", (220,  60,  40, 255),  1e13,  0.08,  0.4),
]

IRON_STAR_AGE   = 1e13
BLACK_DWARF_AGE = 1e15
EVAP_START_GYR  = 1e19
EVAP_END_GYR    = 1e20

TERMINAL_COLORS = {
    "black_hole":   (20,  20,  20,  255),
    "neutron_star": (180, 220, 255, 255),
    "white_dwarf":  (220, 220, 255, 255),
    "black_dwarf":  (40,  40,  50,  255),
    "brown_dwarf":  (80,  40,  20,  255),
    "iron_star":    (140, 110,  70,  255),
}
TERMINAL_OUTLINE = {
    "black_hole":   (80,  80, 200, 255),
    "neutron_star": (100, 200, 255, 255),
    "white_dwarf":  (160, 160, 255, 255),
    "black_dwarf":  (80,  80, 100, 255),
    "brown_dwarf":  (120,  70,  30, 255),
    "iron_star":    (200, 160,  80, 255),
}

TWO_PI = 2.0 * math.pi

def terminal_state(mass):
    if mass < 0.08:  return "brown_dwarf"
    if mass > 25.0:  return "black_hole"
    if mass > 8.0:   return "neutron_star"
    return "white_dwarf"

def pick_star_class():
    weights = [2, 5, 8, 10, 12, 12, 15]
    total = sum(weights)
    r = random.random() * total
    acc = 0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(STAR_CLASSES) - 1

def star_radius_px(mass):
    return max(2.0, min(2.0 + math.sqrt(mass) * 1.2, 14.0))

def make_color(t):
    return rl.Color(t[0], t[1], t[2], t[3])

def make_star():
    ci = pick_star_class()
    sc = STAR_CLASSES[ci]
    mass = random.uniform(sc[3], sc[4])
    lifetime_gyr = sc[2] * random.uniform(0.8, 1.2)
    age = random.uniform(0.0, lifetime_gyr * 0.9)
    orbit_r = random.uniform(30.0, GALAXY_RADIUS)
    # ang_vel in rad/Gyr, keplerian
    ang_vel = 1.0 / math.sqrt(orbit_r / 30.0)
    # angle_accum: start random, already advanced by initial age
    angle_accum = math.fmod(
        random.uniform(0.0, TWO_PI) + ang_vel * age,
        TWO_PI
    )
    is_m = (ci == 6)
    dead = age >= lifetime_gyr
    terminal = terminal_state(mass)
    if is_m and age >= IRON_STAR_AGE:
        dead = True
        terminal = "iron_star"
    if dead and terminal == "white_dwarf" and age >= BLACK_DWARF_AGE:
        terminal = "black_dwarf"
    return {
        "orbit_r":      orbit_r,
        "angle_accum":  angle_accum,
        "ang_vel":      ang_vel,
        "class_idx":    ci,
        "mass":         mass,
        "lifetime":     lifetime_gyr,
        "age":          age,
        "is_m":         is_m,
        "terminal":     terminal,
        "has_supernova": mass > 8.0,
        "markers":      [],
        "dead":         dead,
        "evaporated":   False,
    }

def init_sim():
    random.seed(42)
    return [make_star() for _ in range(STAR_COUNT)]

def galaxy_evap_radius(sim_time):
    if sim_time <= EVAP_START_GYR: return GALAXY_RADIUS
    if sim_time >= EVAP_END_GYR:   return 0.0
    t = (sim_time - EVAP_START_GYR) / (EVAP_END_GYR - EVAP_START_GYR)
    return GALAXY_RADIUS * (1.0 - t)

def format_time(gyr):
    if gyr < 1e3:  return f"{gyr:.3g} Gyr"
    if gyr < 1e6:  return f"{gyr:.3e} Gyr"
    return f"{gyr:.3e} Gyr"

def world_to_screen(wx, wy, cam_x, cam_y, zoom):
    return cam_x + wx * zoom, cam_y + wy * zoom

def screen_to_world(sx, sy, cam_x, cam_y, zoom):
    return (sx - cam_x) / zoom, (sy - cam_y) / zoom

def draw_circle_outline(cx, cy, r, color, thickness=2.0):
    if r < 1.0: return
    steps = max(32, int(r * 0.5))
    for i in range(steps):
        a0 = TWO_PI * i / steps
        a1 = TWO_PI * (i + 1) / steps
        rl.draw_line_ex(
            rl.Vector2(cx + r * math.cos(a0), cy + r * math.sin(a0)),
            rl.Vector2(cx + r * math.cos(a1), cy + r * math.sin(a1)),
            thickness, color
        )

rl.init_window(SCREEN_W, SCREEN_H, "Galaxy Lifecycle Sim")
rl.set_target_fps(60)

stars    = init_sim()
sim_time = 0.0

cam_x, cam_y = float(SCREEN_W // 2), float(SCREEN_H // 2)
zoom = 1.0
panning = False
pan_sx, pan_sy = 0.0, 0.0
pan_cx, pan_cy = 0.0, 0.0

TIME_MIN   = 0.001
TIME_MAX   = 1e40
time_speed = 0.1

SL_X, SL_Y, SL_W, SL_H = 20, SCREEN_H - 50, 300, 20
dragging_slider = False

BTN_X, BTN_Y, BTN_W, BTN_H = 20, SCREEN_H - 90, 120, 30

BG  = rl.Color(40, 40, 45, 255)
FSM = 16
FMD = 20
FLG = 24

LMB = rl.MouseButton.MOUSE_BUTTON_LEFT
RMB = rl.MouseButton.MOUSE_BUTTON_RIGHT
MMB = rl.MouseButton.MOUSE_BUTTON_MIDDLE

while not rl.window_should_close():
    dt_real = rl.get_frame_time()
    # clamp dt to avoid spiral of death on lag spikes
    dt_real = min(dt_real, 0.05)
    dt_gyr  = dt_real * time_speed
    sim_time += dt_gyr

    mouse = rl.get_mouse_position()
    mx, my = mouse.x, mouse.y

    # zoom
    wheel = rl.get_mouse_wheel_move()
    if wheel != 0.0:
        wx_b, wy_b = screen_to_world(mx, my, cam_x, cam_y, zoom)
        zoom = max(0.1, min(zoom * (1.1 ** wheel), 20.0))
        cam_x = mx - wx_b * zoom
        cam_y = my - wy_b * zoom

    # pan
    if rl.is_mouse_button_pressed(RMB) or rl.is_mouse_button_pressed(MMB):
        panning = True
        pan_sx, pan_sy = mx, my
        pan_cx, pan_cy = cam_x, cam_y
    if rl.is_mouse_button_released(RMB) or rl.is_mouse_button_released(MMB):
        panning = False
    if panning:
        cam_x = pan_cx + (mx - pan_sx)
        cam_y = pan_cy + (my - pan_sy)

    # restart button
    btn_hit = (BTN_X <= mx <= BTN_X + BTN_W and BTN_Y <= my <= BTN_Y + BTN_H)
    if rl.is_mouse_button_pressed(LMB) and btn_hit:
        stars    = init_sim()
        sim_time = 0.0

    # slider
    sl_hit = (SL_X <= mx <= SL_X + SL_W and SL_Y - 10 <= my <= SL_Y + SL_H + 10)
    if rl.is_mouse_button_pressed(LMB) and sl_hit and not btn_hit:
        dragging_slider = True
    if rl.is_mouse_button_released(LMB):
        dragging_slider = False
    if dragging_slider:
        t = max(0.0, min(1.0, (mx - SL_X) / SL_W))
        lmin = math.log10(TIME_MIN)
        lmax = math.log10(TIME_MAX)
        time_speed = 10.0 ** (lmin + t * (lmax - lmin))

    # update stars
    for s in stars:
        # angle: accumulate with fmod — stays small, no float precision loss
        s["angle_accum"] = math.fmod(s["angle_accum"] + s["ang_vel"] * dt_gyr, TWO_PI)

        if s["dead"]:
            if s["terminal"] == "white_dwarf" and s["age"] >= BLACK_DWARF_AGE:
                s["terminal"] = "black_dwarf"
            s["age"] += dt_gyr
            continue

        s["age"] += dt_gyr

        if s["is_m"] and s["age"] >= IRON_STAR_AGE:
            s["dead"] = True
            s["terminal"] = "iron_star"
        elif not s["is_m"] and s["age"] >= s["lifetime"]:
            s["dead"] = True
            if s["has_supernova"]:
                s["markers"].append(s["angle_accum"])

    # evaporation
    evap_r = galaxy_evap_radius(sim_time)
    if sim_time >= EVAP_START_GYR:
        for s in stars:
            if not s["evaporated"] and s["orbit_r"] > evap_r:
                s["evaporated"] = True

    # ---- DRAW ----
    rl.begin_drawing()
    rl.clear_background(BG)

    gcx, gcy = world_to_screen(GALAXY_CX, GALAXY_CY, cam_x, cam_y, zoom)

    # evap ring
    evap_rs = evap_r * zoom
    if evap_rs > 2.0:
        draw_circle_outline(gcx, gcy, evap_rs, rl.Color(80, 160, 80, 120), 1.5)
        rl.draw_text("galaxy envelope",
                     int(gcx + evap_rs * 0.707) + 4,
                     int(gcy - evap_rs * 0.707) - FSM,
                     FSM, rl.Color(80, 160, 80, 160))

    # orbit rings
    for s in stars:
        rs = s["orbit_r"] * zoom
        if rs > 1.0:
            alpha = 40 if s["evaporated"] else 70
            rl.draw_circle_lines(int(gcx), int(gcy), rs, rl.Color(60, 60, 65, alpha))

    # galaxy center
    rl.draw_circle(int(gcx), int(gcy), max(3, int(4 * zoom)),
                   rl.Color(255, 200, 100, 200))

    for s in stars:
        angle = s["angle_accum"]
        wx = GALAXY_CX + s["orbit_r"] * math.cos(angle)
        wy = GALAXY_CY + s["orbit_r"] * math.sin(angle)
        sx, sy = world_to_screen(wx, wy, cam_x, cam_y, zoom)
        px, py = int(sx), int(sy)
        r_draw = max(2.0, star_radius_px(s["mass"]) * zoom)
        ea = 80 if s["evaporated"] else 255

        if s["dead"]:
            tc = TERMINAL_COLORS[s["terminal"]]
            oc = TERMINAL_OUTLINE[s["terminal"]]
            rl.draw_circle(px, py, r_draw,
                           rl.Color(tc[0], tc[1], tc[2], ea))
            draw_circle_outline(px, py, r_draw,
                                rl.Color(oc[0], oc[1], oc[2], ea),
                                max(1.5, 2.0 * zoom))
        else:
            c = STAR_CLASSES[s["class_idx"]][1]
            rl.draw_circle(px, py, r_draw, rl.Color(c[0], c[1], c[2], ea))

        # SN markers on orbit
        for ma in s["markers"]:
            mwx = GALAXY_CX + s["orbit_r"] * math.cos(ma)
            mwy = GALAXY_CY + s["orbit_r"] * math.sin(ma)
            msx, msy = world_to_screen(mwx, mwy, cam_x, cam_y, zoom)
            rl.draw_circle(int(msx), int(msy), max(2, int(3 * zoom)),
                           rl.Color(255, 80, 30, 200))

        # SN count dots above star
        nm = len(s["markers"])
        if nm > 0:
            dot_r  = 3
            spacing = 8
            tw = (nm - 1) * spacing
            for i in range(nm):
                rl.draw_circle(px - tw // 2 + i * spacing,
                               py - int(r_draw) - dot_r - 4,
                               dot_r, rl.Color(255, 100, 40, 255))

    # ---- UI ----
    # restart button
    btn_col = rl.Color(80, 120, 80, 255) if btn_hit else rl.Color(60, 90, 60, 255)
    rl.draw_rectangle(BTN_X, BTN_Y, BTN_W, BTN_H, btn_col)
    rl.draw_text("RESTART", BTN_X + 18, BTN_Y + 7, FSM, rl.Color(200, 230, 200, 255))

    # slider
    rl.draw_rectangle(SL_X - 4, SL_Y - 4, SL_W + 8, SL_H + 8,
                      rl.Color(50, 50, 55, 230))
    rl.draw_rectangle(SL_X, SL_Y, SL_W, SL_H, rl.Color(80, 80, 90, 255))
    lmin = math.log10(TIME_MIN)
    lmax = math.log10(TIME_MAX)
    t_cur = max(0.0, min(1.0, (math.log10(time_speed) - lmin) / (lmax - lmin)))
    hx = int(SL_X + t_cur * SL_W)
    rl.draw_rectangle(hx - 5, SL_Y - 4, 10, SL_H + 8, rl.Color(210, 210, 230, 255))
    rl.draw_text("Time speed:", SL_X, SL_Y - FSM - 4, FSM, rl.Color(180, 180, 180, 255))
    ts_str = f"{time_speed:.2e} Gyr/s" if time_speed >= 1.0 else f"{time_speed:.4f} Gyr/s"
    rl.draw_text(ts_str, SL_X + SL_W + 14, SL_Y + 2, FSM, rl.Color(200, 200, 200, 255))
    rl.draw_text("Scroll=zoom  RMB/MMB=pan",
                 SL_X, SL_Y - FSM * 2 - 8, FSM, rl.Color(120, 120, 130, 255))

    # stats
    rl.draw_text(f"Sim time: {format_time(sim_time)}", 20, 16, FLG,
                 rl.Color(160, 160, 160, 255))
    alive    = sum(1 for s in stars if not s["dead"])
    total_sn = sum(len(s["markers"]) for s in stars)
    rl.draw_text(f"Stars alive: {alive}/{STAR_COUNT}", 20, 16 + FLG + 4, FMD,
                 rl.Color(160, 160, 160, 255))
    rl.draw_text(f"Supernovae: {total_sn}", 20, 16 + FLG + 4 + FMD + 4, FMD,
                 rl.Color(255, 120, 60, 255))
    evap_pct = 0.0
    if sim_time >= EVAP_START_GYR:
        evap_pct = min(1.0, (sim_time - EVAP_START_GYR) / (EVAP_END_GYR - EVAP_START_GYR))
    rl.draw_text(f"Galaxy evap: {evap_pct*100:.1f}%",
                 20, 16 + FLG + 4 + (FMD + 4) * 2, FMD,
                 rl.Color(80, 200, 80, 255))

    # legend
    lx = SCREEN_W - 240
    ly = 16
    rl.draw_text("STAR CLASSES", lx, ly, FMD, rl.Color(200, 200, 200, 255))
    ly += FMD + 4
    for sc in STAR_CLASSES:
        rl.draw_circle(lx + 8, ly + FSM // 2, 6, make_color(sc[1]))
        rl.draw_text(f"{sc[0]}  {sc[2]:.3g} Gyr", lx + 20, ly, FSM,
                     rl.Color(155, 155, 155, 255))
        ly += FSM + 3
    ly += 6
    rl.draw_text("TERMINAL STATES", lx, ly, FMD, rl.Color(200, 200, 200, 255))
    ly += FMD + 4
    for key, col in TERMINAL_COLORS.items():
        oc = TERMINAL_OUTLINE[key]
        rl.draw_circle(lx + 8, ly + FSM // 2, 6, make_color(col))
        draw_circle_outline(lx + 8, ly + FSM // 2, 6, make_color(oc), 2.0)
        rl.draw_text(key.replace("_", " "), lx + 20, ly, FSM,
                     rl.Color(155, 155, 155, 255))
        ly += FSM + 3

    rl.end_drawing()

rl.close_window()
