"""
Galaxy travelling simulation
shows how different travel speeds compare with galaxy spinning rate
"""
import pyray as rl
import math
import random

SCREEN_W = 1920
SCREEN_H = 1080
FONT_SIZE = 32
RENDER_STARS = 10_000
GALAXY_RADIUS_LY = 50_000.0
NUM_ARMS = 4

C_KM_S = 299_792.0
LY_PER_YEAR = 1.0

ANDROMEDA_VISUAL_X = -80_000.0
ANDROMEDA_VISUAL_Y = 0.0
ANDROMEDA_VISUAL_RADIUS = 12_000.0
ANDROMEDA_TRAVEL_DIST_LY = 2_500_000.0

SPEED_STEPS = [
    17.0       / C_KM_S,
    70.0       / C_KM_S,
    500.0      / C_KM_S,
    3000.0     / C_KM_S,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]

AMBIENT_SIM_SPEED = 1_000_000.0  # yr/s

def color(r, g, b, a=255):
    return rl.Color(r, g, b, a)

COL_BG            = color(5, 5, 15)
COL_STAR_BLUE     = color(180, 200, 255, 200)
COL_STAR_WHITE    = color(255, 255, 220, 200)
COL_STAR_ORANGE   = color(255, 220, 180, 200)
COL_STAR_RED      = color(255, 180, 180, 200)
COL_SHIP          = color(255, 80, 80)
COL_SHIP_RING     = color(255, 80, 80, 180)
COL_TARGET        = color(100, 255, 100)
COL_HOVER         = color(255, 255, 100, 200)
COL_LINE          = color(100, 200, 255, 120)
COL_UI_TITLE      = color(200, 200, 255)
COL_UI_INFO       = color(160, 160, 200)
COL_UI_SPEED      = color(200, 220, 255)
COL_UI_DIST       = color(100, 255, 150)
COL_UI_TRAVEL     = color(255, 150, 50)
COL_UI_DIM        = color(140, 140, 180)
COL_RESULT_BG     = color(0, 0, 0, 180)
COL_RESULT_TXT    = color(100, 255, 150)
COL_ANDROMEDA     = color(255, 220, 50)
COL_ANDROMEDA_HOV = color(255, 255, 120, 220)

V_ORBIT_LY_YR = 220.0 * 3.154e7 / 9.461e12

ANDROMEDA_HIT_PX = 20

class State:
    def __init__(self):
        self.stars_x     = []
        self.stars_y     = []
        self.stars_r     = []
        self.stars_angle = []
        self.stars_omega = []

        self.ship_star   = 0
        self.target_star = -1
        self.target_andromeda = False
        self.ship_x      = 0.0
        self.ship_y      = 0.0
        self.speed_idx   = 9

        self.cam_x    = 0.0
        self.cam_y    = 0.0
        self.cam_zoom = 0.008

        self.traveling         = False
        self.travel_sim_speed  = 0.0
        self.travel_years      = 0.0
        self.travel_elapsed_yr = 0.0

        self.travel_start_x = 0.0
        self.travel_start_y = 0.0
        self.travel_end_x   = 0.0
        self.travel_end_y   = 0.0

        self.result_text  = ""
        self.result_timer = 0.0
        self.sim_time     = 0.0
        self.hover_star   = -1
        self.hover_andromeda = False

        self.drag_active = False
        self.drag_sx     = 0
        self.drag_sy     = 0
        self.drag_cx     = 0.0
        self.drag_cy     = 0.0

        self.at_andromeda = False

def speed_c(state):
    return SPEED_STEPS[state.speed_idx]

def speed_km_s(state):
    return speed_c(state) * C_KM_S

def orbital_omega(r_ly):
    if r_ly < 1.0:
        return 0.0
    v = V_ORBIT_LY_YR if r_ly >= 3000.0 else V_ORBIT_LY_YR * (r_ly / 3000.0)
    return v / r_ly

def spiral_angle(r_ly, arm_idx):
    k = 0.3
    offset = (2.0 * math.pi / NUM_ARMS) * arm_idx
    r_ly = max(r_ly, 1.0)
    return k * math.log(r_ly / 500.0 + 1.0) + offset

def generate_galaxy(state):
    random.seed(42)
    n = RENDER_STARS
    state.stars_x     = [0.0] * n
    state.stars_y     = [0.0] * n
    state.stars_r     = [0.0] * n
    state.stars_angle = [0.0] * n
    state.stars_omega = [0.0] * n

    for i in range(n):
        r = random.expovariate(1.0 / 15000.0)
        r = max(100.0, min(r, GALAXY_RADIUS_LY))
        arm   = random.randint(0, NUM_ARMS - 1)
        angle = spiral_angle(r, arm) + random.gauss(0, 0.25)

        state.stars_r[i]     = r
        state.stars_angle[i] = angle
        state.stars_x[i]     = r * math.cos(angle)
        state.stars_y[i]     = r * math.sin(angle)
        state.stars_omega[i] = orbital_omega(r)

    state.ship_star = random.randint(0, n - 1)
    state.ship_x    = state.stars_x[state.ship_star]
    state.ship_y    = state.stars_y[state.ship_star]

def world_to_screen(wx, wy, state):
    sx = int((wx - state.cam_x) * state.cam_zoom + SCREEN_W / 2)
    sy = int((wy - state.cam_y) * state.cam_zoom + SCREEN_H / 2)
    return sx, sy

def screen_to_world(sx, sy, state):
    wx = (sx - SCREEN_W / 2) / state.cam_zoom + state.cam_x
    wy = (sy - SCREEN_H / 2) / state.cam_zoom + state.cam_y
    return wx, wy

def find_nearest_star(wx, wy, state, radius_world):
    best    = -1
    best_d2 = radius_world * radius_world
    for i in range(RENDER_STARS):
        dx  = state.stars_x[i] - wx
        dy  = state.stars_y[i] - wy
        d2  = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best    = i
    return best

def andromeda_screen_pos(state):
    return world_to_screen(ANDROMEDA_VISUAL_X, ANDROMEDA_VISUAL_Y, state)

def andromeda_screen_radius(state):
    return max(6, int(ANDROMEDA_VISUAL_RADIUS * state.cam_zoom))

def is_hover_andromeda(mx, my, state):
    ax, ay = andromeda_screen_pos(state)
    ar = andromeda_screen_radius(state)
    hit = max(ar, ANDROMEDA_HIT_PX)
    dx = mx - ax
    dy = my - ay
    return (dx * dx + dy * dy) < hit * hit

def star_color(i):
    t = ((i * 2654435761) & 0xFFFF) / 65535.0
    if t < 0.3:   return COL_STAR_BLUE
    elif t < 0.6: return COL_STAR_WHITE
    elif t < 0.8: return COL_STAR_ORANGE
    else:         return COL_STAR_RED

def rotate_galaxy(state, dt_years):
    for i in range(RENDER_STARS):
        state.stars_angle[i] += state.stars_omega[i] * dt_years
        r = state.stars_r[i]
        state.stars_x[i] = r * math.cos(state.stars_angle[i])
        state.stars_y[i] = r * math.sin(state.stars_angle[i])

def format_years(yr):
    if yr >= 1_000_000_000:
        return f"{yr/1e9:.3f} billion years"
    elif yr >= 1_000_000:
        return f"{yr/1e6:.3f} million years"
    elif yr >= 1_000:
        return f"{yr/1e3:.2f} thousand years"
    return f"{yr:.2f} years"

def format_speed(state):
    sc  = speed_c(state)
    kms = speed_km_s(state)
    if sc >= 0.01:
        return f"{sc:.2f}c  ({kms:,.0f} km/s)"
    elif kms >= 1.0:
        return f"{kms:.1f} km/s  ({sc:.2e}c)"
    else:
        return f"{kms*1000:.1f} m/s  ({sc:.2e}c)"

def travel_dist(state):
    # Andromeda <-> Milky Way always uses the real intergalactic distance
    if state.target_andromeda:
        return ANDROMEDA_TRAVEL_DIST_LY
    if state.at_andromeda and state.target_star >= 0:
        return ANDROMEDA_TRAVEL_DIST_LY
    if state.target_star < 0:
        return 0.0
    dx = state.stars_x[state.target_star] - state.ship_x
    dy = state.stars_y[state.target_star] - state.ship_y
    return math.sqrt(dx * dx + dy * dy)

def main():
    rl.init_window(SCREEN_W, SCREEN_H, "Galaxy Scale Simulation")
    rl.set_target_fps(60)

    state = State()
    generate_galaxy(state)
    state.cam_x = state.ship_x
    state.cam_y = state.ship_y

    while not rl.window_should_close():
        dt = rl.get_frame_time()
        mx = rl.get_mouse_x()
        my = rl.get_mouse_y()
        wx, wy = screen_to_world(mx, my, state)

        # zoom
        wheel = rl.get_mouse_wheel_move()
        if wheel != 0:
            state.cam_zoom *= 1.1 ** wheel
            state.cam_zoom = max(0.00005, min(state.cam_zoom, 5.0))

        # pan
        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_MIDDLE):
            state.drag_active = True
            state.drag_sx     = mx
            state.drag_sy     = my
            state.drag_cx     = state.cam_x
            state.drag_cy     = state.cam_y
        if rl.is_mouse_button_released(rl.MOUSE_BUTTON_MIDDLE):
            state.drag_active = False
        if state.drag_active:
            state.cam_x = state.drag_cx - (mx - state.drag_sx) / state.cam_zoom
            state.cam_y = state.drag_cy - (my - state.drag_sy) / state.cam_zoom

        # hover
        state.hover_andromeda = is_hover_andromeda(mx, my, state) and not state.traveling
        if state.hover_andromeda:
            state.hover_star = -1
        else:
            snap = max(500.0, min(3000.0 / state.cam_zoom, 20000.0))
            state.hover_star = find_nearest_star(wx, wy, state, snap)

        # select target
        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT) and not state.traveling:
            if state.hover_andromeda and not state.at_andromeda:
                state.target_andromeda = True
                state.target_star      = -1
            elif state.hover_star >= 0 and state.hover_star != state.ship_star:
                state.target_star      = state.hover_star
                state.target_andromeda = False

        # speed adjust
        if not state.traveling:
            if rl.is_key_pressed(rl.KEY_UP):
                state.speed_idx = min(len(SPEED_STEPS) - 1, state.speed_idx + 1)
            if rl.is_key_pressed(rl.KEY_DOWN):
                state.speed_idx = max(0, state.speed_idx - 1)

        # start/stop travel
        if rl.is_key_pressed(rl.KEY_SPACE):
            if state.traveling:
                state.traveling        = False
                state.target_star      = -1
                state.target_andromeda = False
                state.result_text  = "Travel aborted after " + format_years(state.travel_elapsed_yr)
                state.result_timer = 8.0
            elif state.target_andromeda or state.target_star >= 0:
                d  = travel_dist(state)
                sc = speed_c(state)
                state.travel_years      = d / sc
                state.travel_elapsed_yr = 0.0
                state.traveling         = True
                state.result_text       = ""
                state.travel_sim_speed  = max(state.travel_years / 5.0, AMBIENT_SIM_SPEED)

                state.travel_start_x = state.ship_x
                state.travel_start_y = state.ship_y
                if state.target_andromeda:
                    state.travel_end_x = ANDROMEDA_VISUAL_X
                    state.travel_end_y = ANDROMEDA_VISUAL_Y
                else:
                    state.travel_end_x = state.stars_x[state.target_star]
                    state.travel_end_y = state.stars_y[state.target_star]

        # --- update ---
        if state.traveling:
            dt_years = state.travel_sim_speed * dt
            state.sim_time          += dt_years
            state.travel_elapsed_yr += dt_years

            frac = min(state.travel_elapsed_yr / state.travel_years, 1.0) if state.travel_years > 0 else 1.0
            state.ship_x = state.travel_start_x + (state.travel_end_x - state.travel_start_x) * frac
            state.ship_y = state.travel_start_y + (state.travel_end_y - state.travel_start_y) * frac

            rotate_galaxy(state, dt_years)

            if frac >= 1.0:
                state.ship_x    = state.travel_end_x
                state.ship_y    = state.travel_end_y
                state.traveling = False
                if state.target_andromeda:
                    state.at_andromeda     = True
                    state.target_andromeda = False
                    state.ship_star        = -1
                else:
                    state.at_andromeda = False
                    state.ship_star    = state.target_star
                    state.target_star  = -1
                state.result_text  = "Travel took " + format_years(state.travel_elapsed_yr)
                state.result_timer = 8.0
        else:
            dt_years = AMBIENT_SIM_SPEED * dt
            state.sim_time += dt_years
            rotate_galaxy(state, dt_years)

        if state.result_timer > 0:
            state.result_timer -= dt

        # --- draw ---
        rl.begin_drawing()
        rl.clear_background(COL_BG)

        # stars
        for i in range(RENDER_STARS):
            sx, sy = world_to_screen(state.stars_x[i], state.stars_y[i], state)
            if 0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H:
                rl.draw_pixel(sx, sy, star_color(i))

        # andromeda
        ax, ay = andromeda_screen_pos(state)
        ar = andromeda_screen_radius(state)
        rl.draw_circle_lines(ax, ay, ar, COL_ANDROMEDA)
        rl.draw_text("Andromeda", ax - rl.measure_text("Andromeda", 20) // 2, ay + ar + 4, 20, COL_ANDROMEDA)

        if state.hover_andromeda and not state.at_andromeda:
            rl.draw_circle_lines(ax, ay, ar + 6, COL_ANDROMEDA_HOV)

        if state.target_andromeda:
            rl.draw_circle_lines(ax, ay, ar + 10, COL_TARGET)

        # travel line
        if state.traveling or state.target_star >= 0 or state.target_andromeda:
            if state.traveling:
                ex, ey = world_to_screen(state.travel_end_x, state.travel_end_y, state)
            elif state.target_andromeda:
                ex, ey = ax, ay
            else:
                ex, ey = world_to_screen(
                    state.stars_x[state.target_star],
                    state.stars_y[state.target_star], state)
            sx1, sy1 = world_to_screen(state.ship_x, state.ship_y, state)
            rl.draw_line(sx1, sy1, ex, ey, COL_LINE)

        # hover star
        if state.hover_star >= 0 and not state.traveling:
            hx, hy = world_to_screen(
                state.stars_x[state.hover_star],
                state.stars_y[state.hover_star], state)
            rl.draw_circle_lines(hx, hy, 6, COL_HOVER)

        # target star
        if state.target_star >= 0:
            tx2, ty2 = world_to_screen(
                state.stars_x[state.target_star],
                state.stars_y[state.target_star], state)
            rl.draw_circle_lines(tx2, ty2, 8, COL_TARGET)

        # ship
        ssx, ssy = world_to_screen(state.ship_x, state.ship_y, state)
        rl.draw_circle(ssx, ssy, 5, COL_SHIP)
        rl.draw_circle_lines(ssx, ssy, 10, COL_SHIP_RING)

        # UI
        pad = 16
        y   = pad
        rl.draw_text("GALAXY SCALE SIMULATION", pad, y, FONT_SIZE, COL_UI_TITLE)
        y += FONT_SIZE + 6
        rl.draw_text(
            f"Real stars: ~300 billion  |  Rendered: {RENDER_STARS:,}",
            pad, y, 22, COL_UI_INFO)
        y += 30

        rl.draw_text(
            f"Speed: {format_speed(state)}   [UP/DOWN]",
            pad, y, 26, COL_UI_SPEED)
        y += 34

        has_target = state.target_star >= 0 or state.target_andromeda
        if has_target and not state.traveling:
            d  = travel_dist(state)
            yr = d / speed_c(state)
            dest = "Andromeda Galaxy" if state.target_andromeda else "target star"
            rl.draw_text(
                f"To {dest}: {d:,.0f} ly  |  Trip: {format_years(yr)}",
                pad, y, 24, COL_UI_DIST)
            y += 30
            rl.draw_text("SPACE to travel", pad, y, 26, color(255, 220, 80))
        elif not state.traveling:
            loc = "Andromeda Galaxy" if state.at_andromeda else "Milky Way"
            rl.draw_text(f"Location: {loc}  |  Click target, SPACE to travel", pad, y, 24, COL_UI_DIM)

        if state.traveling:
            dest_name = "Andromeda" if state.target_andromeda else "target"
            frac_disp = min(state.travel_elapsed_yr / state.travel_years, 1.0) if state.travel_years > 0 else 1.0
            remaining_ly = (1.0 - frac_disp) * travel_dist(state)
            rl.draw_text(
                f"TRAVELING to {dest_name}  |  Elapsed: {format_years(state.travel_elapsed_yr)}  |  Remaining: {remaining_ly:,.0f} ly",
                pad, y, 24, COL_UI_TRAVEL)
            y += 30
            rl.draw_text(
                f"Total trip: {format_years(state.travel_years)}",
                pad, y, 22, COL_UI_INFO)

        # result banner
        if state.result_timer > 0 and state.result_text:
            tw = rl.measure_text(state.result_text, FONT_SIZE + 8)
            rx = (SCREEN_W - tw) // 2
            ry = SCREEN_H // 2 - 40
            rl.draw_rectangle(rx - 20, ry - 10, tw + 40, FONT_SIZE + 28, COL_RESULT_BG)
            rl.draw_text(state.result_text, rx, ry, FONT_SIZE + 8, COL_RESULT_TXT)

        # bottom
        rl.draw_text(
            f"Sim time: {state.sim_time/1e9:.4f} billion years",
            pad, SCREEN_H - 36, 22, COL_UI_DIM)
        hint = "Scroll=zoom  |  Middle-drag=pan  |  Click=target  |  SPACE=travel"
        hw = rl.measure_text(hint, 20)
        rl.draw_text(hint, SCREEN_W - hw - pad, SCREEN_H - 28, 20, COL_UI_DIM)

        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()
