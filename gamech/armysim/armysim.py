"""
army battle simulator example
"""
# main.py
import pyray as rl
import math
from sim import UnitTemplate, Army, SimState, compute_interaction_slots

SCREEN_W = 1920
SCREEN_H = 1080
FONT_UI   = 24
FONT_HP   = 32
PAD       = 8
PANEL_W   = 280
LOG_LINES = 12
UNIT_CELL = 48

COL_LEFT_BG   = rl.Color(220, 230, 255, 255)
COL_RIGHT_BG  = rl.Color(255, 220, 220, 255)
COL_LEFT      = rl.Color(30,  80,  200, 255)
COL_RIGHT     = rl.Color(200, 30,  30,  255)
COL_ACTIVE_BG = rl.Color(240, 240, 200, 255)
COL_DEAD      = rl.Color(180, 180, 180, 80)
COL_BTN       = rl.Color(90,  90,  90,  255)
COL_BTN_HOV   = rl.Color(60,  60,  60,  255)


def make_default_left() -> UnitTemplate:
    return UnitTemplate(hp=100.0, attack=20.0, defense=5.0, regen=2.0,
                        attack_speed=3, regen_speed=4, count=20,
                        hit_chance=0.85, splash_damage=0.0, splash_count=0, unit_size=1)

def make_default_right() -> UnitTemplate:
    return UnitTemplate(hp=80.0, attack=15.0, defense=8.0, regen=3.0,
                        attack_speed=4, regen_speed=3, count=20,
                        hit_chance=0.75, splash_damage=10.0, splash_count=2, unit_size=2)


class TextField:
    def __init__(self, label, x, y, w=90, is_float=False):
        self.label    = label
        self.x        = x
        self.y        = y
        self.w        = w
        self.h        = FONT_UI + 4
        self.is_float = is_float
        self.active   = False
        self.buf      = ""
        self._value   = 0.0

    def set_value(self, v):
        self._value = float(v)
        self.buf = f"{v:.2f}" if self.is_float else str(int(v))

    def get_value(self):
        return self._value

    def _try_commit(self):
        try:
            self._value = float(self.buf) if self.is_float else float(int(float(self.buf)))
        except Exception:
            pass

    def update(self, font):
        mouse   = rl.get_mouse_position()
        rect    = rl.Rectangle(self.x, self.y, self.w, self.h)
        clicked = rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)
        if clicked:
            was         = self.active
            self.active = rl.check_collision_point_rec(mouse, rect)
            if self.active and not was:
                self.buf = f"{self._value:.2f}" if self.is_float else str(int(self._value))
            elif was and not self.active:
                self._try_commit()
        if self.active:
            ch = rl.get_char_pressed()
            while ch > 0:
                c = chr(ch)
                if (c.isdigit()
                        or (c == '.' and self.is_float and '.' not in self.buf)
                        or (c == '-' and len(self.buf) == 0)):
                    self.buf += c
                    self._try_commit()
                ch = rl.get_char_pressed()
            if rl.is_key_pressed(rl.KeyboardKey.KEY_BACKSPACE):
                self.buf = self.buf[:-1]
                self._try_commit()
            if rl.is_key_pressed(rl.KeyboardKey.KEY_ENTER):
                self._try_commit()
                self.active = False
        lw = rl.measure_text_ex(font, self.label + ":", FONT_UI, 1).x
        rl.draw_text_ex(font, self.label + ":",
                        rl.Vector2(self.x - lw - 4, self.y + 2), FONT_UI, 1, rl.BLACK)
        rl.draw_rectangle_rec(rect, rl.RAYWHITE)
        rl.draw_rectangle_lines_ex(rect, 1, rl.BLUE if self.active else rl.DARKGRAY)
        display = self.buf if self.active else (
            f"{self._value:.2f}" if self.is_float else str(int(self._value)))
        rl.draw_text_ex(font, display, rl.Vector2(self.x + 3, self.y + 2), FONT_UI, 1, rl.BLACK)
        if self.active and int(rl.get_time() * 2) % 2 == 0:
            tw = rl.measure_text_ex(font, display, FONT_UI, 1).x
            rl.draw_rectangle(int(self.x + 3 + tw), int(self.y + 2), 2, FONT_UI, rl.BLACK)


FIELD_DEFS = [
    ("HP",        "hp",            True),
    ("Attack",    "attack",        True),
    ("Defense",   "defense",       True),
    ("Regen",     "regen",         True),
    ("AtkSpd",    "attack_speed",  False),
    ("RgnSpd",    "regen_speed",   False),
    ("Count",     "count",         False),
    ("HitChance", "hit_chance",    True),
    ("SplashDmg", "splash_damage", True),
    ("SplashCnt", "splash_count",  False),
    ("UnitSize",  "unit_size",     False),
]
ROW_H = FONT_UI + 8

def make_panel_fields(tmpl, field_x, start_y):
    fields = []
    for i, (label, attr, is_float) in enumerate(FIELD_DEFS):
        f = TextField(label, field_x, start_y + i * ROW_H, 90, is_float)
        f.set_value(getattr(tmpl, attr))
        fields.append((attr, f))
    return fields

def apply_fields_to_template(fields, tmpl):
    for attr, f in fields:
        v = f.get_value()
        if attr in ("attack_speed", "regen_speed", "count", "splash_count", "unit_size"):
            v = max(1, int(v))
        elif attr == "hit_chance":
            v = max(0.0, min(1.0, v))
        else:
            v = max(0.0, v)
        setattr(tmpl, attr, v)

def sync_fields_from_template(fields, tmpl):
    for attr, f in fields:
        f.set_value(getattr(tmpl, attr))


def draw_button(font, text, x, y, w, h):
    mouse   = rl.get_mouse_position()
    rect    = rl.Rectangle(x, y, w, h)
    hovered = rl.check_collision_point_rec(mouse, rect)
    rl.draw_rectangle_rec(rect, COL_BTN_HOV if hovered else COL_BTN)
    tw = rl.measure_text_ex(font, text, FONT_UI, 1)
    rl.draw_text_ex(font, text,
                    rl.Vector2(x + (w - tw.x) / 2, y + (h - tw.y) / 2),
                    FONT_UI, 1, rl.WHITE)
    return hovered and rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)


def draw_stepper(font, label, val_ref, x, y, step=1, min_val=1):
    rl.draw_text_ex(font, label, rl.Vector2(x, y + 2), FONT_UI, 1, rl.BLACK)
    lw      = int(rl.measure_text_ex(font, label, FONT_UI, 1).x)
    bx      = x + lw + 4
    bw, bh  = 24, FONT_UI + 4
    mouse   = rl.get_mouse_position()
    clicked = rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)
    minus   = rl.Rectangle(bx, y, bw, bh)
    plus    = rl.Rectangle(bx + bw + 36, y, bw, bh)
    rl.draw_rectangle_rec(minus, COL_BTN)
    rl.draw_text_ex(font, "-", rl.Vector2(bx + 5, y + 2), FONT_UI, 1, rl.WHITE)
    rl.draw_text_ex(font, str(val_ref[0]), rl.Vector2(bx + bw + 6, y + 2), FONT_UI, 1, rl.BLACK)
    rl.draw_rectangle_rec(plus, COL_BTN)
    rl.draw_text_ex(font, "+", rl.Vector2(bx + bw + 36 + 5, y + 2), FONT_UI, 1, rl.WHITE)
    if clicked:
        if rl.check_collision_point_rec(mouse, minus):
            val_ref[0] = max(min_val, val_ref[0] - step)
        if rl.check_collision_point_rec(mouse, plus):
            val_ref[0] += step


def draw_unit_grid(font, units_hp, active_set, x, y, w, h, base_color, is_active_col):
    indices = [i for i, hp in enumerate(units_hp) if (i in active_set) == is_active_col]
    total   = len(indices)
    if total == 0:
        return
    cols    = max(1, w // UNIT_CELL)
    rows    = math.ceil(total / cols)
    cell_w  = w // cols
    cell_h  = min(UNIT_CELL, h // max(rows, 1))
    for slot, idx in enumerate(indices):
        col = slot % cols
        row = slot // cols
        rx  = x + col * cell_w
        ry  = y + row * cell_h
        if ry + cell_h > y + h:
            break
        hp    = units_hp[idx]
        alive = hp > 0
        c     = base_color if alive else COL_DEAD
        rl.draw_rectangle(rx + 1, ry + 1, cell_w - 2, cell_h - 2, c)
        if alive:
            txt = str(int(hp))
            tw  = rl.measure_text_ex(font, txt, FONT_HP, 1).x
            rl.draw_text_ex(font, txt,
                            rl.Vector2(rx + (cell_w - tw) / 2, ry + (cell_h - FONT_HP) / 2),
                            FONT_HP, 1, rl.WHITE)


def main():
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(SCREEN_W, SCREEN_H, "Army Battle Simulator")
    rl.set_target_fps(60)

    font = rl.load_font_ex("C:/Windows/Fonts/arialbd.ttf", FONT_HP, None, 0)

    left_tmpl  = make_default_left()
    right_tmpl = make_default_right()

    interaction_area = [10]
    sim_speed        = [10]
    paused           = [False]

    L_FIELD_X = 170

    def rebuild_panel_fields():
        lf = make_panel_fields(left_tmpl,  L_FIELD_X, 44)
        rf = make_panel_fields(right_tmpl, 0, 44)
        return lf, rf

    left_fields, right_fields = rebuild_panel_fields()

    def build_sim():
        apply_fields_to_template(left_fields,  left_tmpl)
        apply_fields_to_template(right_fields, right_tmpl)
        s = SimState(left=Army(left_tmpl), right=Army(right_tmpl),
                     interaction_area=interaction_area[0])
        s.reset()
        return s

    state      = build_sim()
    tick_accum = 0.0

    while not rl.window_should_close():
        sw = rl.get_screen_width()
        sh = rl.get_screen_height()
        dt = rl.get_frame_time()

        rp = sw - PANEL_W
        for _, f in right_fields:
            f.x = rp + L_FIELD_X

        if state.running and not paused[0]:
            tick_accum += dt * sim_speed[0]
            while tick_accum >= 1.0 and state.running:
                state.step()
                tick_accum -= 1.0

        state.interaction_area = interaction_area[0]
        l_active_idx = set(compute_interaction_slots(state.left,  interaction_area[0]))
        r_active_idx = set(compute_interaction_slots(state.right, interaction_area[0]))

        bx    = PANEL_W
        bw    = sw - 2 * PANEL_W
        col_w = bw // 4
        boy   = 44
        bh    = sh // 3

        pool_l_x   = bx
        active_l_x = bx + col_w
        active_r_x = bx + col_w * 2
        pool_r_x   = bx + col_w * 3

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)

        rl.draw_rectangle(0,  0, PANEL_W, sh, COL_LEFT_BG)
        rl.draw_rectangle(rp, 0, PANEL_W, sh, COL_RIGHT_BG)
        rl.draw_rectangle(active_l_x, boy, col_w * 2, bh, COL_ACTIVE_BG)

        rl.draw_text_ex(font, "Pool",   rl.Vector2(pool_l_x   + 4, boy - FONT_UI - 4), FONT_UI, 1, rl.DARKBLUE)
        rl.draw_text_ex(font, "Active", rl.Vector2(active_l_x + 4, boy - FONT_UI - 4), FONT_UI, 1, rl.DARKBLUE)
        rl.draw_text_ex(font, "Active", rl.Vector2(active_r_x + 4, boy - FONT_UI - 4), FONT_UI, 1, rl.MAROON)
        rl.draw_text_ex(font, "Pool",   rl.Vector2(pool_r_x   + 4, boy - FONT_UI - 4), FONT_UI, 1, rl.MAROON)

        draw_unit_grid(font, state.left.units,  l_active_idx, pool_l_x,   boy, col_w, bh, COL_LEFT,  False)
        draw_unit_grid(font, state.left.units,  l_active_idx, active_l_x, boy, col_w, bh, COL_LEFT,  True)
        draw_unit_grid(font, state.right.units, r_active_idx, active_r_x, boy, col_w, bh, COL_RIGHT, True)
        draw_unit_grid(font, state.right.units, r_active_idx, pool_r_x,   boy, col_w, bh, COL_RIGHT, False)

        mid = active_l_x + col_w
        rl.draw_line(mid, boy, mid, boy + bh, rl.DARKGRAY)

        rl.draw_text_ex(font, "LEFT",  rl.Vector2(PAD, PAD), FONT_HP, 1, rl.DARKBLUE)
        rl.draw_text_ex(font, "RIGHT", rl.Vector2(rp + PAD, PAD), FONT_HP, 1, rl.MAROON)

        for _, f in left_fields:
            f.update(font)
        for _, f in right_fields:
            f.update(font)

        info_y = boy + bh + 6
        rl.draw_text_ex(font, f"Alive: {state.left.alive_count()}",
                        rl.Vector2(pool_l_x + 4, info_y), FONT_UI, 1, rl.DARKBLUE)
        rl.draw_text_ex(font, f"Alive: {state.right.alive_count()}",
                        rl.Vector2(active_r_x + 4, info_y), FONT_UI, 1, rl.MAROON)

        ctrl_y  = info_y + FONT_UI + 8
        half_bw = bw // 2
        draw_stepper(font, "Interaction Area:", interaction_area, bx + PAD, ctrl_y)
        draw_stepper(font, "Sim Speed:", sim_speed, bx + half_bw, ctrl_y)

        tick_y   = ctrl_y + FONT_UI + 12
        tick_str = f"Tick: {state.tick}"
        tw       = rl.measure_text_ex(font, tick_str, FONT_UI, 1).x
        rl.draw_text_ex(font, tick_str, rl.Vector2(bx + (bw - tw) / 2, tick_y), FONT_UI, 1, rl.BLACK)

        btn_y       = tick_y + FONT_UI + 12
        btn_h       = FONT_UI + 12
        btn_w       = 180
        total_btn_w = btn_w * 3 + PAD * 2
        btn_start   = bx + (bw - total_btn_w) // 2

        if draw_button(font, "Advance to Attack", btn_start, btn_y, btn_w, btn_h):
            if state.running:
                for _ in range(state.next_attack_tick()):
                    state.step()
                tick_accum = 0.0

        if draw_button(font, "Resume" if paused[0] else "Pause",
                       btn_start + btn_w + PAD, btn_y, btn_w, btn_h):
            paused[0] = not paused[0]

        if draw_button(font, "Reset Sim", btn_start + (btn_w + PAD) * 2, btn_y, btn_w, btn_h):
            state      = build_sim()
            tick_accum = 0.0
            paused[0]  = False

        log_y = btn_y + btn_h + 8
        log_h = sh - log_y - 4
        rl.draw_rectangle(bx, log_y, bw, log_h, rl.Color(245, 245, 245, 255))
        rl.draw_rectangle_lines(bx, log_y, bw, log_h, rl.LIGHTGRAY)
        for i, line in enumerate(state.log[-LOG_LINES:]):
            rl.draw_text_ex(font, line,
                            rl.Vector2(bx + 6, log_y + 4 + i * (FONT_UI + 2)),
                            FONT_UI, 1, rl.DARKGRAY)

        if state.winner:
            wt = f"{state.winner} Wins!"
            tw = rl.measure_text_ex(font, wt, FONT_HP + 16, 1).x
            rl.draw_text_ex(font, wt, rl.Vector2(bx + (bw - tw) / 2, sh // 2 - 40),
                            FONT_HP + 16, 1, rl.GOLD)

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
