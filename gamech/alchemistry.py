"""
Chemical mixing mechanic to make alchemy more interesting
"""
import pyray as rl
import math

# --- Data ---
FREQ_A = 3.2
FREQ_B = 3.7
PERFECT_A = 18
PERFECT_B = 17
MIN_RESONANCE_X = 8.0
SCREEN_W = 1920
SCREEN_H = 1080
PAD = 30

FS_TITLE = 32
FS_BODY  = 32
FS_SMALL = 32

MAX_A = 50
MAX_B = 50
MAX_RESONANCE_DISPLAY = 15.0

# ratio tolerance: count_a should be ~2x count_b for AAB
AAB_RATIO_MIN = 1.5
AAB_RATIO_MAX = 2.5

def calculate_resonance(freq_a, count_a, freq_b, count_b):
    if count_a == 0 or count_b == 0:
        return 0.0
    freq_harmony = 1.0 / (1.0 + abs(freq_a - freq_b))
    count_balance = min(count_a, count_b) / max(count_a, count_b)
    total_strength = math.sqrt(count_a * count_b)
    return freq_harmony * count_balance * total_strength

def process_reaction(count_a, count_b):
    resonance = calculate_resonance(FREQ_A, count_a, FREQ_B, count_b)
    result = {
        'X': 0, 'AAB': 0,
        'leftover_A': count_a, 'leftover_B': count_b,
        'resonance': resonance,
        'mode': "NORMAL",
        'why': ""
    }

    # PERFECT path
    if count_a == PERFECT_A and count_b == PERFECT_B and resonance >= MIN_RESONANCE_X:
        result['X'] = 1
        result['leftover_A'] = 0
        result['leftover_B'] = 0
        result['mode'] = "PERFECT"
        result['why'] = (
            f"Exact ratio A={PERFECT_A} B={PERFECT_B} met.\n"
            f"Resonance {resonance:.2f} >= {MIN_RESONANCE_X}.\n"
            "All consumed into X."
        )
        return result

    # Check ratio for AAB formation
    ratio = count_a / count_b if count_b > 0 else 0.0
    ratio_ok = AAB_RATIO_MIN <= ratio <= AAB_RATIO_MAX

    if not ratio_ok:
        result['mode'] = "NO_REACTION"
        result['why'] = (
            f"Ratio A/B = {ratio:.2f}.\n"
            f"Need ratio between {AAB_RATIO_MIN} and {AAB_RATIO_MAX}\n"
            "for AAB formation. Nothing reacts."
        )
        return result

    # How many AAB can form given 2A + 1B -> AAB
    aab_possible = min(count_a // 2, count_b)

    # AAB_CHAIN path
    if aab_possible >= 8 and resonance >= MIN_RESONANCE_X * 1.5:
        x_from_aab = aab_possible // 8
        aab_left   = aab_possible - x_from_aab * 8
        used_a = aab_possible * 2
        used_b = aab_possible
        result['X']          = x_from_aab
        result['AAB']        = aab_left
        result['leftover_A'] = count_a - used_a
        result['leftover_B'] = count_b - used_b
        result['mode']       = "AAB_CHAIN"
        result['why'] = (
            f"Ratio {ratio:.2f} ok. {aab_possible} AAB formed.\n"
            f"Resonance {resonance:.2f} >= {MIN_RESONANCE_X*1.5:.1f}.\n"
            f"Every 8 AAB collapse into X: {x_from_aab} X made.\n"
            f"{aab_left} AAB remain."
        )
        return result

    # NORMAL AAB
    used_a = aab_possible * 2
    used_b = aab_possible
    result['AAB']        = aab_possible
    result['leftover_A'] = count_a - used_a
    result['leftover_B'] = count_b - used_b
    result['mode']       = "NORMAL"
    result['why'] = (
        f"Ratio A/B = {ratio:.2f} (ok).\n"
        f"2A + B -> AAB: {aab_possible} formed.\n"
        f"Resonance {resonance:.2f} — "
        + ("too low for chain." if resonance < MIN_RESONANCE_X * 1.5 else "chain needs >=8 AAB.")
    )
    return result

# --- Layout constants ---
LEFT_W  = 1280
RIGHT_X = LEFT_W + PAD * 2
RIGHT_W = SCREEN_W - RIGHT_X - PAD

dragging = None
count_a  = 18
count_b  = 17

SLIDER_W = 700
SLIDER_H = 12
KNOB_R   = 18

SL_A_X = PAD + 220
SL_A_Y = 160
SL_B_X = PAD + 220
SL_B_Y = 280

def slider_value_to_x(val, max_val, sx, sw):
    return sx + int((val / max_val) * sw)

def slider_x_to_value(mx, sx, sw, max_val):
    t = (mx - sx) / sw
    t = max(0.0, min(1.0, t))
    return int(round(t * max_val))

def draw_label(text, x, y, size, color):
    rl.draw_text(text, x, y, size, color)

def draw_bar(x, y, w, h, value, max_val, color):
    rl.draw_rectangle_lines(x, y, w, h, rl.DARKGRAY)
    fill = int(w * max(0.0, min(1.0, value / max(max_val, 1))))
    if fill > 0:
        rl.draw_rectangle(x, y, fill, h, color)

def draw_slider(label, sx, sy, val, max_val, knob_color):
    draw_label(label, sx, sy - FS_BODY - 6, FS_BODY, rl.BLACK)
    rl.draw_rectangle(sx, sy - SLIDER_H // 2, SLIDER_W, SLIDER_H, rl.LIGHTGRAY)
    rl.draw_rectangle_lines(sx, sy - SLIDER_H // 2, SLIDER_W, SLIDER_H, rl.DARKGRAY)
    kx = slider_value_to_x(val, max_val, sx, SLIDER_W)
    rl.draw_rectangle(sx, sy - SLIDER_H // 2, kx - sx, SLIDER_H, knob_color)
    rl.draw_circle(kx, sy, KNOB_R, knob_color)
    rl.draw_circle_lines(kx, sy, KNOB_R, rl.DARKGRAY)
    vstr = str(val)
    vw = rl.measure_text(vstr, FS_BODY)
    draw_label(vstr, kx - vw // 2, sy + KNOB_R + 6, FS_BODY, rl.BLACK)

def draw_wrapped(text, x, y, size, color, max_w):
    """Draw text with \n line breaks."""
    line_h = size + 8
    for line in text.split('\n'):
        draw_label(line, x, y, size, color)
        y += line_h
    return y

def draw_info_panel(result, rx, ry, rw):
    mode = result['mode']
    mode_colors = {
        "PERFECT":    rl.GOLD,
        "AAB_CHAIN":  rl.ORANGE,
        "NORMAL":     rl.SKYBLUE,
        "NO_REACTION":rl.LIGHTGRAY,
    }
    mode_labels = {
        "PERFECT":    "PERFECT RESONANCE",
        "AAB_CHAIN":  "AAB CHAIN REACTION",
        "NORMAL":     "Normal Reaction",
        "NO_REACTION":"No Reaction",
    }
    mc = mode_colors[mode]
    ml = mode_labels[mode]

    # Panel background
    rl.draw_rectangle(rx - PAD, ry - PAD, rw + PAD, SCREEN_H - ry, rl.Color(240, 240, 240, 255))
    rl.draw_rectangle_lines(rx - PAD, ry - PAD, rw + PAD, SCREEN_H - ry, rl.DARKGRAY)

    y = ry
    draw_label("What is happening", rx, y, FS_BODY, rl.DARKGRAY)
    y += FS_BODY + 12

    # Mode badge
    mw = rl.measure_text(ml, FS_BODY) + 20
    rl.draw_rectangle(rx, y, mw, FS_BODY + 14, mc)
    draw_label(ml, rx + 10, y + 7, FS_BODY, rl.WHITE if mode != "NO_REACTION" else rl.DARKGRAY)
    y += FS_BODY + 30

    # Why text
    y = draw_wrapped(result['why'], rx, y, FS_SMALL, rl.BLACK, rw)
    y += 20

    rl.draw_line(rx, y, rx + rw, y, rl.DARKGRAY)
    y += 16

    # Resonance info
    res = result['resonance']
    draw_label(f"Resonance: {res:.3f}", rx, y, FS_SMALL, rl.BLACK)
    y += FS_SMALL + 8
    draw_label(f"Threshold (X):       {MIN_RESONANCE_X}", rx, y, FS_SMALL,
               rl.RED if res < MIN_RESONANCE_X else rl.DARKGREEN)
    y += FS_SMALL + 8
    draw_label(f"Threshold (chain):   {MIN_RESONANCE_X*1.5:.1f}", rx, y, FS_SMALL,
               rl.RED if res < MIN_RESONANCE_X * 1.5 else rl.DARKGREEN)
    y += FS_SMALL + 20

    rl.draw_line(rx, y, rx + rw, y, rl.DARKGRAY)
    y += 16

    # Ratio info
    ratio = count_a / count_b if count_b > 0 else 0.0
    ratio_color = rl.DARKGREEN if AAB_RATIO_MIN <= ratio <= AAB_RATIO_MAX else rl.RED
    draw_label(f"A/B ratio: {ratio:.2f}", rx, y, FS_SMALL, ratio_color)
    y += FS_SMALL + 8
    draw_label(f"Valid range: {AAB_RATIO_MIN} - {AAB_RATIO_MAX}", rx, y, FS_SMALL, rl.DARKGRAY)
    y += FS_SMALL + 20

    rl.draw_line(rx, y, rx + rw, y, rl.DARKGRAY)
    y += 16

    # Reaction formula reminder
    draw_label("Reactions:", rx, y, FS_SMALL, rl.DARKGRAY)
    y += FS_SMALL + 8
    draw_label("2A + B  ->  AAB", rx, y, FS_SMALL, rl.BLACK)
    y += FS_SMALL + 8
    draw_label("8 AAB + resonance  ->  X", rx, y, FS_SMALL, rl.BLACK)
    y += FS_SMALL + 8
    draw_label(f"A={PERFECT_A} B={PERFECT_B} + resonance  ->  X", rx, y, FS_SMALL, rl.BLACK)

rl.init_window(SCREEN_W, SCREEN_H, "Al-Chemistry Demo")
rl.set_target_fps(60)

PURPLE = rl.PURPLE
LBROWN = rl.Color(180, 120, 60, 255)

while not rl.window_should_close():
    mouse = rl.get_mouse_position()
    mx, my = int(mouse.x), int(mouse.y)
    lmb         = rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT)
    lmb_pressed = rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)

    if lmb_pressed:
        kax = slider_value_to_x(count_a, MAX_A, SL_A_X, SLIDER_W)
        kbx = slider_value_to_x(count_b, MAX_B, SL_B_X, SLIDER_W)
        if math.hypot(mx - kax, my - SL_A_Y) <= KNOB_R + 4:
            dragging = 'a'
        elif math.hypot(mx - kbx, my - SL_B_Y) <= KNOB_R + 4:
            dragging = 'b'
        elif SL_A_X <= mx <= SL_A_X + SLIDER_W and abs(my - SL_A_Y) <= KNOB_R + 4:
            dragging = 'a'
        elif SL_B_X <= mx <= SL_B_X + SLIDER_W and abs(my - SL_B_Y) <= KNOB_R + 4:
            dragging = 'b'

    if not lmb:
        dragging = None

    if dragging == 'a':
        count_a = slider_x_to_value(mx, SL_A_X, SLIDER_W, MAX_A)
    elif dragging == 'b':
        count_b = slider_x_to_value(mx, SL_B_X, SLIDER_W, MAX_B)

    result = process_reaction(count_a, count_b)

    rl.begin_drawing()
    rl.clear_background(rl.RAYWHITE)

    # --- Left panel ---
    draw_label("Al-Chemistry Reaction Demo", PAD, PAD, FS_TITLE, rl.BLACK)

    draw_slider(f"Count A  (freq={FREQ_A})", SL_A_X, SL_A_Y, count_a, MAX_A, rl.BLUE)
    draw_slider(f"Count B  (freq={FREQ_B})", SL_B_X, SL_B_Y, count_b, MAX_B, rl.GREEN)

    rl.draw_line(PAD, 340, LEFT_W, 340, rl.DARKGRAY)

    res = result['resonance']
    draw_label(f"Resonance: {res:.3f}", PAD, 360, FS_BODY, rl.BLACK)
    bar_x = PAD + 380
    bar_y = 364
    bar_w = 820
    bar_h = 34
    res_color = rl.GOLD if res >= MIN_RESONANCE_X else rl.BLUE
    draw_bar(bar_x, bar_y, bar_w, bar_h, res, MAX_RESONANCE_DISPLAY, res_color)
    thresh_x = bar_x + int(bar_w * MIN_RESONANCE_X / MAX_RESONANCE_DISPLAY)
    rl.draw_line(thresh_x, bar_y - 6, thresh_x, bar_y + bar_h + 6, rl.RED)
    draw_label(f"min={MIN_RESONANCE_X}", thresh_x - 30, bar_y + bar_h + 8, FS_SMALL, rl.RED)

    rl.draw_line(PAD, 510, LEFT_W, 510, rl.DARKGRAY)
    draw_label("Results", PAD, 525, FS_BODY, rl.DARKGRAY)

    items = [
        (f"X (rare):        {result['X']}",         PURPLE,  result['X'],             10),
        (f"AAB compounds:   {result['AAB']}",        LBROWN,  result['AAB'],           30),
        (f"Leftover A:      {result['leftover_A']}",  rl.BLUE, result['leftover_A'],   max(count_a, 1)),
        (f"Leftover B:      {result['leftover_B']}",  rl.GREEN,result['leftover_B'],   max(count_b, 1)),
    ]
    for i, (label, color, val, max_v) in enumerate(items):
        y = 570 + i * 100
        draw_label(label, PAD, y, FS_BODY, rl.BLACK)
        draw_bar(PAD + 500, y + 4, 700, 36, val, max(max_v, 1), color)

    rl.draw_line(PAD, SCREEN_H - 60, LEFT_W, SCREEN_H - 60, rl.DARKGRAY)
    hint = f"Perfect ratio: A={PERFECT_A}, B={PERFECT_B}  |  AAB chain: >=8 AAB + resonance>={MIN_RESONANCE_X*1.5:.1f}"
    draw_label(hint, PAD, SCREEN_H - 46, FS_SMALL, rl.GRAY)

    # Vertical divider
    rl.draw_line(LEFT_W + PAD, PAD, LEFT_W + PAD, SCREEN_H - PAD, rl.DARKGRAY)

    # --- Right panel ---
    draw_info_panel(result, RIGHT_X, PAD * 2, RIGHT_W)

    rl.end_drawing()

rl.close_window()
