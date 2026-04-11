import pyray as rl
import os
import ast
import subprocess
import sys

SCREEN_W = 900
SCREEN_H = 600
SIMS_DIR = "sims"

LIST_W = 260
PREVIEW_X = LIST_W
INFO_W = SCREEN_W - LIST_W

BG      = rl.Color(30, 30, 35, 255)
SEL_COL = rl.Color(60, 90, 60, 255)
HOV_COL = rl.Color(50, 50, 60, 255)
TEXT_COL= rl.Color(200, 200, 200, 255)
DIM_COL = rl.Color(120, 120, 130, 255)
DIV_COL = rl.Color(60, 60, 70, 255)
BTN_COL = rl.Color(50, 100, 50, 255)
BTN_HOV = rl.Color(70, 140, 70, 255)

FS  = 24
FSS = 21

ITEM_H = 52
HEADER_H = 48

BTN_W, BTN_H = 160, 44
BTN_X = LIST_W + (INFO_W - BTN_W) // 2
BTN_Y = SCREEN_H - BTN_H - 20


def read_sim_meta(path):
    desc = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)):
            desc = tree.body[0].value.value.strip()
        else:
            for node in tree.body:
                if (isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "SIM_DESC"
                        and isinstance(node.value, ast.Constant)):
                    desc = node.value.value.strip()
                    break
    except Exception:
        pass
    return desc


def scan_sims():
    sims = []
    if not os.path.isdir(SIMS_DIR):
        return sims
    for fname in sorted(os.listdir(SIMS_DIR)):
        if not fname.endswith(".py"):
            continue
        name = fname[:-3]
        path = os.path.join(SIMS_DIR, fname)
        desc = read_sim_meta(path)
        png  = os.path.join(SIMS_DIR, name + ".png")
        sims.append({"name": name, "path": path, "desc": desc, "png": png})
    return sims


def wrap_text(text, max_chars):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= max_chars:
            cur = cur + (" " if cur else "") + w
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def launch(sim_path):
    subprocess.Popen(
        [sys.executable, sim_path],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )


rl.init_window(SCREEN_W, SCREEN_H, "Sim Runner")
rl.set_target_fps(60)

font = rl.load_font_ex("C:/Windows/Fonts/arialbd.ttf", FS, None, 0)
font_s = rl.load_font_ex("C:/Windows/Fonts/arialbd.ttf", FSS, None, 0)

sims = scan_sims()
selected = 0
preview_tex = None
preview_name = None
scroll_offset = 0

# preview area: top half of info panel, leaving room for text + button
PREVIEW_AREA_H = SCREEN_H // 2 - HEADER_H
DESC_Y_START   = HEADER_H + PREVIEW_AREA_H + 12

while not rl.window_should_close():
    mouse = rl.get_mouse_position()
    mx, my = mouse.x, mouse.y
    lmb = rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)

    if rl.is_key_pressed(rl.KeyboardKey.KEY_F5):
        if preview_tex:
            rl.unload_texture(preview_tex)
            preview_tex = None
            preview_name = None
        sims = scan_sims()
        selected = min(selected, max(0, len(sims) - 1))

    wheel = rl.get_mouse_wheel_move()
    if mx < LIST_W and wheel != 0.0:
        scroll_offset -= int(wheel) * ITEM_H
        max_scroll = max(0, len(sims) * ITEM_H - (SCREEN_H - HEADER_H))
        scroll_offset = max(0, min(scroll_offset, max_scroll))

    if lmb and mx < LIST_W and my > HEADER_H:
        idx = (int(my) - HEADER_H + scroll_offset) // ITEM_H
        if 0 <= idx < len(sims):
            selected = idx

    if sims:
        cur = sims[selected]
        if cur["name"] != preview_name:
            if preview_tex:
                rl.unload_texture(preview_tex)
                preview_tex = None
            preview_name = cur["name"]
            if os.path.isfile(cur["png"]):
                preview_tex = rl.load_texture(cur["png"])

    btn_hov = (BTN_X <= mx <= BTN_X + BTN_W and BTN_Y <= my <= BTN_Y + BTN_H)
    if lmb and btn_hov and sims:
        launch(sims[selected]["path"])

    # ---- DRAW ----
    rl.begin_drawing()
    rl.clear_background(BG)

    # header bar
    rl.draw_rectangle(0, 0, SCREEN_W, HEADER_H, rl.Color(40, 40, 50, 255))
    rl.draw_text_ex(font, "SIM RUNNER", rl.Vector2(12, 10), FS, 1, TEXT_COL)
    rl.draw_text_ex(font_s, "F5 reload", rl.Vector2(LIST_W + 12, 14), FSS, 1, DIM_COL)

    # divider list | info
    rl.draw_rectangle(LIST_W - 1, 0, 1, SCREEN_H, DIV_COL)

    # --- list ---
    rl.begin_scissor_mode(0, HEADER_H, LIST_W, SCREEN_H - HEADER_H)
    for i, s in enumerate(sims):
        iy = HEADER_H + i * ITEM_H - scroll_offset
        if iy + ITEM_H < HEADER_H or iy > SCREEN_H:
            continue
        if i == selected:
            rl.draw_rectangle(0, iy, LIST_W - 1, ITEM_H, SEL_COL)
        elif 0 <= mx < LIST_W and iy <= my <= iy + ITEM_H:
            rl.draw_rectangle(0, iy, LIST_W - 1, ITEM_H, HOV_COL)
        rl.draw_text_ex(font, s["name"], rl.Vector2(10, iy + (ITEM_H - FS) // 2), FS, 1, TEXT_COL)
        rl.draw_rectangle(0, iy + ITEM_H - 1, LIST_W - 1, 1, DIV_COL)
    rl.end_scissor_mode()

    # --- info panel: image top, text below ---
    pad = 12
    img_x = LIST_W + pad
    img_y = HEADER_H + pad
    img_w = INFO_W - pad * 2
    img_h = PREVIEW_AREA_H - pad

    if preview_tex:
        tr = preview_tex.width / preview_tex.height
        if img_w / img_h > tr:
            dh = img_h
            dw = int(dh * tr)
        else:
            dw = img_w
            dh = int(dw / tr)
        dx = LIST_W + pad + (img_w - dw) // 2
        dy = img_y + (img_h - dh) // 2
        rl.draw_texture_ex(
            preview_tex,
            rl.Vector2(float(dx), float(dy)),
            0.0,
            float(dw) / preview_tex.width,
            rl.WHITE
        )
    else:
        rl.draw_text_ex(font_s, "no preview",
                        rl.Vector2(LIST_W + pad, img_y + img_h // 2),
                        FSS, 1, DIM_COL)

    # divider image | desc
    rl.draw_rectangle(LIST_W, HEADER_H + PREVIEW_AREA_H, INFO_W, 1, DIV_COL)

    # description text
    if sims:
        desc = sims[selected]["desc"] or "No description."
        chars_per_line = max(10, (INFO_W - pad * 2) // (FSS // 2 + 2))
        lines = wrap_text(desc, chars_per_line)
        ty = DESC_Y_START
        for line in lines:
            if ty + FSS > BTN_Y - 8:
                break
            rl.draw_text_ex(font_s, line, rl.Vector2(LIST_W + pad, ty), FSS, 1, TEXT_COL)
            ty += FSS + 6

    # run button
    if sims:
        bc = BTN_HOV if btn_hov else BTN_COL
        rl.draw_rectangle(BTN_X, BTN_Y, BTN_W, BTN_H, bc)
        lbl = "RUN"
        lbl_w = rl.measure_text_ex(font, lbl, FS, 1).x
        rl.draw_text_ex(font, lbl,
                        rl.Vector2(BTN_X + (BTN_W - lbl_w) // 2, BTN_Y + (BTN_H - FS) // 2),
                        FS, 1, TEXT_COL)

    if not sims:
        rl.draw_text_ex(font_s, f"No sims found in '{SIMS_DIR}/'",
                        rl.Vector2(LIST_W + 20, SCREEN_H // 2), FSS, 1, DIM_COL)

    rl.end_drawing()

if preview_tex:
    rl.unload_texture(preview_tex)
rl.unload_font(font)
rl.unload_font(font_s)
rl.close_window()
