import pyray as rl
import os
import ast
import subprocess
import sys

SCREEN_W = 900
SCREEN_H = 600
SIMS_DIR = "sims"

LIST_X = 0
LIST_W = 260
PREVIEW_X = LIST_W
PREVIEW_W = 380
DESC_X = LIST_W + PREVIEW_W
DESC_W = SCREEN_W - DESC_X

BG        = rl.Color(30, 30, 35, 255)
SEL_COL   = rl.Color(60, 90, 60, 255)
HOV_COL   = rl.Color(50, 50, 60, 255)
TEXT_COL  = rl.Color(200, 200, 200, 255)
DIM_COL   = rl.Color(120, 120, 130, 255)
DIV_COL   = rl.Color(60, 60, 70, 255)
BTN_COL   = rl.Color(50, 100, 50, 255)
BTN_HOV   = rl.Color(70, 140, 70, 255)

FS  = 16
FSS = 14

def read_sim_meta(path):
    desc = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
        # try module docstring
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)):
            desc = tree.body[0].value.value.strip()
        else:
            # try SIM_DESC = "..."
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
    subprocess.Popen([sys.executable, sim_path],
                     creationflags=subprocess.CREATE_NEW_CONSOLE
                     if sys.platform == "win32" else 0)

rl.init_window(SCREEN_W, SCREEN_H, "Sim Runner")
rl.set_target_fps(60)

sims = scan_sims()
selected = 0
preview_tex = None
preview_name = None
procs = []

scroll_offset = 0
ITEM_H = 36

BTN_W, BTN_H = 120, 34
BTN_X = DESC_X + (DESC_W - BTN_W) // 2
BTN_Y = SCREEN_H - BTN_H - 16

while not rl.window_should_close():
    mouse = rl.get_mouse_position()
    mx, my = mouse.x, mouse.y
    lmb = rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)

    # reload sims list
    if rl.is_key_pressed(rl.KeyboardKey.KEY_F5):
        if preview_tex:
            rl.unload_texture(preview_tex)
            preview_tex = None
            preview_name = None
        sims = scan_sims()
        selected = min(selected, max(0, len(sims) - 1))

    # scroll list
    wheel = rl.get_mouse_wheel_move()
    if mx < LIST_W and wheel != 0.0:
        scroll_offset -= int(wheel) * ITEM_H
        max_scroll = max(0, len(sims) * ITEM_H - (SCREEN_H - 40))
        scroll_offset = max(0, min(scroll_offset, max_scroll))

    # click list
    if lmb and mx < LIST_W:
        idx = (int(my) + scroll_offset - 40) // ITEM_H
        if 0 <= idx < len(sims):
            selected = idx

    # load preview texture
    if sims:
        cur = sims[selected]
        if cur["name"] != preview_name:
            if preview_tex:
                rl.unload_texture(preview_tex)
                preview_tex = None
            preview_name = cur["name"]
            if os.path.isfile(cur["png"]):
                preview_tex = rl.load_texture(cur["png"])

    # run button
    btn_hov = (BTN_X <= mx <= BTN_X + BTN_W and BTN_Y <= my <= BTN_Y + BTN_H)
    if lmb and btn_hov and sims:
        launch(sims[selected]["path"])

    # ---- DRAW ----
    rl.begin_drawing()
    rl.clear_background(BG)

    # dividers
    rl.draw_rectangle(LIST_W - 1, 0, 1, SCREEN_H, DIV_COL)
    rl.draw_rectangle(PREVIEW_X + PREVIEW_W - 1, 0, 1, SCREEN_H, DIV_COL)

    # header
    rl.draw_rectangle(0, 0, SCREEN_W, 36, rl.Color(40, 40, 50, 255))
    rl.draw_text("SIM RUNNER", 10, 8, FS, TEXT_COL)
    rl.draw_text("F5 reload", LIST_W + 8, 10, FSS, DIM_COL)

    # list
    rl.begin_scissor_mode(LIST_X, 40, LIST_W, SCREEN_H - 40)
    for i, s in enumerate(sims):
        iy = 40 + i * ITEM_H - scroll_offset
        if iy + ITEM_H < 40 or iy > SCREEN_H:
            continue
        if i == selected:
            rl.draw_rectangle(LIST_X, iy, LIST_W - 1, ITEM_H, SEL_COL)
        elif LIST_X <= mx <= LIST_X + LIST_W and iy <= my <= iy + ITEM_H:
            rl.draw_rectangle(LIST_X, iy, LIST_W - 1, ITEM_H, HOV_COL)
        rl.draw_text(s["name"], LIST_X + 10, iy + (ITEM_H - FS) // 2, FS, TEXT_COL)
        rl.draw_rectangle(LIST_X, iy + ITEM_H - 1, LIST_W - 1, 1, DIV_COL)
    rl.end_scissor_mode()

    # preview
    if preview_tex:
        pw = PREVIEW_W - 16
        ph = SCREEN_H - 40 - 16
        tr = preview_tex.width / preview_tex.height
        if pw / ph > tr:
            dh = ph
            dw = int(dh * tr)
        else:
            dw = pw
            dh = int(dw / tr)
        dx = PREVIEW_X + (PREVIEW_W - dw) // 2
        dy = 40 + (ph - dh) // 2 + 8
        rl.draw_texture_ex(preview_tex, rl.Vector2(float(dx), float(dy)),
                           0.0, float(dw) / preview_tex.width,
                           rl.WHITE)
    else:
        rl.draw_text("no preview", PREVIEW_X + 20, SCREEN_H // 2, FS, DIM_COL)

    # description
    if sims:
        desc = sims[selected]["desc"] or "No description."
        max_chars = DESC_W // (FSS // 2 + 3)
        lines = wrap_text(desc, max(10, max_chars))
        ty = 48
        for line in lines:
            if ty + FSS > BTN_Y - 8:
                break
            rl.draw_text(line, DESC_X + 10, ty, FSS, TEXT_COL)
            ty += FSS + 4

    # run button
    if sims:
        bc = BTN_HOV if btn_hov else BTN_COL
        rl.draw_rectangle(BTN_X, BTN_Y, BTN_W, BTN_H, bc)
        rl.draw_text("RUN", BTN_X + BTN_W // 2 - 16, BTN_Y + 8, FS, TEXT_COL)

    if not sims:
        rl.draw_text(f"No sims found in '{SIMS_DIR}/'",
                     LIST_W + 20, SCREEN_H // 2, FS, DIM_COL)

    rl.end_drawing()

if preview_tex:
    rl.unload_texture(preview_tex)
rl.close_window()
