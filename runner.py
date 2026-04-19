import pyray as rl
import os
import ast
import subprocess
import sys

SCREEN_W = 1280
SCREEN_H = 900

LIST_W   = 260
INFO_W   = SCREEN_W - LIST_W

BG      = rl.Color(30, 30, 35, 255)
SEL_COL = rl.Color(60, 90, 60, 255)
HOV_COL = rl.Color(50, 50, 60, 255)
TEXT_COL= rl.Color(200, 200, 200, 255)
DIM_COL = rl.Color(120, 120, 130, 255)
DIV_COL = rl.Color(60, 60, 70, 255)
BTN_COL = rl.Color(50, 100, 50, 255)
BTN_HOV = rl.Color(70, 140, 70, 255)
CAT_COL = rl.Color(40, 40, 50, 255)
CAT_TXT = rl.Color(160, 160, 180, 255)

FS  = 24
FSS = 21
FSC = 18  # category label

ITEM_H   = 52
CAT_H    = 32
HEADER_H = 48

BTN_W, BTN_H = 160, 44
BTN_X = LIST_W + (INFO_W - BTN_W) // 2
BTN_Y = SCREEN_H - BTN_H - 20

PREVIEW_AREA_H = SCREEN_H // 2 - HEADER_H
DESC_Y_START   = HEADER_H + PREVIEW_AREA_H + 12

SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}


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


def scan_all():
    """
    Returns list of row dicts:
      type="cat"  -> {type, label}
      type="sim"  -> {type, name, path, desc, png}
    Also returns flat list of sim-only dicts for indexed selection.
    """
    rows = []
    sims = []
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d not in SKIP_DIRS and not d.startswith(".")
    )
    for d in dirs:
        folder = os.path.join(base, d)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".py"))
        if not files:
            continue
        rows.append({"type": "cat", "label": d})
        for fname in files:
            name = fname[:-3]
            path = os.path.join(folder, fname)
            desc = read_sim_meta(path)
            png  = os.path.join(folder, name + ".png")
            entry = {"type": "sim", "name": name, "path": path, "desc": desc, "png": png}
            rows.append(entry)
            sims.append(entry)
    return rows, sims


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


def total_list_height(rows):
    h = 0
    for r in rows:
        h += CAT_H if r["type"] == "cat" else ITEM_H
    return h


rl.init_window(SCREEN_W, SCREEN_H, "Sim Runner")
rl.set_target_fps(60)

font   = rl.load_font_ex("C:/Windows/Fonts/arialbd.ttf", FS,  None, 0)
font_s = rl.load_font_ex("C:/Windows/Fonts/arialbd.ttf", FSS, None, 0)
font_c = rl.load_font_ex("C:/Windows/Fonts/arialbd.ttf", FSC, None, 0)

rows, sims = scan_all()
selected     = 0  # index into sims[]
preview_tex  = None
preview_name = None
scroll_offset = 0

while not rl.window_should_close():
    mouse = rl.get_mouse_position()
    mx, my = mouse.x, mouse.y
    lmb = rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)

    if rl.is_key_pressed(rl.KeyboardKey.KEY_F5):
        if preview_tex:
            rl.unload_texture(preview_tex)
            preview_tex  = None
            preview_name = None
        rows, sims = scan_all()
        selected = min(selected, max(0, len(sims) - 1))

    wheel = rl.get_mouse_wheel_move()
    if mx < LIST_W and wheel != 0.0:
        scroll_offset -= int(wheel) * ITEM_H
        max_scroll = max(0, total_list_height(rows) - (SCREEN_H - HEADER_H))
        scroll_offset = max(0, min(scroll_offset, max_scroll))

    # click in list: map pixel -> row -> sim index
    if lmb and mx < LIST_W and my > HEADER_H:
        cy = HEADER_H - scroll_offset
        sim_idx = 0
        for r in rows:
            rh = CAT_H if r["type"] == "cat" else ITEM_H
            if r["type"] == "sim":
                if cy <= my < cy + rh:
                    selected = sim_idx
                    break
                sim_idx += 1
            cy += rh

    # load preview
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

    # header
    rl.draw_rectangle(0, 0, SCREEN_W, HEADER_H, rl.Color(40, 40, 50, 255))
    rl.draw_text_ex(font,   "SIM RUNNER", rl.Vector2(12, 10),          FS,  1, TEXT_COL)
    rl.draw_text_ex(font_s, "F5 reload",  rl.Vector2(LIST_W + 12, 14), FSS, 1, DIM_COL)

    rl.draw_rectangle(LIST_W - 1, 0, 1, SCREEN_H, DIV_COL)

    # --- list ---
    rl.begin_scissor_mode(0, HEADER_H, LIST_W, SCREEN_H - HEADER_H)
    cy = HEADER_H - scroll_offset
    sim_idx = 0
    for r in rows:
        if r["type"] == "cat":
            rh = CAT_H
            if cy + rh >= HEADER_H and cy <= SCREEN_H:
                rl.draw_rectangle(0, cy, LIST_W - 1, rh, CAT_COL)
                rl.draw_text_ex(font_c, r["label"].upper(),
                                rl.Vector2(8, cy + (rh - FSC) // 2), FSC, 1, CAT_TXT)
                rl.draw_rectangle(0, cy + rh - 1, LIST_W - 1, 1, DIV_COL)
            cy += rh
        else:
            rh = ITEM_H
            if cy + rh >= HEADER_H and cy <= SCREEN_H:
                if sim_idx == selected:
                    rl.draw_rectangle(0, cy, LIST_W - 1, rh, SEL_COL)
                elif 0 <= mx < LIST_W and cy <= my < cy + rh:
                    rl.draw_rectangle(0, cy, LIST_W - 1, rh, HOV_COL)
                rl.draw_text_ex(font, r["name"],
                                rl.Vector2(18, cy + (rh - FS) // 2), FS, 1, TEXT_COL)
                rl.draw_rectangle(0, cy + rh - 1, LIST_W - 1, 1, DIV_COL)
            cy += rh
            sim_idx += 1
    rl.end_scissor_mode()

    # --- info panel ---
    pad   = 12
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
                        rl.Vector2(LIST_W + pad, img_y + img_h // 2), FSS, 1, DIM_COL)

    rl.draw_rectangle(LIST_W, HEADER_H + PREVIEW_AREA_H, INFO_W, 1, DIV_COL)

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

    if sims:
        bc  = BTN_HOV if btn_hov else BTN_COL
        rl.draw_rectangle(BTN_X, BTN_Y, BTN_W, BTN_H, bc)
        lbl   = "RUN"
        lbl_w = rl.measure_text_ex(font, lbl, FS, 1).x
        rl.draw_text_ex(font, lbl,
                        rl.Vector2(BTN_X + (BTN_W - lbl_w) // 2, BTN_Y + (BTN_H - FS) // 2),
                        FS, 1, TEXT_COL)

    if not sims:
        rl.draw_text_ex(font_s, "No sims found.",
                        rl.Vector2(LIST_W + 20, SCREEN_H // 2), FSS, 1, DIM_COL)

    rl.end_drawing()

if preview_tex:
    rl.unload_texture(preview_tex)
rl.unload_font(font)
rl.unload_font(font_s)
rl.unload_font(font_c)
rl.close_window()
