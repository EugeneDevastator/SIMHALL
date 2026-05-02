"""
Unstructured Grid Visualizer
============================
Requires:
    pip install raylib numpy scipy

Controls:
    1 / 2 / 3   - Square / Hex / Delaunay mesh
    R           - Regenerate Delaunay with new seed
    +/-         - More/fewer Delaunay points
    F           - Cycle field display  (OFF → SOLID → EDGES ONLY)
    A           - Toggle animated field (two blobs orbiting)
    N           - Toggle neighbor highlight on click
    Mouse click - Select cell, highlight neighbors
    ESC         - Quit
"""

import pyray as rl
import numpy as np
from scipy.spatial import Delaunay
import math, random

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
W, H      = 1920, 900
TITLE     = "Unstructured Grid Visualizer"
GRID_COLS = 18
GRID_ROWS = 12
CELL_W    = W // GRID_COLS
CELL_H    = H // GRID_ROWS
N_RANDOM  = 220
MARGIN    = 48

def C(r, g, b, a=255): return rl.Color(r, g, b, a)

BG        = C(13, 14, 18)
COL_EMPTY = C(28, 32, 45)          # cell fill when field is OFF
COL_EDGE  = C(55, 65, 90)          # default edge
COL_SEL   = C(255, 220, 60)        # selected cell
COL_NEIGH = C(255, 160, 40, 180)   # neighbor highlight
COL_DIM   = C(100, 115, 145)
COL_WHITE = C(255, 255, 255)

# Vivid temperature ramp: deep blue → cyan → green → yellow → hot red
RAMP = [
    (0.00, (  8,  20, 180)),   # cold blue
    (0.20, (  0, 180, 220)),   # cyan
    (0.45, ( 20, 210,  60)),   # green
    (0.70, (240, 210,  10)),   # yellow
    (0.85, (255, 100,   0)),   # orange
    (1.00, (230,  15,  15)),   # hot red
]

def ramp_color(t, alpha=255):
    t = max(0.0, min(1.0, t))
    for i in range(len(RAMP) - 1):
        t0, c0 = RAMP[i]
        t1, c1 = RAMP[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return C(int(c0[0] + f*(c1[0]-c0[0])),
                     int(c0[1] + f*(c1[1]-c0[1])),
                     int(c0[2] + f*(c1[2]-c0[2])), alpha)
    return C(*RAMP[-1][1], alpha)

def lerp_color(a, b, t):
    return C(int(a.r + t*(b.r-a.r)),
             int(a.g + t*(b.g-a.g)),
             int(a.b + t*(b.b-a.b)),
             int(a.a + t*(b.a-a.a)))

# ──────────────────────────────────────────────────────────
# FIELD COMPUTATION
# ──────────────────────────────────────────────────────────

def compute_field(cells, t=0.0):
    """Two gaussian heat blobs that can orbit over time t (seconds)."""
    r1 = W * 0.32
    r2 = W * 0.22
    cx1 = W/2 + r1 * math.cos(t * 0.5)
    cy1 = H/2 + r1 * math.sin(t * 0.5) * 0.5
    cx2 = W/2 + r2 * math.cos(t * 0.8 + 2.1)
    cy2 = H/2 + r2 * math.sin(t * 0.8 + 2.1) * 0.6
    sigma1 = (W * 0.20) ** 2
    sigma2 = (W * 0.13) ** 2
    for cell in cells:
        x, y = cell['center']
        v  = 0.85 * math.exp(-((x-cx1)**2 + (y-cy1)**2) / (2*sigma1))
        v += 0.70 * math.exp(-((x-cx2)**2 + (y-cy2)**2) / (2*sigma2))
        cell['field'] = min(1.0, v)

# ──────────────────────────────────────────────────────────
# MESH GENERATORS
# ──────────────────────────────────────────────────────────

def make_square_grid():
    cells, index = [], {}
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x0, y0 = col * CELL_W, row * CELL_H
            idx = row * GRID_COLS + col
            index[(row, col)] = idx
            cells.append({
                'verts'    : [(x0,y0),(x0+CELL_W,y0),(x0+CELL_W,y0+CELL_H),(x0,y0+CELL_H)],
                'center'   : (x0 + CELL_W/2, y0 + CELL_H/2),
                'neighbors': [],
                'field'    : 0.0,
            })
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            idx = index[(row, col)]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                k = (row+dr, col+dc)
                if k in index:
                    cells[idx]['neighbors'].append(index[k])
    return cells


def make_hex_grid():
    cells, index = [], {}
    size = min(CELL_W, CELL_H) * 0.55
    dx   = size * math.sqrt(3)
    dy   = size * 1.5

    def hex_corners(cx, cy, s):
        return [(cx + s*math.cos(math.radians(60*i-30)),
                 cy + s*math.sin(math.radians(60*i-30))) for i in range(6)]

    for row in range(int(H/dy) + 2):
        for col in range(int(W/dx) + 2):
            cx = col*dx + (dx/2 if row%2 else 0)
            cy = row*dy
            if cx > W+size or cy > H+size: continue
            idx = len(cells)
            index[(row, col)] = idx
            cells.append({'verts': hex_corners(cx,cy,size), 'center': (cx,cy),
                          'row': row, 'col': col, 'neighbors': [], 'field': 0.0})

    even_d = [(-1,-1),(-1,0),(0,1),(1,0),(1,-1),(0,-1)]
    odd_d  = [(-1, 0),(-1,1),(0,1),(1,1),(1, 0),(0,-1)]
    for cell in cells:
        r, c = cell['row'], cell['col']
        for dr, dc in (odd_d if r%2 else even_d):
            k = (r+dr, c+dc)
            if k in index:
                cell['neighbors'].append(index[k])
    return cells


def make_delaunay_mesh(n=N_RANDOM, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.uniform([MARGIN, MARGIN], [W-MARGIN, H-MARGIN], (n, 2))
    bnd = []
    for t in np.linspace(0, 1, 22, endpoint=False):
        bnd += [(MARGIN + t*(W-2*MARGIN), MARGIN),
                (MARGIN + t*(W-2*MARGIN), H-MARGIN),
                (MARGIN,                  MARGIN + t*(H-2*MARGIN)),
                (W-MARGIN,                MARGIN + t*(H-2*MARGIN))]
    pts = np.vstack([pts, bnd])
    tri = Delaunay(pts)

    cells = []
    for s in tri.simplices:
        verts = [tuple(pts[i]) for i in s]
        cx = sum(v[0] for v in verts)/3
        cy = sum(v[1] for v in verts)/3
        cells.append({'verts': verts, 'center': (cx,cy),
                      'simplex': tuple(sorted(s)), 'neighbors': [], 'field': 0.0})

    edge_map = {}
    for i, cell in enumerate(cells):
        s = cell['simplex']
        for edge in [(s[0],s[1]),(s[1],s[2]),(s[0],s[2])]:
            edge_map.setdefault(edge, []).append(i)
    for ts in edge_map.values():
        if len(ts) == 2:
            a, b = ts
            if b not in cells[a]['neighbors']: cells[a]['neighbors'].append(b)
            if a not in cells[b]['neighbors']: cells[b]['neighbors'].append(a)
    return cells

# ──────────────────────────────────────────────────────────
# HIT TEST
# ──────────────────────────────────────────────────────────

def point_in_poly(px, py, verts):
    inside, j = False, len(verts)-1
    for i in range(len(verts)):
        xi,yi = verts[i]; xj,yj = verts[j]
        if ((yi>py) != (yj>py)) and (px < (xj-xi)*(py-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

def find_cell(cells, mx, my):
    for i, c in enumerate(cells):
        if point_in_poly(mx, my, c['verts']): return i
    return -1

# ──────────────────────────────────────────────────────────
# DRAW HELPERS
# ──────────────────────────────────────────────────────────

def draw_poly_filled(verts, color):
    cx = sum(v[0] for v in verts)/len(verts)
    cy = sum(v[1] for v in verts)/len(verts)
    for i in range(len(verts)):
        ax,ay = verts[i]; bx,by = verts[(i+1)%len(verts)]
        rl.draw_triangle(rl.Vector2(cx,cy), rl.Vector2(bx,by), rl.Vector2(ax,ay), color)

def draw_poly_lines(verts, color, thick=1.0):
    for i in range(len(verts)):
        ax,ay = verts[i]; bx,by = verts[(i+1)%len(verts)]
        rl.draw_line_ex(rl.Vector2(ax,ay), rl.Vector2(bx,by), thick, color)

# ──────────────────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────────────────
FIELD_MODES = ["OFF", "SOLID", "EDGES"]

# Raylib's default font has a base pixel height of 10px.
# All sizes must be multiples of 10 to render crisply without internal rounding.
FS_TITLE  = 40   # mode name in top-left    (was 18 → 20 → snap up to 40)
FS_TAGS   = 30   # field/animated tags      (was 14 → 28 → snap to 30)
FS_HINTS  = 20   # keyboard hints           (was 13 → 26 → snap to 30... keep 20 so it fits)
FS_INFO   = 30   # bottom status bar        (was 14 → 28 → snap to 30)
FS_LEGEND = 20   # HOT / COLD labels        (was 12 → 24 → snap to 20... keeps legend tidy)

TOP_H    = 60    # top bar height to fit FS_TITLE
BOT_H    = 50    # bottom bar height to fit FS_INFO

def draw_hud(mode, field_mode, animated, n_pts, selected, cells):
    # top bar
    rl.draw_rectangle(0, 0, W, TOP_H, C(8, 9, 14, 230))
    mode_name = ["","SQUARE GRID","HEX GRID","UNSTRUCTURED (DELAUNAY)"][mode]
    rl.draw_text(mode_name, 12, (TOP_H - FS_TITLE)//2, FS_TITLE, COL_SEL)

    tags = []
    tags.append(f"field:{FIELD_MODES[field_mode]}")
    if animated: tags.append("ANIMATED")
    tag_str = "  [" + "]  [".join(tags) + "]"
    rl.draw_text(tag_str, W//2 - 100, (TOP_H - FS_TAGS)//2, FS_TAGS, C(180,230,120))

    hints = "[1/2/3] grid  [R] regen  [+/-] pts  [F] field  [A] animate  [N] neighbors"
    rl.draw_text(hints, W - 740, (TOP_H - FS_HINTS)//2, FS_HINTS, COL_DIM)

    # bottom bar
    rl.draw_rectangle(0, H - BOT_H, W, BOT_H, C(8, 9, 14, 220))
    info = f"cells: {len(cells)}"
    if mode == 3: info += f"  pts: {n_pts}"
    if selected >= 0:
        cell = cells[selected]
        info += (f"   ●  cell #{selected}  "
                 f"neighbors: {len(cell['neighbors'])}  "
                 f"field value: {cell['field']:.3f}")
    rl.draw_text(info, 12, H - BOT_H + (BOT_H - FS_INFO)//2, FS_INFO, COL_DIM)

    # field legend (only when field is on)
    if field_mode > 0:
        lx, ly, lw, lh = W-54, TOP_H + 10, 26, 200
        for i in range(lh):
            t = 1.0 - i/lh
            rl.draw_rectangle(lx, ly+i, lw, 1, ramp_color(t, 255))
        rl.draw_rectangle_lines_ex(rl.Rectangle(lx, ly, lw, lh), 1, COL_EDGE)
        rl.draw_text("HOT",  lx - 50, ly - 2,       FS_LEGEND, C(230,80,40))
        rl.draw_text("COLD", lx - 60, ly + lh - 16, FS_LEGEND, C(60,120,220))

# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    rl.init_window(W, H, TITLE)
    rl.set_target_fps(60)

    mode       = 3
    field_mode = 1      # 0=off, 1=solid fill, 2=edges only
    animated   = False
    show_nbrs  = True
    n_pts      = N_RANDOM
    selected   = -1
    seed       = 42
    cells      = []

    def build(m):
        nonlocal cells, selected
        selected = -1
        if   m == 1: cells = make_square_grid()
        elif m == 2: cells = make_hex_grid()
        elif m == 3: cells = make_delaunay_mesh(n_pts, seed)
        compute_field(cells, 0.0)

    build(mode)

    while not rl.window_should_close():

        # ── INPUT ──────────────────────────────────────────
        if rl.is_key_pressed(rl.KeyboardKey.KEY_ONE):   mode=1; build(mode)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_TWO):   mode=2; build(mode)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_THREE): mode=3; build(mode)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_R) and mode==3:
            seed = random.randint(0,9999); build(mode)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_EQUAL):
            n_pts = min(600, n_pts+30)
            if mode==3: build(mode)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_MINUS):
            n_pts = max(40, n_pts-30)
            if mode==3: build(mode)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F):
            field_mode = (field_mode + 1) % 3
        if rl.is_key_pressed(rl.KeyboardKey.KEY_A):
            animated = not animated
        if rl.is_key_pressed(rl.KeyboardKey.KEY_N):
            show_nbrs = not show_nbrs
        if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
            mp  = rl.get_mouse_position()
            hit = find_cell(cells, mp.x, mp.y)
            selected = hit if hit != selected else -1

        # ── UPDATE field ───────────────────────────────────
        if animated:
            compute_field(cells, rl.get_time())

        # ── DRAW ───────────────────────────────────────────
        rl.begin_drawing()
        rl.clear_background(BG)
        

        for i, cell in enumerate(cells):
            verts    = cell['verts']
            is_sel   = (i == selected)
            is_neigh = show_nbrs and selected >= 0 and i in cells[selected]['neighbors']

            # ── fill ──
            if is_sel:
                draw_poly_filled(verts, COL_SEL)
            elif is_neigh:
                draw_poly_filled(verts, COL_NEIGH)
            elif field_mode == 1:
                # SOLID: vivid full-brightness field colour
                draw_poly_filled(verts, ramp_color(cell['field'], 255))
            elif field_mode == 2:
                # EDGES ONLY: dark fill, coloured border
                draw_poly_filled(verts, COL_EMPTY)
            else:
                # OFF: flat dark fill
                draw_poly_filled(verts, COL_EMPTY)

            # ── edges ──
            if is_sel:
                draw_poly_lines(verts, COL_SEL, 2.5)
            elif is_neigh:
                draw_poly_lines(verts, COL_NEIGH, 2.0)
            elif field_mode == 2:
                # edges coloured by field value, thick enough to read
                draw_poly_lines(verts, ramp_color(cell['field'], 255), 2.5)
            else:
                draw_poly_lines(verts, COL_EDGE, 0.8)

            # centroid dot for Delaunay
            if mode == 3 and field_mode != 1:
                cx, cy = cell['center']
                rl.draw_circle(int(cx), int(cy), 2,
                               COL_WHITE if is_sel else C(70,85,120))

        draw_hud(mode, field_mode, animated, n_pts, selected, cells)
        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()