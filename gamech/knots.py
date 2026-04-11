import pyray as rl
import math

SCREEN_W = 1920
SCREEN_H = 1080
FONT_SIZE = 32
BG = rl.Color(240, 240, 240, 255)
OUTLINE_COL = rl.Color(30, 30, 30, 255)
HOVER_R = 18.0
MIN_THICK = 4.0
MAX_THICK = 60.0
THICK_STEP = 2.0
SEGMENTS = 120
OUTLINE_PAD = 6.0

def catmull_to_bezier(pts):
    n = len(pts)
    segs = []
    for i in range(n - 1):
        p0 = pts[max(i-1, 0)]
        p1 = pts[i]
        p2 = pts[i+1]
        p3 = pts[min(i+2, n-1)]
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0
        segs.append((p1, (c1x, c1y), (c2x, c2y), p2))
    return segs

def eval_cubic(seg, t):
    p0, c1, c2, p1 = seg
    mt = 1.0 - t
    x = mt**3*p0[0] + 3*mt**2*t*c1[0] + 3*mt*t**2*c2[0] + t**3*p1[0]
    y = mt**3*p0[1] + 3*mt**2*t*c1[1] + 3*mt*t**2*c2[1] + t**3*p1[1]
    return (x, y)

def build_segments(pts):
    # pts: list of (x, y, z)
    # returns list of (poly_points, z_value) per control-point interval
    if len(pts) < 2:
        return []
    xy = [(p[0], p[1]) for p in pts]
    bezier_segs = catmull_to_bezier(xy)
    result = []
    for i, seg in enumerate(bezier_segs):
        z = pts[i][2]
        poly = []
        for j in range(SEGMENTS):
            poly.append(eval_cubic(seg, j / SEGMENTS))
        poly.append(seg[3])
        result.append((poly, z))
    return result

def seg_normal(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    ln = math.hypot(dx, dy)
    if ln < 1e-9:
        return 0.0, 1.0
    return -dy / ln, dx / ln

def build_vnormals(poly):
    n = len(poly)
    if n < 2:
        return []
    normals = []
    for i in range(n - 1):
        ax, ay = poly[i]
        bx, by = poly[i+1]
        normals.append(seg_normal(ax, ay, bx, by))
    vnorm = [None] * n
    vnorm[0] = normals[0]
    vnorm[n-1] = normals[-1]
    for i in range(1, n - 1):
        nx0, ny0 = normals[i-1]
        nx1, ny1 = normals[i]
        mx, my = (nx0+nx1)*0.5, (ny0+ny1)*0.5
        ln = math.hypot(mx, my)
        vnorm[i] = (mx/ln, my/ln) if ln > 1e-9 else (nx0, ny0)
    return vnorm

def v2(x, y):
    return rl.Vector2(x, y)

def draw_strip_2d(poly, vnorm, half, color):
    n = len(poly)
    if n < 2:
        return
    for i in range(n - 1):
        ax, ay = poly[i]
        bx, by = poly[i+1]
        nx0, ny0 = vnorm[i]
        nx1, ny1 = vnorm[i+1]
        l0 = v2(ax + nx0*half, ay + ny0*half)
        r0 = v2(ax - nx0*half, ay - ny0*half)
        l1 = v2(bx + nx1*half, by + ny1*half)
        r1 = v2(bx - nx1*half, by - ny1*half)
        rl.draw_triangle(l0, r0, l1, color)
        rl.draw_triangle(r0, r1, l1, color)

def nearest_pt(pts, mx, my):
    best_i, best_d = -1, float('inf')
    for i, p in enumerate(pts):
        d = math.hypot(p[0] - mx, p[1] - my)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d

def nearest_seg_poly(poly, mx, my):
    best_i, best_d = 0, float('inf')
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        bx, by = poly[i+1]
        dx, dy = bx-ax, by-ay
        ln2 = dx*dx + dy*dy
        if ln2 == 0:
            continue
        t = max(0.0, min(1.0, ((mx-ax)*dx + (my-ay)*dy) / ln2))
        d = math.hypot(ax + t*dx - mx, ay + t*dy - my)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d

def full_poly(pts):
    # flat poly for hover detection
    if len(pts) < 2:
        return []
    xy = [(p[0], p[1]) for p in pts]
    segs = catmull_to_bezier(xy)
    poly = []
    for seg in segs:
        for i in range(SEGMENTS):
            poly.append(eval_cubic(seg, i / SEGMENTS))
    poly.append(segs[-1][3])
    return poly

def hovered_spline_idx(splines, full_polys, mx, my, threshold=40.0):
    for si in range(len(splines) - 1, -1, -1):
        if len(full_polys[si]) < 2:
            continue
        _, d = nearest_seg_poly(full_polys[si], mx, my)
        if d < threshold:
            return si
    return -1

def insert_near(pts, fp, mx, my):
    if len(fp) < 2:
        pts.append((mx, my, 0))
        return
    seg_i, _ = nearest_seg_poly(fp, mx, my)
    insert_at = min(seg_i // SEGMENTS + 1, len(pts))
    z = pts[min(insert_at, len(pts)-1)][2]
    pts.insert(insert_at, (mx, my, z))

def lighten(c, a=80):
    return rl.Color(min(255,c.r+a), min(255,c.g+a), min(255,c.b+a), 255)

def rebuild(splines):
    fp = [full_poly(s['pts']) for s in splines]
    # per-spline segments: list of (poly, z, si, thick, color)
    draw_segs = []
    for si, s in enumerate(splines):
        for poly, z in build_segments(s['pts']):
            vnorm = build_vnormals(poly)
            draw_segs.append({'poly': poly, 'vnorm': vnorm, 'z': z,
                              'si': si, 'thick': s['thick'], 'color': s['color']})
    return fp, draw_segs

def main():
    rl.init_window(SCREEN_W, SCREEN_H, "Knot Spline Editor")
    rl.set_target_fps(60)
    rl.rl_disable_backface_culling()

    splines = [
        {'pts': [(400.0,400.0,0),(700.0,300.0,1),(1000.0,500.0,0),(1300.0,400.0,0)],
         'color': rl.Color(210, 100, 20, 255), 'thick': 18.0},
        {'pts': [(400.0,600.0,0),(700.0,700.0,0),(1000.0,500.0,1),(1300.0,650.0,0)],
         'color': rl.Color(20, 150, 40, 255), 'thick': 18.0},
    ]

    full_polys, draw_segs = rebuild(splines)
    drag_s = -1
    drag_p = -1
    hov_s  = -1
    hov_p  = -1  # hovered control point index
    dirty  = True

    while not rl.window_should_close():
        mx = float(rl.get_mouse_x())
        my = float(rl.get_mouse_y())
        shift = rl.is_key_down(rl.KEY_LEFT_SHIFT) or rl.is_key_down(rl.KEY_RIGHT_SHIFT)

        if dirty:
            full_polys, draw_segs = rebuild(splines)
            dirty = False

        # hover detection
        if drag_s == -1:
            hov_s = hovered_spline_idx(splines, full_polys, mx, my)
            hov_p = -1
            if hov_s >= 0:
                bi, bd = nearest_pt(splines[hov_s]['pts'], mx, my)
                if bd < HOVER_R * 2.5:
                    hov_p = bi

        # z adjust on hovered control point
        if hov_s >= 0 and hov_p >= 0:
            p = splines[hov_s]['pts'][hov_p]
            if rl.is_key_pressed(rl.KEY_W):
                splines[hov_s]['pts'][hov_p] = (p[0], p[1], p[2] + 1)
                dirty = True
            if rl.is_key_pressed(rl.KEY_S):
                splines[hov_s]['pts'][hov_p] = (p[0], p[1], p[2] - 1)
                dirty = True

        # thickness / z-order of whole spline
        if hov_s >= 0:
            if rl.is_key_pressed(rl.KEY_A):
                splines[hov_s]['thick'] = max(MIN_THICK, splines[hov_s]['thick'] - THICK_STEP)
                dirty = True
            if rl.is_key_pressed(rl.KEY_D):
                splines[hov_s]['thick'] = min(MAX_THICK, splines[hov_s]['thick'] + THICK_STEP)
                dirty = True
            if rl.is_key_pressed(rl.KEY_Q) and hov_s > 0:
                splines.insert(hov_s-1, splines.pop(hov_s))
                hov_s -= 1
                dirty = True
            if rl.is_key_pressed(rl.KEY_E) and hov_s < len(splines)-1:
                splines.insert(hov_s+1, splines.pop(hov_s))
                hov_s += 1
                dirty = True

        # drag
        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
            for si in range(len(splines)-1, -1, -1):
                bi, bd = nearest_pt(splines[si]['pts'], mx, my)
                if bd < HOVER_R:
                    drag_s = si
                    drag_p = bi
                    break

        if rl.is_mouse_button_released(rl.MOUSE_BUTTON_LEFT):
            drag_s = -1
            drag_p = -1

        if drag_s >= 0:
            delta = rl.get_mouse_delta()
            p = splines[drag_s]['pts'][drag_p]
            splines[drag_s]['pts'][drag_p] = (p[0]+delta.x, p[1]+delta.y, p[2])
            dirty = True

        # add/remove points
        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_RIGHT):
            t = hov_s if hov_s >= 0 else 0
            sp = splines[t]
            if shift:
                if len(sp['pts']) > 2:
                    bi, bd = nearest_pt(sp['pts'], mx, my)
                    if bd < HOVER_R * 3:
                        sp['pts'].pop(bi)
                        dirty = True
            else:
                insert_near(sp['pts'], full_polys[t], mx, my)
                dirty = True

        # --- draw ---
        rl.begin_drawing()
        rl.clear_background(BG)

        # sort segments by z, then by spline index for stable order
        sorted_segs = sorted(draw_segs, key=lambda s: (s['z'], s['si']))

        for seg in sorted_segs:
            draw_strip_2d(seg['poly'], seg['vnorm'],
                          seg['thick'] * 0.5 + OUTLINE_PAD, OUTLINE_COL)
            draw_strip_2d(seg['poly'], seg['vnorm'],
                          seg['thick'] * 0.5, seg['color'])

        # control points
        for si, sp in enumerate(splines):
            pt_col = lighten(sp['color'], 80)
            for pi, p in enumerate(sp['pts']):
                px, py, pz = p
                is_drag = (si == drag_s and pi == drag_p)
                is_hov  = (si == hov_s  and pi == hov_p)
                r = HOVER_R * 1.4 if is_drag else HOVER_R
                rl.draw_circle(int(px), int(py), int(r), pt_col)
                rl.draw_circle_lines(int(px), int(py), int(r), OUTLINE_COL)
                # z label
                rl.draw_text(str(pz), int(px)+int(r)+2, int(py)-FONT_SIZE//2,
                             FONT_SIZE-8, OUTLINE_COL)

        rl.draw_text(
            "LMB drag | RMB add | Shift+RMB del | QE spline-order | AD thick | WS pt-z",
            20, 20, FONT_SIZE, rl.Color(60,60,60,255)
        )
        if hov_s >= 0:
            sp = splines[hov_s]
            info = f"spline={hov_s}  thick={sp['thick']:.0f}"
            if hov_p >= 0:
                info += f"  pt={hov_p}  z={sp['pts'][hov_p][2]}"
            rl.draw_text(info, 20, 60, FONT_SIZE, sp['color'])

        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()
