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
DEFAULT_THICK = 18.0

ROPE_COLORS = [
    rl.Color(210, 100,  20, 255),
    rl.Color( 20, 150,  40, 255),
    rl.Color( 30,  80, 200, 255),
    rl.Color(160,  30, 160, 255),
    rl.Color(180, 160,  10, 255),
]

Z_DARK  = 0.45
Z_LIGHT = 1.55
Z_MIN   = -4
Z_MAX   =  4
CAM_Z   = 100.0

def screen_to_world(sx, sy):
    return (sx - SCREEN_W * 0.5+100, -(sy - SCREEN_H * 0.5))

def z_tint(base_color, z):
    t = (z - Z_MIN) / max(Z_MAX - Z_MIN, 1)
    t = max(0.0, min(1.0, t))
    f = Z_DARK + (Z_LIGHT - Z_DARK) * t
    return rl.Color(
        int(max(0, min(255, base_color.r * f))),
        int(max(0, min(255, base_color.g * f))),
        int(max(0, min(255, base_color.b * f))),
        255
    )

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
    if len(pts) < 2:
        return []
    xy = [(p[0], p[1]) for p in pts]
    bezier_segs = catmull_to_bezier(xy)
    result = []
    for i, seg in enumerate(bezier_segs):
        z0 = pts[i][2]
        z1 = pts[i+1][2]
        poly = []
        for j in range(SEGMENTS):
            poly.append(eval_cubic(seg, j / SEGMENTS))
        poly.append(seg[3])
        result.append((poly, z0, z1))
    return result

def seg_normal(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    ln = math.hypot(dx, dy)
    if ln < 1e-9:
        return 0.0, 1.0
    return -dy / ln, dx / ln

def build_poly_and_normals(poly):
    n = len(poly)
    if n < 2:
        return poly, []
    seg_norms = []
    for i in range(n - 1):
        ax, ay = poly[i]
        bx, by = poly[i+1]
        seg_norms.append(seg_normal(ax, ay, bx, by))
    vnorm = [None] * n
    vnorm[0] = seg_norms[0]
    vnorm[n-1] = seg_norms[-1]
    for i in range(1, n - 1):
        nx0, ny0 = seg_norms[i-1]
        nx1, ny1 = seg_norms[i]
        mx, my = (nx0+nx1)*0.5, (ny0+ny1)*0.5
        ln = math.hypot(mx, my)
        vnorm[i] = (mx/ln, my/ln) if ln > 1e-9 else (nx0, ny0)
    return poly, vnorm

def draw_strip_3d(poly, vnorm, half, base_color, z0, z1, is_outline):
    n = len(poly)
    if n < 2:
        return
    total = n - 1
    rl.rl_begin(rl.RL_TRIANGLES)
    for i in range(total):
        t0 = i / total
        t1 = (i + 1) / total
        z_a = z0 + (z1 - z0) * t0
        z_b = z0 + (z1 - z0) * t1
        if is_outline:
            col = OUTLINE_COL
            za, zb = z_a - 0.15, z_b - 0.15
        else:
            col = z_tint(base_color, (z_a + z_b) * 0.5)
            za, zb = z_a, z_b

        ax, ay = poly[i]
        bx, by = poly[i+1]
        nx0, ny0 = vnorm[i]
        nx1, ny1 = vnorm[i+1]

        l0x, l0y = ax + nx0*half, ay + ny0*half
        r0x, r0y = ax - nx0*half, ay - ny0*half
        l1x, l1y = bx + nx1*half, by + ny1*half
        r1x, r1y = bx - nx1*half, by - ny1*half

        rl.rl_color4ub(col.r, col.g, col.b, col.a)
        rl.rl_vertex3f(l0x, l0y, za)
        rl.rl_vertex3f(r0x, r0y, za)
        rl.rl_vertex3f(l1x, l1y, zb)
        rl.rl_vertex3f(r0x, r0y, za)
        rl.rl_vertex3f(r1x, r1y, zb)
        rl.rl_vertex3f(l1x, l1y, zb)
    rl.rl_end()

def draw_point_3d(wx, wy, pz, radius, col):
    r = radius
    rl.rl_begin(rl.RL_TRIANGLES)
    rl.rl_color4ub(col.r, col.g, col.b, col.a)
    rl.rl_vertex3f(wx - r, wy - r, pz)
    rl.rl_vertex3f(wx + r, wy - r, pz)
    rl.rl_vertex3f(wx + r, wy + r, pz)
    rl.rl_vertex3f(wx - r, wy - r, pz)
    rl.rl_vertex3f(wx + r, wy + r, pz)
    rl.rl_vertex3f(wx - r, wy + r, pz)
    rl.rl_end()
    rl.rl_begin(rl.RL_LINES)
    rl.rl_color4ub(OUTLINE_COL.r, OUTLINE_COL.g, OUTLINE_COL.b, OUTLINE_COL.a)
    steps = 16
    for k in range(steps):
        a0 = 2*math.pi * k / steps
        a1 = 2*math.pi * (k+1) / steps
        rl.rl_vertex3f(wx + math.cos(a0)*r, wy + math.sin(a0)*r, pz + 0.01)
        rl.rl_vertex3f(wx + math.cos(a1)*r, wy + math.sin(a1)*r, pz + 0.01)
    rl.rl_end()

def nearest_pt(pts, wx, wy):
    best_i, best_d = -1, float('inf')
    for i, p in enumerate(pts):
        d = math.hypot(p[0] - wx, p[1] - wy)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d

def nearest_seg_poly(poly, wx, wy):
    best_i, best_d = 0, float('inf')
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        bx, by = poly[i+1]
        dx, dy = bx-ax, by-ay
        ln2 = dx*dx + dy*dy
        if ln2 == 0:
            continue
        t = max(0.0, min(1.0, ((wx-ax)*dx + (wy-ay)*dy) / ln2))
        d = math.hypot(ax + t*dx - wx, ay + t*dy - wy)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d

def full_poly(pts):
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

def hovered_spline_idx(splines, full_polys, wx, wy, threshold=40.0):
    for si in range(len(splines) - 1, -1, -1):
        if len(full_polys[si]) < 2:
            continue
        _, d = nearest_seg_poly(full_polys[si], wx, wy)
        if d < threshold:
            return si
    return -1

def insert_near(pts, fp, wx, wy):
    if len(fp) < 2:
        pts.append((wx, wy, 0))
        return
    seg_i, _ = nearest_seg_poly(fp, wx, wy)
    insert_at = min(seg_i // SEGMENTS + 1, len(pts))
    z = pts[min(insert_at, len(pts)-1)][2]
    pts.insert(insert_at, (wx, wy, z))

def lighten(c, a=80):
    return rl.Color(min(255,c.r+a), min(255,c.g+a), min(255,c.b+a), 255)

def rebuild(splines):
    fp = [full_poly(s['pts']) for s in splines]
    draw_segs = []
    for si, s in enumerate(splines):
        for poly, z0, z1 in build_segments(s['pts']):
            wpoly, vnorm = build_poly_and_normals(poly)
            draw_segs.append({'wpoly': wpoly, 'vnorm': vnorm,
                              'z0': z0, 'z1': z1,
                              'si': si, 'thick': s['thick'], 'color': s['color']})
    return fp, draw_segs

def is_endpoint(pts, pi):
    return pi == 0 or pi == len(pts) - 1

def make_camera():
    cam = rl.Camera3D()
    cam.position   = rl.Vector3(0.0, 0.0, CAM_Z)
    cam.target     = rl.Vector3(0.0, 0.0, 0.0)
    cam.up         = rl.Vector3(0.0, 1.0, 0.0)
    cam.fovy       = float(SCREEN_H)
    cam.projection = rl.CAMERA_ORTHOGRAPHIC
    return cam

def main():
    rl.init_window(SCREEN_W, SCREEN_H, "Knot Spline Editor")
    rl.set_target_fps(60)
    rl.rl_disable_backface_culling()

    camera = make_camera()

    # pts stored in WORLD space (x right, y up, origin center)
    splines = [
        {'pts': [screen_to_world(400,400)+(0,), screen_to_world(700,300)+(1,),
                 screen_to_world(1000,500)+(0,), screen_to_world(1300,400)+(0,)],
         'color': ROPE_COLORS[0], 'thick': DEFAULT_THICK},
        {'pts': [screen_to_world(400,600)+(0,), screen_to_world(700,700)+(0,),
                 screen_to_world(1000,500)+(1,), screen_to_world(1300,650)+(0,)],
         'color': ROPE_COLORS[1], 'thick': DEFAULT_THICK},
    ]

    full_polys, draw_segs = rebuild(splines)
    drag_s        = -1
    drag_p        = -1
    hov_s         = -1
    hov_p         = -1
    dirty         = True
    extend_active = False

    while not rl.window_should_close():
        msx = float(rl.get_mouse_x())
        msy = float(rl.get_mouse_y())
        # convert mouse to world space once — all hit-testing uses world coords
        mx, my = screen_to_world(msx, msy)
        shift = rl.is_key_down(rl.KEY_LEFT_SHIFT) or rl.is_key_down(rl.KEY_RIGHT_SHIFT)

        if dirty:
            full_polys, draw_segs = rebuild(splines)
            dirty = False

        if drag_s == -1:
            hov_s = hovered_spline_idx(splines, full_polys, mx, my)
            hov_p = -1
            if hov_s >= 0:
                bi, bd = nearest_pt(splines[hov_s]['pts'], mx, my)
                if bd < HOVER_R * 2.5:
                    hov_p = bi

        if hov_s >= 0 and hov_p >= 0:
            p = splines[hov_s]['pts'][hov_p]
            if rl.is_key_pressed(rl.KEY_W):
                splines[hov_s]['pts'][hov_p] = (p[0], p[1], p[2] + 1)
                dirty = True
            if rl.is_key_pressed(rl.KEY_S):
                splines[hov_s]['pts'][hov_p] = (p[0], p[1], p[2] - 1)
                dirty = True

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

        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
            extend_active = False
            for si in range(len(splines)-1, -1, -1):
                bi, bd = nearest_pt(splines[si]['pts'], mx, my)
                if bd < HOVER_R * 2:
                    if shift and is_endpoint(splines[si]['pts'], bi):
                        pts = splines[si]['pts']
                        new_pt = (mx, my, pts[bi][2])
                        if bi == 0:
                            pts.insert(0, new_pt)
                            drag_p = 0
                        else:
                            pts.append(new_pt)
                            drag_p = len(pts) - 1
                        drag_s = si
                        extend_active = True
                        dirty = True
                    else:
                        drag_s = si
                        drag_p = bi
                    break

        if rl.is_mouse_button_released(rl.MOUSE_BUTTON_LEFT):
            drag_s = -1
            drag_p = -1
            extend_active = False

        if drag_s >= 0:
            # delta is in screen pixels; y must be flipped for world space
            delta = rl.get_mouse_delta()
            p = splines[drag_s]['pts'][drag_p]
            splines[drag_s]['pts'][drag_p] = (p[0] + delta.x, p[1] - delta.y, p[2])
            dirty = True

        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_RIGHT):
            t = hov_s if hov_s >= 0 else 0
            sp = splines[t]
            if shift:
                if len(sp['pts']) > 2:
                    bi, bd = nearest_pt(sp['pts'], mx, my)
                    if bd < HOVER_R * 3 and not is_endpoint(sp['pts'], bi):
                        sp['pts'].pop(bi)
                        dirty = True
            else:
                insert_near(sp['pts'], full_polys[t], mx, my)
                dirty = True

        rl.begin_drawing()
        rl.clear_background(BG)

        rl.begin_mode_3d(camera)

        for seg in draw_segs:
            half_out = seg['thick'] * 0.5 + OUTLINE_PAD
            half_in  = seg['thick'] * 0.5
            draw_strip_3d(seg['wpoly'], seg['vnorm'], half_out,
                          seg['color'], seg['z0'], seg['z1'], True)
            draw_strip_3d(seg['wpoly'], seg['vnorm'], half_in,
                          seg['color'], seg['z0'], seg['z1'], False)

        for si, sp in enumerate(splines):
            pt_col = lighten(sp['color'], 80)
            for pi, p in enumerate(sp['pts']):
                wx, wy, pz = p
                is_end = is_endpoint(sp['pts'], pi)
                r = HOVER_R * 1.4 if (si == drag_s and pi == drag_p) else HOVER_R
                fill = lighten(pt_col, 40) if is_end else pt_col
                draw_point_3d(wx, wy, pz + 0.5, r, fill)

        rl.end_mode_3d()

        rl.draw_text(
            "LMB drag | Shift+LMB endpoint=extend | RMB add | Shift+RMB del | QE order | AD thick | WS pt-z",
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
