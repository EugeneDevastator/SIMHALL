"""
Tree simulator with progressively growing tree segment and branch-inherited growth parameters.
"""
import pyray as rl
import math
import ctypes

# --- Constants ---
SCREEN_W = 1100
SCREEN_H = 600
TREE_W = 800
TREE_H = 600
PANEL_X = 810
PANEL_W = 280
GROWTH_SPEED = 2.0
MAIN_DEFLECTION = math.pi / 36
BRANCH_ANGLE = math.pi / 3
CURVE_X = PANEL_X + 10
CURVE_Y = 60
CURVE_W = 260
CURVE_H = 160

# --- Data Structures ---
class Node:
    __slots__ = ('x','y','id','is_endpoint','is_root','branch_accumulator','branch_side','birth_time')
    def __init__(self, x, y, nid, is_endpoint=False, is_root=False, birth_time=0):
        self.x = x
        self.y = y
        self.id = nid
        self.is_endpoint = is_endpoint
        self.is_root = is_root
        self.branch_accumulator = 0.0
        self.branch_side = 1
        self.birth_time = birth_time

class Edge:
    __slots__ = ('from_id','to_id','angle')
    def __init__(self, from_id, to_id, angle):
        self.from_id = from_id
        self.to_id = to_id
        self.angle = angle

class TreeState:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_id_counter = 0
        self.max_height = 0.0
        self.global_time = 0
        self.max_branches = 50
        self.growing = False
        self.grow_timer = 0.0
        self.grow_interval = 0.1  # seconds

    def node_by_id(self, nid):
        for n in self.nodes:
            if n.id == nid:
                return n
        return None

# --- Tree Management ---
def tree_init(state):
    state.nodes.clear()
    state.edges.clear()
    state.node_id_counter = 0
    state.max_height = 0.0
    state.global_time = 0

    root = Node(TREE_W / 2, TREE_H - 50, state.node_id_counter, False, True, 0)
    state.node_id_counter += 1
    tip = Node(TREE_W / 2, TREE_H - 50, state.node_id_counter, True, False, 0)
    state.node_id_counter += 1
    state.nodes.append(root)
    state.nodes.append(tip)
    state.edges.append(Edge(root.id, tip.id, 0.0))

def tree_get_branch_rate(curve, age_ratio):
    for i in range(len(curve) - 1):
        if curve[i][0] <= age_ratio <= curve[i+1][0]:
            t = (age_ratio - curve[i][0]) / (curve[i+1][0] - curve[i][0])
            return curve[i][1] + t * (curve[i+1][1] - curve[i][1])
    return curve[-1][1]

def tree_step(state, curve):
    state.global_time += 1

    if len(state.nodes) >= state.max_branches:
        return

    # Build lookup
    node_map = {n.id: n for n in state.nodes}

    # Move non-endpoint, non-root children along their edge direction
    for edge in state.edges:
        parent = node_map.get(edge.from_id)
        child = node_map.get(edge.to_id)
        if parent is None or child is None:
            continue
        if parent.is_root or parent.is_endpoint:
            continue
        dx = math.sin(edge.angle) * GROWTH_SPEED / 3
        dy = -math.cos(edge.angle) * GROWTH_SPEED / 3
        child.x += dx
        child.y += dy

    # Build parent-edge lookup for endpoints
    to_edge = {e.to_id: e for e in state.edges}

    endpoints = [n for n in state.nodes if n.is_endpoint]
    new_nodes = []
    new_edges = []

    for ep in endpoints:
        parent_edge = to_edge.get(ep.id)
        angle = parent_edge.angle if parent_edge else 0.0

        dx = math.sin(angle) * GROWTH_SPEED * 2
        dy = -math.cos(angle) * GROWTH_SPEED * 2
        ep.x += dx
        ep.y += dy

        height = (TREE_H - 50) - ep.y
        if height > state.max_height:
            state.max_height = height

        age = state.global_time - ep.birth_time
        age_ratio = min(1.0, age / state.global_time) if state.global_time > 0 else 0.0
        rate = tree_get_branch_rate(curve, age_ratio)
        ep.branch_accumulator += rate

        total = len(state.nodes) + len(new_nodes)
        if total < state.max_branches and ep.branch_accumulator >= 1.0 and ep.y > 50:
            ep.branch_accumulator -= 1.0
            ep.is_endpoint = False

            main_angle = angle + ep.branch_side * MAIN_DEFLECTION
            branch_angle = angle + ep.branch_side * BRANCH_ANGLE

            mn = Node(ep.x, ep.y, state.node_id_counter, True, False, state.global_time)
            state.node_id_counter += 1
            bn = Node(ep.x, ep.y, state.node_id_counter, True, False, state.global_time)
            state.node_id_counter += 1

            mn.branch_side = ep.branch_side * -1
            bn.branch_side = ep.branch_side * -1

            new_nodes.append(mn)
            new_nodes.append(bn)
            new_edges.append(Edge(ep.id, mn.id, main_angle))
            new_edges.append(Edge(ep.id, bn.id, branch_angle))

            if len(state.nodes) + len(new_nodes) >= state.max_branches:
                break

    state.nodes.extend(new_nodes)
    state.edges.extend(new_edges)

# --- Rendering ---
def draw_tree(state):
    # Background
    rl.draw_rectangle(0, 0, TREE_W, TREE_H, rl.Color(10, 10, 10, 255))

    node_map = {n.id: n for n in state.nodes}

    # Edges
    for edge in state.edges:
        a = node_map.get(edge.from_id)
        b = node_map.get(edge.to_id)
        if a and b:
            rl.draw_line(int(a.x), int(a.y), int(b.x), int(b.y), rl.Color(139, 69, 19, 255))

    # Nodes
    for n in state.nodes:
        if n.is_root:
            rl.draw_circle(int(n.x), int(n.y), 5, rl.Color(170, 68, 68, 255))
        elif n.is_endpoint:
            rl.draw_circle(int(n.x), int(n.y), 4, rl.Color(68, 170, 68, 255))
        else:
            rl.draw_circle(int(n.x), int(n.y), 2, rl.Color(68, 170, 68, 255))

def draw_panel(state, curve, dragging_point):
    rl.draw_rectangle(PANEL_X, 0, PANEL_W, SCREEN_H, rl.Color(42, 42, 42, 255))

    # Title
    rl.draw_text("Branching Rate Curve", PANEL_X + 10, 10, 14, rl.WHITE)
    rl.draw_text("Edge Age vs Spawn Rate", PANEL_X + 10, 30, 11, rl.Color(170,170,170,255))

    draw_curve(curve, dragging_point)

    # Stats
    branch_count = len(state.nodes)
    height_m = state.max_height / 100.0
    stats = f"Branches: {branch_count}  Height: {height_m:.2f}m"
    rl.draw_text(stats, PANEL_X + 10, CURVE_Y + CURVE_H + 10, 12, rl.Color(170,170,170,255))

    # Max branches label
    rl.draw_text(f"Max Branches: {state.max_branches}", PANEL_X + 10, CURVE_Y + CURVE_H + 30, 13, rl.WHITE)

    # Slider bar (manual)
    slider_x = PANEL_X + 10
    slider_y = CURVE_Y + CURVE_H + 50
    slider_w = CURVE_W
    slider_h = 14
    rl.draw_rectangle(slider_x, slider_y, slider_w, slider_h, rl.Color(60,60,60,255))
    t = (state.max_branches - 5) / (200 - 5)
    thumb_x = int(slider_x + t * slider_w)
    rl.draw_rectangle(thumb_x - 5, slider_y - 2, 10, slider_h + 4, rl.Color(150,150,150,255))

    # Buttons
    btn_y = slider_y + 30
    draw_button("Step Forward", PANEL_X + 10, btn_y, CURVE_W, 30)
    draw_button("Auto Grow" if not state.growing else "Stop", PANEL_X + 10, btn_y + 38, CURVE_W, 30)
    draw_button("Reset", PANEL_X + 10, btn_y + 76, CURVE_W, 30)

def draw_button(label, x, y, w, h):
    rl.draw_rectangle(x, y, w, h, rl.Color(74, 74, 74, 255))
    rl.draw_text(label, x + 8, y + 8, 13, rl.WHITE)

def draw_curve(curve, dragging_point):
    # Background
    rl.draw_rectangle(CURVE_X, CURVE_Y, CURVE_W, CURVE_H, rl.Color(26, 26, 26, 255))

    # Grid
    for i in range(11):
        gx = CURVE_X + int(i / 10 * CURVE_W)
        gy = CURVE_Y + int(i / 10 * CURVE_H)
        rl.draw_line(gx, CURVE_Y, gx, CURVE_Y + CURVE_H, rl.Color(68,68,68,255))
        rl.draw_line(CURVE_X, gy, CURVE_X + CURVE_W, gy, rl.Color(68,68,68,255))

    # Curve line
    for i in range(len(curve) - 1):
        ax = CURVE_X + int(curve[i][0] * CURVE_W)
        ay = CURVE_Y + int(curve[i][1] * CURVE_H)
        bx = CURVE_X + int(curve[i+1][0] * CURVE_W)
        by = CURVE_Y + int(curve[i+1][1] * CURVE_H)
        rl.draw_line(ax, ay, bx, by, rl.Color(68, 170, 68, 255))

    # Points
    for i, pt in enumerate(curve):
        px = CURVE_X + int(pt[0] * CURVE_W)
        py = CURVE_Y + int(pt[1] * CURVE_H)
        col = rl.Color(68, 170, 255, 255) if i != dragging_point else rl.Color(255, 200, 50, 255)
        rl.draw_circle(px, py, 6, col)

# --- Input Handling ---
def handle_curve_drag(curve, dragging_point):
    mx = rl.get_mouse_x()
    my = rl.get_mouse_y()

    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
        for i, pt in enumerate(curve):
            px = CURVE_X + int(pt[0] * CURVE_W)
            py = CURVE_Y + int(pt[1] * CURVE_H)
            dist = math.sqrt((px - mx)**2 + (py - my)**2)
            if dist < 10:
                return i

    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
        return None

    if dragging_point is not None:
        rel_y = (my - CURVE_Y) / CURVE_H
        rel_y = max(0.0, min(1.0, rel_y))
        curve[dragging_point] = (curve[dragging_point][0], rel_y)

    return dragging_point

def handle_slider(state):
    mx = rl.get_mouse_x()
    my = rl.get_mouse_y()
    slider_x = PANEL_X + 10
    slider_y = CURVE_Y + CURVE_H + 50
    slider_w = CURVE_W
    slider_h = 14

    if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
        if slider_x <= mx <= slider_x + slider_w and slider_y - 4 <= my <= slider_y + slider_h + 4:
            t = (mx - slider_x) / slider_w
            t = max(0.0, min(1.0, t))
            state.max_branches = int(5 + t * (200 - 5))

def handle_buttons(state):
    mx = rl.get_mouse_x()
    my = rl.get_mouse_y()
    btn_y = CURVE_Y + CURVE_H + 80

    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
        if PANEL_X + 10 <= mx <= PANEL_X + 10 + CURVE_W:
            if btn_y <= my <= btn_y + 30:
                return 'step'
            if btn_y + 38 <= my <= btn_y + 68:
                return 'auto'
            if btn_y + 76 <= my <= btn_y + 106:
                return 'reset'
    return None

# --- Main ---
def main():
    rl.init_window(SCREEN_W, SCREEN_H, "Tree Generator 2D")
    rl.set_target_fps(60)

    curve = [
        (0.0,  0.98),
        (0.25, 0.85),
        (0.5,  0.88),
        (0.75, 0.92),
        (1.0,  0.99),
    ]

    state = TreeState()
    tree_init(state)
    dragging_point = None

    while not rl.window_should_close():
        dt = rl.get_frame_time()

        # Auto grow timer
        if state.growing:
            state.grow_timer += dt
            if state.grow_timer >= state.grow_interval:
                state.grow_timer = 0.0
                if len(state.nodes) < state.max_branches:
                    tree_step(state, curve)
                else:
                    state.growing = False

        dragging_point = handle_curve_drag(curve, dragging_point)
        handle_slider(state)
        action = handle_buttons(state)

        if action == 'step':
            tree_step(state, curve)
        elif action == 'auto':
            state.growing = not state.growing
            state.grow_timer = 0.0
        elif action == 'reset':
            state.growing = False
            tree_init(state)

        rl.begin_drawing()
        rl.clear_background(rl.Color(26, 26, 26, 255))
        draw_tree(state)
        draw_panel(state, curve, dragging_point)
        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()
