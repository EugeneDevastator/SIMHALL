"""
wgpu transparent quads demo  (wgpu 0.31+ / rendercanvas)
A = add 1000 random quads
D = remove 1000 quads
ESC = quit

Requirements:
    pip install wgpu rendercanvas glfw numpy
"""

import time
import numpy as np
import wgpu
import wgpu.backends.auto
import glfw

from rendercanvas.glfw import RenderCanvas, loop

# ── canvas & device ──────────────────────────────────────────────────────────

canvas  = RenderCanvas(title="wgpu quads",
                       size=(900, 700), update_mode="continuous")
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device  = adapter.request_device_sync()
context = canvas.get_context("wgpu")
fmt     = context.get_preferred_format(adapter)
context.configure(device=device, format=fmt)

# ── shaders ──────────────────────────────────────────────────────────────────

SHADER = """
struct Quad {
    pos   : vec2f,
    size  : vec2f,
    color : vec4f,
};

@group(0) @binding(0) var<storage, read> quads : array<Quad>;

struct VOut {
    @builtin(position) pos : vec4f,
    @location(0)       col : vec4f,
};

const CORNERS = array<vec2f, 6>(
    vec2f(-1,  1), vec2f( 1,  1), vec2f(-1, -1),
    vec2f(-1, -1), vec2f( 1,  1), vec2f( 1, -1),
);

@vertex
fn vs_main(
    @builtin(vertex_index)   vi : u32,
    @builtin(instance_index) ii : u32,
) -> VOut {
    let q  = quads[ii];
    let p  = q.pos + CORNERS[vi] * q.size;
    var out : VOut;
    out.pos = vec4f(p, 0.0, 1.0);
    out.col = q.color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    return in.col;
}
"""

shader_module = device.create_shader_module(code=SHADER)

# ── bind group layout ─────────────────────────────────────────────────────────

bgl = device.create_bind_group_layout(entries=[{
    "binding":    0,
    "visibility": wgpu.ShaderStage.VERTEX,
    "buffer":     {"type": wgpu.BufferBindingType.read_only_storage},
}])

pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

# ── render pipeline ───────────────────────────────────────────────────────────

render_pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={
        "module":      shader_module,
        "entry_point": "vs_main",
    },
    fragment={
        "module":      shader_module,
        "entry_point": "fs_main",
        "targets": [{
            "format": fmt,
            "blend": {
                "color": {
                    "src_factor": wgpu.BlendFactor.src_alpha,
                    "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                    "operation":  wgpu.BlendOperation.add,
                },
                "alpha": {
                    "src_factor": wgpu.BlendFactor.one,
                    "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                    "operation":  wgpu.BlendOperation.add,
                },
            },
        }],
    },
    primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    depth_stencil=None,
    multisample=None,
)

# ── quad storage ──────────────────────────────────────────────────────────────

QUAD_STRIDE  = 8
MAX_QUADS    = 200_000
quad_data    = np.zeros((MAX_QUADS, QUAD_STRIDE), dtype=np.float32)
quad_vel     = np.zeros((MAX_QUADS, 2), dtype=np.float32)   # vx, vy per quad
quad_count   = 0

COLORS = [
    (0.95, 0.27, 0.27),
    (0.27, 0.62, 0.95),
    (0.35, 0.85, 0.55),
    (0.95, 0.75, 0.25),
    (0.75, 0.35, 0.95),
    (0.95, 0.55, 0.25),
    (0.25, 0.85, 0.85),
    (0.95, 0.35, 0.75),
]

BUFFER_BYTES = MAX_QUADS * QUAD_STRIDE * 4

storage_buf = device.create_buffer(
    size=max(BUFFER_BYTES, 64),
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
)

bind_group = device.create_bind_group(
    layout=bgl,
    entries=[{
        "binding":  0,
        "resource": {
            "buffer": storage_buf,
            "offset": 0,
            "size":   BUFFER_BYTES,
        },
    }],
)


def upload_quads():
    if quad_count == 0:
        return
    device.queue.write_buffer(
    storage_buf, 0,
    quad_data,          # pass the array directly
    0,
    quad_count * QUAD_STRIDE * 4   # byte count
)


def add_quads(n=1000):
    global quad_count
    n = min(n, MAX_QUADS - quad_count)
    if n <= 0:
        return
    palette = np.array(COLORS, dtype=np.float32)
    idx  = np.random.randint(0, len(COLORS), n)
    rows = np.empty((n, 8), dtype=np.float32)
    rows[:, 0]   = np.random.uniform(-0.95,  0.95, n)
    rows[:, 1]   = np.random.uniform(-0.95,  0.95, n)
    rows[:, 2]   = np.random.uniform(0.02,   0.12, n)
    rows[:, 3]   = np.random.uniform(0.02,   0.10, n)
    rows[:, 4:7] = palette[idx]
    rows[:, 7]   = np.random.uniform(0.01,   0.1, n)
    quad_data[quad_count:quad_count + n] = rows

    # velocities: random direction, speed ~0.3 units/sec in NDC
    speed = np.random.uniform(0.1, 0.4, n)
    angle = np.random.uniform(0, 2 * np.pi, n)
    quad_vel[quad_count:quad_count + n, 0] = np.cos(angle) * speed
    quad_vel[quad_count:quad_count + n, 1] = np.sin(angle) * speed

    quad_count += n
    upload_quads()
    print(f"  quads: {quad_count:,}")


def remove_quads(n=1000):
    global quad_count
    quad_count = max(0, quad_count - n)
    print(f"  quads: {quad_count:,}")


def update_quads(dt):
    if quad_count == 0:
        return
    pos = quad_data[:quad_count, 0:2]
    sz  = quad_data[:quad_count, 2:4]
    vel = quad_vel[:quad_count]

    pos += vel * dt

    # bounce: left/right walls
    over_r = pos[:, 0] + sz[:, 0] >  1.0
    over_l = pos[:, 0] - sz[:, 0] < -1.0
    vel[over_r | over_l, 0] *= -1.0
    pos[:, 0] = np.clip(pos[:, 0], -1.0 + sz[:, 0], 1.0 - sz[:, 0])

    # bounce: top/bottom walls
    over_t = pos[:, 1] + sz[:, 1] >  1.0
    over_b = pos[:, 1] - sz[:, 1] < -1.0
    vel[over_t | over_b, 1] *= -1.0
    pos[:, 1] = np.clip(pos[:, 1], -1.0 + sz[:, 1], 1.0 - sz[:, 1])

    upload_quads()


# ── keyboard via glfw ─────────────────────────────────────────────────────────

def on_key(window, key, scancode, action, mods):
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    if   key == glfw.KEY_A:      add_quads(1000)
    elif key == glfw.KEY_D:      remove_quads(1000)
    elif key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, True)

glfw.set_key_callback(canvas._window, on_key)

# ── timing ────────────────────────────────────────────────────────────────────

_last_time   = time.perf_counter()
_fps_time    = _last_time
_fps_frames  = 0

# ── draw callback ─────────────────────────────────────────────────────────────

def draw_frame():
    global _last_time, _fps_time, _fps_frames

    now = time.perf_counter()
    dt  = now - _last_time
    _last_time = now

    # fps title update every 0.5 s
    _fps_frames += 1
    if now - _fps_time >= 0.5:
        fps = _fps_frames / (now - _fps_time)
        glfw.set_window_title(
            canvas._window,
            f"wgpu quads  |  {fps:.0f} FPS  |  {quad_count:,} quads  |  A=+1000  D=-1000  ESC=quit"
        )
        _fps_time   = now
        _fps_frames = 0

    update_quads(dt)

    texture = context.get_current_texture()
    view    = texture.create_view()

    encoder = device.create_command_encoder()
    rp = encoder.begin_render_pass(
        color_attachments=[{
            "view":           view,
            "resolve_target": None,
            "load_op":        wgpu.LoadOp.clear,
            "store_op":       wgpu.StoreOp.store,
            "clear_value":    (0.08, 0.08, 0.10, 1.0),
        }],
    )
    rp.set_pipeline(render_pipeline)
    rp.set_bind_group(0, bind_group)
    if quad_count > 0:
        rp.draw(6, quad_count)
    rp.end()
    device.queue.submit([encoder.finish()])


# ── go ────────────────────────────────────────────────────────────────────────

print("Controls:  A = +1000 quads   D = -1000 quads   ESC = quit")
add_quads(1000)
canvas.request_draw(draw_frame)
loop.run()
