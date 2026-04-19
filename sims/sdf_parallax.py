"""
SDF Volume Demo - raylib + Python  (Windows / Linux / macOS)
POM-style SDF shell texture on a rotating quad.

Controls:
  WASD  - rotate quad
  Q/E   - depth scale (how thick the shell appears)
  R/F   - SDF hit threshold
  ESC   - quit

OS compatibility:
  No platform GL library is loaded directly (no find_library("GL")).
  Raw GL functions needed for 3D texture upload are resolved via
  glfwGetProcAddress() — already exposed through raylib's own cffi binding —
  which works identically on Windows (opengl32 + wgl), Linux (libGL + glX/EGL)
  and macOS (OpenGL.framework).  Everything else uses rlgl wrappers.
"""

import ctypes
import numpy as np

try:
    import pyray
    import pyray as rl
    from pyray import *
    import raylib._raylib_cffi as _rlib   # cffi lib for glfwGetProcAddress
except ImportError:
    raise ImportError("pip install raylib")

# ── GL constants ─────────────────────────────────────────────────────────────
# All hardcoded from the OpenGL spec — identical on every platform/OS.
# We do NOT use rl.RL_* here: those live on the pyray module object, not on
# the cffi lib, and are unavailable before the window is created on some
# pyray builds.  The numeric values are guaranteed by the GL specification.
GL_TEXTURE_2D      = 0x0DE1
GL_TEXTURE_3D      = 0x806F
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S  = 0x2802
GL_TEXTURE_WRAP_T  = 0x2803
GL_TEXTURE_WRAP_R  = 0x8072
GL_LINEAR          = 0x2601
GL_NEAREST         = 0x2600
GL_CLAMP_TO_EDGE   = 0x812F
GL_R8              = 0x8229
GL_RED             = 0x1903
GL_RGBA            = 0x1908
GL_RGBA8           = 0x8058
GL_UNSIGNED_BYTE   = 0x1401
GL_TEXTURE0        = 0x84C0   # GL_TEXTURE0+n = texture unit n

# ── Lazy GL proc loader ───────────────────────────────────────────────────────
# Must be called AFTER init_window() so the GL context exists.
_gl_cache: dict = {}

def _gl(name: str, restype, *argtypes):
    """
    Fetch a GL function by name via glfwGetProcAddress.
    Works on Windows, Linux and macOS without find_library().
    Results are cached so the address is only resolved once.
    """
    if name in _gl_cache:
        return _gl_cache[name]
    ffi  = _rlib.ffi
    ptr  = _rlib.lib.glfwGetProcAddress(name.encode())
    addr = int(ffi.cast("uintptr_t", ptr))
    if addr == 0:
        raise RuntimeError(
            f"glfwGetProcAddress could not find {name!r}. "
            "Ensure a valid OpenGL context exists (call after init_window)."
        )
    fn = ctypes.CFUNCTYPE(restype, *argtypes)(addr)
    _gl_cache[name] = fn
    return fn

def _glActiveTexture():
    return _gl("glActiveTexture", None, ctypes.c_uint)

def _glGenTextures():
    return _gl("glGenTextures", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def _glBindTexture():
    return _gl("glBindTexture", None, ctypes.c_uint, ctypes.c_uint)

def _glTexParameteri():
    return _gl("glTexParameteri", None, ctypes.c_uint, ctypes.c_uint, ctypes.c_int)

def _glTexImage3D():
    return _gl("glTexImage3D", None,
               ctypes.c_uint, ctypes.c_int,  ctypes.c_int,
               ctypes.c_int,  ctypes.c_int,  ctypes.c_int,
               ctypes.c_int,  ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)

def _glTexImage2D():
    return _gl("glTexImage2D", None,
               ctypes.c_uint, ctypes.c_int,  ctypes.c_int,
               ctypes.c_int,  ctypes.c_int,  ctypes.c_int,
               ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)

# ── Volume ────────────────────────────────────────────────────────────────────
VOL_W, VOL_H, NUM_SLICES = 128, 128, 16

def generate_parametric_sdf():
    """
    Pyramid crevices: at depth z=0 (surface) openings are largest;
    at z=1 (deep) they taper to a point. SDF encoded as R8: 0.5=surface,
    >0.5=inside hole, <0.5=solid wall.
    """
    rng = np.random.default_rng(7)
    pyramids = []
    for _ in range(7):
        hw = rng.uniform(0.06, 0.18)
        hh = rng.uniform(0.06, 0.18)
        cx = rng.uniform(hw + 0.02, 1.0 - hw - 0.02)
        cy = rng.uniform(hh + 0.02, 1.0 - hh - 0.02)
        pyramids.append((cx, cy, hw, hh))

    xs = (np.arange(VOL_W)      + 0.5) / VOL_W
    ys = (np.arange(VOL_H)      + 0.5) / VOL_H
    zs = (np.arange(NUM_SLICES) + 0.5) / NUM_SLICES

    X = xs[np.newaxis, np.newaxis, :]
    Y = ys[np.newaxis, :,          np.newaxis]
    Z = zs[:,          np.newaxis, np.newaxis]

    best = np.full((NUM_SLICES, VOL_H, VOL_W), 1e9, dtype=np.float32)
    for (cx, cy, hw_base, hh_base) in pyramids:
        hw = hw_base * (1.0 - Z)
        hh = hh_base * (1.0 - Z)
        dx = np.abs(X - cx) - hw
        dy = np.abs(Y - cy) - hh
        box_sdf = (np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
                   + np.minimum(np.maximum(dx, dy), 0.0))
        best = np.minimum(best, box_sdf)

    inside_d  = max(float(np.abs(best[best <  0]).max(initial=1e-3)), 1e-3)
    outside_d = max(float(best[(best >= 0) & (best < 1e8)].max(initial=1e-3)), 1e-3)
    encoded   = np.where(
        best < 0,
        0.5 + (-best / inside_d)  * 0.5,
        0.5 - ( best / outside_d) * 0.5,
    )
    encoded[best >= 1e8] = 0.0
    return np.clip(encoded, 0.0, 1.0).astype(np.float32)


def upload_texture_3d(sdf):
    """Upload SDF volume as a GL_TEXTURE_3D (R8).  Must be called after init_window."""
    data = np.ascontiguousarray(np.clip(sdf, 0, 1) * 255, dtype=np.uint8)
    tid  = ctypes.c_uint(0)
    _glGenTextures()(1, ctypes.byref(tid))
    _glBindTexture()(GL_TEXTURE_3D, tid.value)
    _glTexParameteri()(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    _glTexParameteri()(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    _glTexParameteri()(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE)
    _glTexParameteri()(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE)
    _glTexParameteri()(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R,     GL_CLAMP_TO_EDGE)
    _glTexImage3D()(GL_TEXTURE_3D, 0, GL_R8,
                   VOL_W, VOL_H, NUM_SLICES,
                   0, GL_RED, GL_UNSIGNED_BYTE,
                   data.ctypes.data_as(ctypes.c_void_p))
    _glBindTexture()(GL_TEXTURE_3D, 0)
    return tid.value


def make_debug_texture(layer_u8):
    """Upload a grayscale layer as an RGBA Texture2D raylib can draw."""
    h, w = layer_u8.shape
    rgba = np.stack([layer_u8, layer_u8, layer_u8,
                     np.full_like(layer_u8, 255)], axis=-1)
    rgba = np.ascontiguousarray(rgba)
    tid  = ctypes.c_uint(0)
    _glGenTextures()(1, ctypes.byref(tid))
    _glBindTexture()(GL_TEXTURE_2D, tid.value)
    _glTexParameteri()(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    _glTexParameteri()(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    _glTexImage2D()(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE,
                   rgba.ctypes.data_as(ctypes.c_void_p))
    _glBindTexture()(GL_TEXTURE_2D, 0)
    t         = Texture2D()
    t.id      = tid.value
    t.width   = w
    t.height  = h
    t.mipmaps = 1
    t.format  = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    return t


def bind_texture_3d_to_slot(tex_id, slot):
    """Bind a GL_TEXTURE_3D to a numbered texture unit via rlgl + raw glBindTexture.
    rl_active_texture_slot calls glActiveTexture(GL_TEXTURE0+slot) cross-platform."""
    _glActiveTexture()(GL_TEXTURE0 + slot)
    _glBindTexture()(GL_TEXTURE_3D, tex_id)


# ── Shaders ───────────────────────────────────────────────────────────────────
VERT_SRC = b"""
#version 330 core
in vec3 vertexPosition;
in vec2 vertexTexCoord;

uniform mat4 mvp;
uniform mat4 uModel;
uniform vec3 uCamPos;

out vec2 vUV;
out vec3 vObjView;

void main() {
    mat3 invRot   = transpose(mat3(uModel));
    vec3 worldPos = (uModel * vec4(vertexPosition, 1.0)).xyz;
    vObjView      = invRot * (uCamPos - worldPos);
    vUV           = vertexTexCoord * 2.0;
    gl_Position   = mvp * vec4(vertexPosition, 1.0);
}
"""

FRAG_SRC = b"""
#version 330 core
in vec2 vUV;
in vec3 vObjView;

uniform sampler3D sdfVol;
uniform float     uDepth;
uniform float     uThresh;
uniform float     uMaxParallax;

out vec4 fragColor;

const int   BASE_STEPS   = 48;
const int   REFINE_STEPS = 8;
const float SURF         = 0.5;
const float FADE_START   = 0.85;

float sd(vec3 p) {
    if (any(lessThan(p.xy, vec2(0.0))) || any(greaterThan(p.xy, vec2(1.0))))
        return -999.0;
    return (texture(sdfVol, p).r - SURF) * 2.0;
}

vec3 sdfNormal(vec3 p) {
    const float e = 0.025;
    return normalize(vec3(
        sd(p+vec3(e,0,0)) - sd(p-vec3(e,0,0)),
        sd(p+vec3(0,e,0)) - sd(p-vec3(0,e,0)),
        sd(p+vec3(0,0,e)) - sd(p-vec3(0,0,e))
    ));
}

void main() {
    float ny = vObjView.y;
    if (ny <= 0.0) {
        fragColor = vec4(vec3(texture(sdfVol, vec3(vUV, 0.0)).r * 0.3), 1.0);
        return;
    }

    vec2  rawRatio   = -vObjView.xz / ny;
    float ratioLen   = length(rawRatio);
    float clampedLen = min(ratioLen, uMaxParallax);
    float fadeT      = smoothstep(FADE_START * uMaxParallax, uMaxParallax, ratioLen);
    float weight     = 1.0 - fadeT;

    vec2 uvPerDepth = (ratioLen > 1e-6)
                      ? (rawRatio / ratioLen) * clampedLen * weight
                      : vec2(0.0);
    vec2 totalShift = uvPerDepth * uDepth;

    int STEPS = int(float(BASE_STEPS) * clamp(clampedLen / 3.0, 1.0, 3.0));

    float dz    = 1.0 / float(STEPS);
    float tPrev = 0.0;
    float tHit  = -1.0;

    for (int i = 1; i <= STEPS; i++) {
        float t = float(i) * dz;
        float d = sd(vec3(vUV + totalShift * t, t));
        if (d < -999.0 + 1.0) break;
        if (d > uThresh) { tHit = t; break; }
        tPrev = t;
    }

    if (tHit > 0.0) {
        float tLo = tPrev, tHi = tHit;
        for (int r = 0; r < REFINE_STEPS; r++) {
            float tMid = (tLo + tHi) * 0.5;
            if (sd(vec3(vUV + totalShift * tMid, tMid)) > uThresh) tHi = tMid;
            else tLo = tMid;
        }
        tHit = (tLo + tHi) * 0.5;
    }

    if (tHit < 0.0) {
        float raw  = texture(sdfVol, vec3(vUV, 0.0)).r;
        vec2  gf   = abs(fract(vUV * 8.0) - 0.5);
        float grid = step(0.45, max(gf.x, gf.y));
        vec3  col  = mix(vec3(0.3,0.0,0.0), vec3(0.0,0.3,0.0), raw*2.0);
        fragColor  = vec4(col + vec3(grid*0.1), 1.0);
        return;
    }

    vec3  hitP   = vec3(vUV + totalShift * tHit, tHit);
    vec3  n      = normalize(mix(sdfNormal(hitP), vec3(0.0,1.0,0.0), fadeT));
    vec3  vDir   = normalize(vObjView);
    vec3  l      = normalize(vec3(0.5, 1.0, 0.3));
    float diff   = max(dot(n, l), 0.0);
    float spec   = pow(max(dot(reflect(-l, n), vDir), 0.0), 48.0);
    float ao     = 1.0 - tHit * 0.65;
    vec3  col    = mix(vec3(0.25,0.50,0.95), vec3(1.0,0.78,0.38), 1.0 - tHit);
    fragColor    = vec4(col * (0.08 + diff*0.82 + spec*0.28) * ao, 1.0);
}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_rot_matrix(rx, ry):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    return np.array([
        [ cy,    0,      sy,    0],
        [ sx*sy, cx,    -sx*cy, 0],
        [-cx*sy, sx,     cx*cy, 0],
        [ 0,     0,      0,     1],
    ], dtype=np.float32)


def np_to_rl_matrix(m):
    return Matrix(
        m[0,0], m[1,0], m[2,0], m[3,0],
        m[0,1], m[1,1], m[2,1], m[3,1],
        m[0,2], m[1,2], m[2,2], m[3,2],
        m[0,3], m[1,3], m[2,3], m[3,3],
    )


def set_float(shader, loc, val):
    p = pyray.ffi.new("float *", float(val))
    set_shader_value(shader, loc, p, SHADER_UNIFORM_FLOAT)


def set_vec3(shader, loc, arr):
    a = np.ascontiguousarray(arr, np.float32)
    p = pyray.ffi.cast("float *", a.ctypes.data)
    set_shader_value(shader, loc, p, SHADER_UNIFORM_VEC3)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating parametric SDF volume...")
    sdf = generate_parametric_sdf()
    print(f"SDF range: [{sdf.min():.3f}, {sdf.max():.3f}]")

    init_window(960, 720, b"SDF Shell | WASD=rotate QE=depth RF=thresh ESC=quit")
    set_target_fps(60)

    # GL proc addresses resolved here — context is guaranteed live after init_window
    tex3d   = upload_texture_3d(sdf)
    bin_tex = make_debug_texture(((sdf[0] > 0.5) * 255).astype(np.uint8))
    sdf_tex = make_debug_texture((np.clip(sdf[0], 0, 1) * 255).astype(np.uint8))
    print(f"3D tex id={tex3d}")

    mesh   = gen_mesh_plane(2.0, 2.0, 4, 4)
    model  = load_model_from_mesh(mesh)
    shader = load_shader_from_memory(VERT_SRC, FRAG_SRC)
    model.materials[0].shader = shader

    loc_sdf    = get_shader_location(shader, "sdfVol")
    loc_model  = get_shader_location(shader, "uModel")
    loc_cam    = get_shader_location(shader, "uCamPos")
    loc_depth  = get_shader_location(shader, "uDepth")
    loc_thresh = get_shader_location(shader, "uThresh")
    loc_maxpar = get_shader_location(shader, "uMaxParallax")

    # Pin the 3D sampler to texture unit 1 (done once; re-bound each frame
    # because raylib's batch flush can disturb texture unit state)
    bind_texture_3d_to_slot(tex3d, 1)
    slot = pyray.ffi.new("int *", 1)
    set_shader_value(shader, loc_sdf, slot, SHADER_UNIFORM_INT)

    cam = Camera3D(
        Vector3(0.0, 3.0, 2.5),
        Vector3(0.0, 0.0, 0.0),
        Vector3(0.0, 1.0, 0.0),
        45.0, CAMERA_PERSPECTIVE,
    )
    cam_arr = np.array([cam.position.x, cam.position.y, cam.position.z], np.float32)

    rot_x, rot_y = 0.0, 0.0
    depth        = 0.3
    thresh       = 0.05
    max_parallax = 8.0

    RSPEED, DSPEED, TSPEED = 1.6, 0.3, 0.04
    DBG = 128

    while not window_should_close():
        dt = get_frame_time()

        if is_key_down(KEY_W): rot_x -= RSPEED * dt
        if is_key_down(KEY_S): rot_x += RSPEED * dt
        if is_key_down(KEY_A): rot_y -= RSPEED * dt
        if is_key_down(KEY_D): rot_y += RSPEED * dt
        if is_key_down(KEY_Q): depth  = max(0.02, depth  - DSPEED * dt)
        if is_key_down(KEY_E): depth  = min(2.0,  depth  + DSPEED * dt)
        if is_key_down(KEY_R): thresh = max(0.001, thresh - TSPEED * dt)
        if is_key_down(KEY_F): thresh = min(0.5,   thresh + TSPEED * dt)

        rot_mat = make_rot_matrix(rot_x, rot_y)
        model.transform = np_to_rl_matrix(rot_mat)

        set_shader_value_matrix(shader, loc_model, np_to_rl_matrix(rot_mat))
        set_vec3(shader, loc_cam, cam_arr)
        set_float(shader, loc_depth,  depth)
        set_float(shader, loc_thresh, thresh)
        set_float(shader, loc_maxpar, max_parallax)

        bind_texture_3d_to_slot(tex3d, 1)  # re-bind after any rl batch flush

        begin_drawing()
        clear_background(Color(10, 10, 14, 255))

        begin_mode_3d(cam)
        draw_model(model, Vector3(0, 0, 0), 1.0, WHITE)
        end_mode_3d()

        pad = 8
        draw_texture_pro(
            bin_tex,
            Rectangle(0, 0, VOL_W, VOL_H),
            Rectangle(pad, pad, DBG, DBG),
            Vector2(0, 0), 0.0, WHITE,
        )
        draw_rectangle_lines(pad, pad, DBG, DBG, Color(200, 200, 100, 200))
        draw_text(b"bin[0]", pad+2, pad+DBG+2, 14, Color(200,200,100,220))

        draw_texture_pro(
            sdf_tex,
            Rectangle(0, 0, VOL_W, VOL_H),
            Rectangle(pad*2+DBG, pad, DBG, DBG),
            Vector2(0, 0), 0.0, WHITE,
        )
        draw_rectangle_lines(pad*2+DBG, pad, DBG, DBG, Color(100, 200, 200, 200))
        draw_text(b"sdf[0]", pad*2+DBG+2, pad+DBG+2, 14, Color(100,200,200,220))

        draw_fps(12, 720-28)
        draw_text(b"WASD=rotate  QE=depth  RF=thresh", 12, 720-50, 16,
                  Color(180,180,180,200))
        draw_text(
            f"depth={depth:.3f}  thresh={thresh:.4f}  "
            f"rot=({np.degrees(rot_x):.0f},{np.degrees(rot_y):.0f})".encode(),
            12, 720-68, 15, Color(130,130,130,200))

        end_drawing()

    close_window()


if __name__ == "__main__":
    main()