"""
sample demo for RePaint custom blending and looparounds.
"""
import pyray as rl

W, H = 1920, 1080
CW, CH = 512, 512

FRAG_SMUDGE = """
#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec4 rectBounds;
uniform float seed;
uniform float aspect;
out vec4 finalColor;

float hash(vec2 p, float s) {
    return fract(sin(dot(p, vec2(127.1 + s, 311.7 + s))) * 43758.5453);
}

void main() {
    vec2 uv = fract(fragTexCoord);
    vec4 dst = texture(texture0, uv);
    vec2 center  = rectBounds.xy;
    vec2 halfSize = rectBounds.zw;
    vec2 d = uv - center;
    d = fract(d + 0.5) - 0.5;
    d.x *= aspect;
    float dist = length(d) / length(halfSize * vec2(aspect, 1.0));
    float mask = 1.0 - smoothstep(0.7, 1.0, dist);
    if (mask < 0.01) { finalColor = dst; return; }
    vec2 smudgeOffset = vec2(0.0, halfSize.y * 2.0);
    vec4 smudgeSrc = texture(texture0, fract(uv + smudgeOffset));
    vec3 smudged = mix(dst.rgb, smudgeSrc.rgb, mask * 0.5);
    vec3 noise = vec3(
        hash(uv / 5.0, seed)       * 0.08 - 0.04,
        hash(uv / 5.0, seed + 1.0) * 0.08 - 0.04,
        hash(uv / 5.0, seed + 2.0) * 0.08 - 0.04
    ) * mask;
    finalColor = vec4(clamp(smudged + noise, 0.0, 1.0), dst.a);
}
"""

FRAG_PAINT = """
#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec4 rectBounds;
uniform vec4 paintColor;
uniform float aspect;
out vec4 finalColor;

void main() {
    vec2 uv = fract(fragTexCoord);
    vec4 dst = texture(texture0, uv);
    vec2 center   = rectBounds.xy;
    vec2 halfSize  = rectBounds.zw;
    vec2 d = uv - center;
    d = fract(d + 0.5) - 0.5;
    d.x *= aspect;
    float dist = length(d) / length(halfSize * vec2(aspect, 1.0));
    float mask = 1.0 - smoothstep(0.7, 1.0, dist);
    if (mask < 0.01) { finalColor = dst; return; }
    finalColor = vec4(mix(dst.rgb, paintColor.rgb, mask * paintColor.a), dst.a);
}
"""

VERT = """
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec4 vertexColor;
uniform mat4 mvp;
out vec2 fragTexCoord;
out vec4 fragColor;
void main() {
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

def make_rt():
    rt = rl.load_render_texture(CW, CH)
    # Explicit repeat wrap on both color and depth
    rl.set_texture_wrap(rt.texture, rl.TextureWrap.TEXTURE_WRAP_REPEAT)
    return rt

def screen_to_uv(mx, my, cx, cy):
    return ((mx - cx) / CW) % 1.0, (1.0 - (my - cy) / CH) % 1.0


def apply_smudge(shader, locs, src_rt, dst_rt, cu, cv, seed):
    brush_hu = 40.0 / CW
    brush_hv = 40.0 / CH
    rl.set_shader_value(shader, locs["rectBounds"],
        rl.Vector4(cu, cv, brush_hu, brush_hv),
        rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)
    seed_val = rl.ffi.new("float[1]", [seed])
    rl.set_shader_value(shader, locs["seed"], seed_val,
        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.begin_texture_mode(dst_rt)
    rl.begin_shader_mode(shader)
    rl.draw_texture_rec(src_rt.texture, rl.Rectangle(0, 0, CW, -CH), rl.Vector2(0, 0), rl.WHITE)
    rl.end_shader_mode()
    rl.end_texture_mode()

def apply_paint(shader, locs, src_rt, dst_rt, cu, cv, color):
    brush_hu = 40.0 / CW
    brush_hv = 40.0 / CH
    rl.set_shader_value(shader, locs["rectBounds"],
        rl.Vector4(cu, cv, brush_hu, brush_hv),
        rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)
    rl.set_shader_value(shader, locs["paintColor"],
        rl.Vector4(color.r / 255.0, color.g / 255.0, color.b / 255.0, 0.8),
        rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)
    rl.begin_texture_mode(dst_rt)
    rl.begin_shader_mode(shader)
    rl.draw_texture_rec(src_rt.texture, rl.Rectangle(0, 0, CW, -CH), rl.Vector2(0, 0), rl.WHITE)
    rl.end_shader_mode()
    rl.end_texture_mode()

def draw_tiled(texture, cx, cy, cols, rows):
    # Draw cols x rows grid of the texture to show seamless tiling
    tile_w = CW
    tile_h = CH
    # top-left of the grid, centered on cx,cy being the middle tile
    ox = cx - (cols // 2) * tile_w
    oy = cy - (rows // 2) * tile_h
    for row in range(rows):
        for col in range(cols):
            is_center = (col == cols // 2 and row == rows // 2)
            tint = rl.WHITE if is_center else rl.Color(180, 180, 180, 255)
            rl.draw_texture_rec(
                texture,
                rl.Rectangle(0, 0, CW, -CH),
                rl.Vector2(ox + col * tile_w, oy + row * tile_h),
                tint
            )
    # Draw grid lines
    for row in range(rows + 1):
        rl.draw_line(ox, oy + row * tile_h, ox + cols * tile_w, oy + row * tile_h, rl.Color(255, 255, 255, 60))
    for col in range(cols + 1):
        rl.draw_line(ox + col * tile_w, oy, ox + col * tile_w, oy + rows * tile_h, rl.Color(255, 255, 255, 60))
    # Highlight center tile border
    rl.draw_rectangle_lines(cx, cy, CW, CH, rl.WHITE)

def main():
    rl.init_window(W, H, "Seamless Brush")
    rl.set_target_fps(60)

    smudge_shader = rl.load_shader_from_memory(VERT, FRAG_SMUDGE)
    paint_shader  = rl.load_shader_from_memory(VERT, FRAG_PAINT)

    aspect_val = rl.ffi.new("float[1]", [CW / CH])

    smudge_locs = {
        "rectBounds": rl.get_shader_location(smudge_shader, "rectBounds"),
        "seed":       rl.get_shader_location(smudge_shader, "seed"),
        "aspect":     rl.get_shader_location(smudge_shader, "aspect"),
    }
    rl.set_shader_value(smudge_shader, smudge_locs["aspect"], aspect_val,
        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    paint_locs = {
        "rectBounds": rl.get_shader_location(paint_shader, "rectBounds"),
        "paintColor": rl.get_shader_location(paint_shader, "paintColor"),
        "aspect":     rl.get_shader_location(paint_shader, "aspect"),
    }
    rl.set_shader_value(paint_shader, paint_locs["aspect"], aspect_val,
        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    rt = [make_rt(), make_rt()]
    cur = 0

    # Center tile position — shift left to make room for 3x3 grid
    cx = (W - CW) // 2
    cy = (H - CH) // 2

    rl.begin_texture_mode(rt[cur])
    rl.clear_background(rl.Color(180, 120, 60, 255))
    rl.draw_circle(CW // 2, CH // 2, 120, rl.Color(60, 180, 200, 255))
    rl.end_texture_mode()

    seed_counter = 0.0

    while not rl.window_should_close():
        mp = rl.get_mouse_position()
        cu, cv = screen_to_uv(mp.x, mp.y, cx, cy)
        nxt = 1 - cur

        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
            apply_smudge(smudge_shader, smudge_locs, rt[cur], rt[nxt], cu, cv, seed_counter)
            cur = nxt
            seed_counter += 1.0

        elif rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT):
            apply_paint(paint_shader, paint_locs, rt[cur], rt[nxt], cu, cv, rl.Color(220, 30, 30, 255))
            cur = nxt

        rl.begin_drawing()
        rl.clear_background(rl.Color(40, 40, 40, 255))

        # 3x3 tiled view — center tile is full brightness, neighbors dimmed
        draw_tiled(rt[cur].texture, cx, cy, 3, 3)

        rl.draw_text("LMB: smudge  RMB: paint red  |  3x3 tile shows seamless wrap", 20, 20, 32, rl.WHITE)
        rl.end_drawing()

    rl.unload_render_texture(rt[0])
    rl.unload_render_texture(rt[1])
    rl.unload_shader(smudge_shader)
    rl.unload_shader(paint_shader)
    rl.close_window()

main()
