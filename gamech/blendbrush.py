import pyray as rl

W, H = 1920, 1080

FRAG = """
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
    vec2 uv = fragTexCoord;
    vec4 dst = texture(texture0, uv);

    vec2 rMin = rectBounds.xy;
    vec2 rMax = rectBounds.xy + rectBounds.zw;

    vec2 center = (rMin + rMax) * 0.5;
    vec2 halfSize = (rMax - rMin) * 0.5;

    vec2 d = (uv - center) / halfSize;
    d.x *= aspect;
    float dist = length(d);

    float mask = 1.0 - smoothstep(0.7, 1.0, dist);

    if (mask < 0.01) {
        finalColor = dst;
        return;
    }

    vec3 noise = vec3(
        hash(uv / 5.0, seed)       * 0.08 - 0.04,
        hash(uv / 5.0, seed + 1.0) * 0.08 - 0.04,
        hash(uv / 5.0, seed + 2.0) * 0.08 - 0.04
    ) * mask;

    finalColor = vec4(clamp(dst.rgb + noise, 0.0, 1.0), dst.a);
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
    return rl.load_render_texture(W, H)

def apply_rect(shader, locs, src_rt, dst_rt, px, py, pw, ph, seed):
    rl.set_shader_value(shader, locs["rectBounds"],
        rl.Vector4(px/W, py/H, pw/W, ph/H),
        rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)

    seed_val = rl.ffi.new("float[1]", [seed])
    rl.set_shader_value(shader, locs["seed"],
        seed_val,
        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    rl.begin_texture_mode(dst_rt)
    rl.begin_shader_mode(shader)
    rl.draw_texture_rec(src_rt.texture, rl.Rectangle(0, 0, W, -H), rl.Vector2(0, 0), rl.WHITE)
    rl.end_shader_mode()
    rl.end_texture_mode()

def main():
    rl.init_window(W, H, "Noise Brush Round")
    rl.set_target_fps(60)

    shader = rl.load_shader_from_memory(VERT, FRAG)

    locs = {
        "rectBounds": rl.get_shader_location(shader, "rectBounds"),
        "seed":       rl.get_shader_location(shader, "seed"),
        "aspect":     rl.get_shader_location(shader, "aspect"),
    }

    aspect_val = rl.ffi.new("float[1]", [W / H])
    rl.set_shader_value(shader, locs["aspect"],
        aspect_val,
        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    rt = [make_rt(), make_rt()]
    cur = 0

    rl.begin_texture_mode(rt[cur])
    rl.clear_background(rl.Color(180, 120, 60, 255))
    rl.draw_circle(W//2, H//2, 300, rl.Color(60, 180, 200, 255))
    rl.end_texture_mode()

    seed_counter = 0.0

    while not rl.window_should_close():
        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
            mp = rl.get_mouse_position()
            rt_y = H - mp.y
            nxt = 1 - cur
            apply_rect(shader, locs, rt[cur], rt[nxt],
                       mp.x - 40, rt_y - 40, 80, 80,
                       seed_counter)
            cur = nxt
            seed_counter += 1.0

        rl.begin_drawing()
        rl.draw_texture_rec(rt[cur].texture, rl.Rectangle(0, 0, W, -H), rl.Vector2(0, 0), rl.WHITE)
        rl.draw_text("Hold LMB to paint", 20, 20, 32, rl.BLACK)
        rl.end_drawing()

    rl.unload_render_texture(rt[0])
    rl.unload_render_texture(rt[1])
    rl.unload_shader(shader)
    rl.close_window()

main()
