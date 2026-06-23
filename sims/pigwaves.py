import pyray as rl
from pyray import ffi
import math

SW, SH = 1400, 900
PANEL_W = 380
CX = PANEL_W + 10
CY = 40
CANVAS_W = SW - CX - 10
CANVAS_H = SH - CY - 10

VERT_SHADER = """
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
out vec2 fragTexCoord;
uniform mat4 mvp;
void main() {
    fragTexCoord = vertexTexCoord;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

FRAG_SHADER = """
#version 330
in vec2 fragTexCoord;
out vec4 finalColor;

uniform float waves_h[8];
uniform float waves_v[8];
uniform float hues[8];

const float PI = 3.14159265;

vec3 hue_to_rgb(float h) {
    h = mod(h, 360.0);
    float r = clamp(abs(mod(h/60.0 - 3.0, 6.0) - 3.0) - 1.0, 0.0, 1.0);
    float g = clamp(2.0 - abs(mod(h/60.0 - 2.0, 6.0) - 3.0), 0.0, 1.0);
    float b = clamp(2.0 - abs(mod(h/60.0 - 4.0, 6.0) - 3.0), 0.0, 1.0);
    return vec3(r, g, b);
}

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055)/1.055, vec3(2.4)), step(0.04045, c));
}
vec3 linear_to_srgb(vec3 c) {
    return mix(c * 12.92, 1.055*pow(max(c,0.0), vec3(1.0/2.4)) - 0.055, step(0.0031308, c));
}

void main() {
    float u = fragTexCoord.x;
    float v = abs(fragTexCoord.y);

    float ramp_h = sin(u * PI);
    float ramp_v = sin(v * PI);

    vec3 acc = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        vec3 hue_rgb = srgb_to_linear(hue_to_rgb(hues[i]));
        acc += hue_rgb * waves_h[i] * ramp_h;
        acc += hue_rgb * waves_v[i] * ramp_v;
    }

    acc = clamp(acc, 0.0, 1.0);
    vec3 out_rgb = linear_to_srgb(acc);
    finalColor = vec4(out_rgb, 1.0);
}
"""

ALL_HUES = [i * 45.0 for i in range(8)]

amps_h = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
amps_v = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]

dragging_h = [False] * 8
dragging_v = [False] * 8

FONT_SIZE = 22
VAL_W = 52       # value label column width
SLIDER_H = 18
ITEM_H = 28
SLIDER_X = 8
SLIDER_W = PANEL_W - SLIDER_X - VAL_W - 12

def hue_to_pyray_color(h, brightness=1.0):
    h = h % 360.0
    r = max(0.0, min(1.0, abs(((h/60.0 - 3.0) % 6.0) - 3.0) - 1.0))
    g = max(0.0, min(1.0, 2.0 - abs(((h/60.0 - 2.0) % 6.0) - 3.0)))
    b = max(0.0, min(1.0, 2.0 - abs(((h/60.0 - 4.0) % 6.0) - 3.0)))
    return rl.Color(int(r*255*brightness), int(g*255*brightness), int(b*255*brightness), 255)

def draw_slider(i, sy, amps, dragging, mx, my, mouse_down, mouse_pressed):
    hue = ALL_HUES[i]
    amp = amps[i]
    col = hue_to_pyray_color(hue)
    col_dim = hue_to_pyray_color(hue, 0.45)

    # value label on the left, colored
    val_str = f"{amp:+.2f}"
    rl.draw_text(val_str, SLIDER_X, sy, FONT_SIZE, col)

    sx = SLIDER_X + VAL_W
    rect = rl.Rectangle(sx, sy + 1, SLIDER_W, SLIDER_H)

    hit = rl.check_collision_point_rec(rl.Vector2(mx, my), rect)
    if mouse_down and (dragging[i] or (mouse_pressed and hit)):
        dragging[i] = True
        val = (mx - sx) / SLIDER_W * 2.0 - 1.0
        amps[i] = max(-1.0, min(1.0, val))
        amp = amps[i]
    elif not mouse_down:
        dragging[i] = False

    # track bg
    rl.draw_rectangle(sx, sy + 1, SLIDER_W, SLIDER_H, rl.Color(50, 50, 55, 255))

    cx_px = sx + SLIDER_W // 2
    fill_w = int(amp * (SLIDER_W // 2))
    if fill_w > 0:
        rl.draw_rectangle(cx_px, sy + 1, fill_w, SLIDER_H, col)
    elif fill_w < 0:
        rl.draw_rectangle(cx_px + fill_w, sy + 1, -fill_w, SLIDER_H, col_dim)

    # center line
    rl.draw_line(cx_px, sy + 1, cx_px, sy + 1 + SLIDER_H, rl.Color(130, 130, 130, 255))
    rl.draw_rectangle_lines_ex(rect, 1, rl.Color(75, 75, 80, 255))

    # thumb
    thumb_x = sx + int((amp + 1.0) / 2.0 * SLIDER_W)
    rl.draw_rectangle(thumb_x - 3, sy, 6, SLIDER_H + 2, rl.WHITE)

def set_uniforms(shader):
    loc_h    = rl.get_shader_location(shader, "waves_h")
    loc_v    = rl.get_shader_location(shader, "waves_v")
    loc_hues = rl.get_shader_location(shader, "hues")
    rl.set_shader_value_v(shader, loc_h,    ffi.new('float[8]', amps_h), rl.SHADER_UNIFORM_FLOAT, 8)
    rl.set_shader_value_v(shader, loc_v,    ffi.new('float[8]', amps_v), rl.SHADER_UNIFORM_FLOAT, 8)
    rl.set_shader_value_v(shader, loc_hues, ffi.new('float[8]', ALL_HUES), rl.SHADER_UNIFORM_FLOAT, 8)

def main():
    rl.init_window(SW, SH, "EM Wave Interference")
    rl.set_target_fps(60)

    shader = rl.load_shader_from_memory(VERT_SHADER, FRAG_SHADER)
    canvas_rt = rl.load_render_texture(CANVAS_W, CANVAS_H)

    img = rl.gen_image_color(1, 1, rl.WHITE)
    white_tex = rl.load_texture_from_image(img)
    rl.unload_image(img)

    src_small  = rl.Rectangle(0, 0, 1, 1)
    full_quad  = rl.Rectangle(0, 0, CANVAS_W, CANVAS_H)
    origin     = rl.Vector2(0, 0)
    src_rt     = rl.Rectangle(0, 0, CANVAS_W, -CANVAS_H)
    dst_canvas = rl.Rectangle(CX, CY, CANVAS_W, CANVAS_H)

    # section headers + 8 rows each + separator
    HEADER_H = 28
    SEP_H = 10
    PANEL_CONTENT_H = HEADER_H + 8*ITEM_H + SEP_H + HEADER_H + 8*ITEM_H + 20
    scroll_y = 0

    while not rl.window_should_close():
        mx = rl.get_mouse_x()
        my = rl.get_mouse_y()
        mouse_down    = rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT)
        mouse_pressed = rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)

        wheel = rl.get_mouse_wheel_move()
        if mx < PANEL_W:
            scroll_y -= int(wheel * 30)
            max_scroll = max(0, PANEL_CONTENT_H - SH)
            scroll_y = max(0, min(scroll_y, max_scroll))

        panel_my = my + scroll_y if mx < PANEL_W else my

        set_uniforms(shader)

        rl.begin_texture_mode(canvas_rt)
        rl.clear_background(rl.BLACK)
        rl.begin_shader_mode(shader)
        rl.draw_texture_pro(white_tex, src_small, full_quad, origin, 0.0, rl.WHITE)
        rl.end_shader_mode()
        rl.end_texture_mode()

        rl.begin_drawing()
        rl.clear_background(rl.Color(22, 22, 26, 255))

        rl.begin_scissor_mode(0, 0, PANEL_W, SH)
        rl.draw_rectangle(0, 0, PANEL_W, SH, rl.Color(30, 30, 34, 255))

        oy = -scroll_y

        # --- Layer 1 ---
        rl.draw_text("Horizontal", SLIDER_X, oy + 4, FONT_SIZE, rl.Color(180, 180, 180, 255))
        for i in range(8):
            sy = oy + HEADER_H + i * ITEM_H
            draw_slider(i, sy, amps_h, dragging_h, mx, panel_my, mouse_down, mouse_pressed)

        # --- separator ---
        sep_y = oy + HEADER_H + 8 * ITEM_H + 4
        rl.draw_line(8, sep_y, PANEL_W - 8, sep_y, rl.Color(70, 70, 75, 255))

        # --- Layer 2 ---
        h2y = sep_y + SEP_H
        rl.draw_text("Vertical", SLIDER_X, h2y + 2, FONT_SIZE, rl.Color(180, 180, 180, 255))
        for i in range(8):
            sy = h2y + HEADER_H + i * ITEM_H
            draw_slider(i, sy, amps_v, dragging_v, mx, panel_my, mouse_down, mouse_pressed)

        rl.end_scissor_mode()

        # --- canvas ---
        rl.draw_texture_pro(canvas_rt.texture, src_rt, dst_canvas, origin, 0.0, rl.WHITE)
        rl.draw_rectangle_lines_ex(dst_canvas, 1, rl.Color(70, 70, 75, 255))

        rl.end_drawing()

    rl.unload_shader(shader)
    rl.unload_texture(white_tex)
    rl.unload_render_texture(canvas_rt)
    rl.close_window()

if __name__ == "__main__":
    main()
