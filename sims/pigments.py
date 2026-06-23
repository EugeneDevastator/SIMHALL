import pyray as rl
from pyray import ffi
import math

SW, SH = 940, 680
PANEL_X = 10
PANEL_W = 270
CX = PANEL_X + PANEL_W + 16
CY = 40
CANVAS_W = SW - CX - 10
CANVAS_H = 400

PIGMENT_DEFS = [
    ("Cadmium Red",    (0.89, 0.09, 0.05)),
    ("Cadmium Yellow", (1.00, 0.85, 0.00)),
    ("Phthalo Blue",   (0.00, 0.20, 0.60)),
    ("Phthalo Green",  (0.00, 0.53, 0.33)),
    ("Burnt Sienna",   (0.54, 0.20, 0.07)),
    ("Titanium White", (0.97, 0.97, 0.97)),
    ("Ivory Black",    (0.08, 0.08, 0.08)),
    ("Quinacridone",   (0.85, 0.05, 0.45)),
]

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

uniform vec3 pigments[8];
uniform float w1[8];
uniform float w2[8];

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
vec3 linear_to_srgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0/2.4)) - 0.055, step(0.0031308, c));
}
vec3 linear_to_oklab(vec3 c) {
    float l = 0.4122214708*c.r + 0.5363325363*c.g + 0.0514459929*c.b;
    float m = 0.2119034982*c.r + 0.6806995451*c.g + 0.1073969566*c.b;
    float s = 0.0883024619*c.r + 0.2817188376*c.g + 0.6299787005*c.b;
    vec3 lms = pow(max(vec3(l,m,s), 0.0), vec3(1.0/3.0));
    return vec3(
        0.2104542553*lms.x + 0.7936177850*lms.y - 0.0040720468*lms.z,
        1.9779984951*lms.x - 2.4285922050*lms.y + 0.4505937099*lms.z,
        0.0259040371*lms.x + 0.7827717662*lms.y - 0.8086757660*lms.z
    );
}
vec3 oklab_to_linear(vec3 lab) {
    float l = lab.x + 0.3963377774*lab.y + 0.2158037573*lab.z;
    float m = lab.x - 0.1055613458*lab.y - 0.0638541728*lab.z;
    float s = lab.x - 0.0894841775*lab.y - 1.2914855480*lab.z;
    vec3 lms = pow(max(vec3(l,m,s), 0.0), vec3(3.0));
    return vec3(
         4.0767416621*lms.x - 3.3077115913*lms.y + 0.2309699292*lms.z,
        -1.2684380046*lms.x + 2.6097574011*lms.y - 0.3413193965*lms.z,
        -0.0041960863*lms.x - 0.7034186147*lms.y + 1.7076147010*lms.z
    );
}

void main() {
    // UV: x goes left-right, y goes top-bottom (may be flipped by RT)
    float u = fragTexCoord.x;
    float v = abs(fragTexCoord.y);  // abs handles RT flip

    // ramp: 0 at edges, 1 at center
    float s1 = sin(u * 3.14159265);          // horizontal ramp for layer 1
    float s2 = sin(v * 3.14159265);          // vertical ramp for layer 2

    vec3 paper = linear_to_oklab(vec3(1.0));

    vec3 lab1 = vec3(0.0);
    vec3 lab2 = vec3(0.0);
    float total1 = 0.0, total2 = 0.0;

    for (int i = 0; i < 8; i++) {
        vec3 lab = linear_to_oklab(srgb_to_linear(pigments[i]));
        lab1 += w1[i] * lab;
        lab2 += w2[i] * lab;
        total1 += w1[i];
        total2 += w2[i];
    }

    if (total1 > 0.001) lab1 /= total1;
    if (total2 > 0.001) lab2 /= total2;

    float c1 = (total1 > 0.001) ? s1 : 0.0;
    float c2 = (total2 > 0.001) ? s2 : 0.0;

    // Subtractive-ish blend: composite both layers over paper
    // Use Kubelka-Munk-inspired approach: multiply transmittances
    // Layer opacity = coverage * max_weight (so slider amount controls density)
    float density1 = clamp(total1, 0.0, 1.0);
    float density2 = clamp(total2, 0.0, 1.0);

    float alpha1 = c1 * density1;
    float alpha2 = c2 * density2;

    // Composite: paper -> layer1 -> layer2
    vec3 col = paper;
    col = mix(col, lab1, alpha1);
    col = mix(col, lab2, alpha2);

    vec3 rgb = linear_to_srgb(clamp(oklab_to_linear(col), 0.0, 1.0));
    finalColor = vec4(rgb, 1.0);
}
"""

layer_weights = [
    [0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0],
]
sliders_dragging = [[False]*8, [False]*8]

def get_pigment_color(i):
    r, g, b = PIGMENT_DEFS[i][1]
    return rl.Color(int(r*255), int(g*255), int(b*255), 255)

def set_shader_uniforms(shader):
    flat_pig = []
    for _, (r, g, b) in PIGMENT_DEFS:
        flat_pig += [r, g, b]
    loc = rl.get_shader_location(shader, "pigments")
    arr = ffi.new('float[24]', flat_pig)
    rl.set_shader_value_v(shader, loc, arr, rl.SHADER_UNIFORM_VEC3, 8)
    for layer_idx, name in enumerate(["w1", "w2"]):
        loc2 = rl.get_shader_location(shader, name)
        arr2 = ffi.new('float[8]', layer_weights[layer_idx])
        rl.set_shader_value_v(shader, loc2, arr2, rl.SHADER_UNIFORM_FLOAT, 8)

def main():
    rl.init_window(SW, SH, "Pigment Blender — GPU Oklab")
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

    SLIDER_X = PANEL_X + 10
    SLIDER_W = 170
    layer_labels  = ["Layer 1  — horizontal", "Layer 2  — vertical"]
    layer_ui_cols = [rl.Color(200, 60, 40, 255), rl.Color(40, 90, 200, 255)]

    while not rl.window_should_close():
        mx = rl.get_mouse_x()
        my = rl.get_mouse_y()
        mouse_down    = rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT)
        mouse_pressed = rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT)

        for layer in range(2):
            base_y = 50 + layer * 290
            for i in range(8):
                sy = base_y + 28 + i * 28
                rect = rl.Rectangle(SLIDER_X, sy, SLIDER_W, 16)
                hit = rl.check_collision_point_rec(rl.Vector2(mx, my), rect)
                if mouse_down and (sliders_dragging[layer][i] or (mouse_pressed and hit)):
                    sliders_dragging[layer][i] = True
                    val = (mx - SLIDER_X) / SLIDER_W
                    layer_weights[layer][i] = max(0.0, min(1.0, val))
                elif not mouse_down:
                    sliders_dragging[layer][i] = False

        set_shader_uniforms(shader)

        rl.begin_texture_mode(canvas_rt)
        rl.clear_background(rl.WHITE)
        rl.begin_shader_mode(shader)
        rl.draw_texture_pro(white_tex, src_small, full_quad, origin, 0.0, rl.WHITE)
        rl.end_shader_mode()
        rl.end_texture_mode()

        rl.begin_drawing()
        rl.clear_background(rl.Color(240, 238, 232, 255))

        rl.draw_texture_pro(canvas_rt.texture, src_rt, dst_canvas, origin, 0.0, rl.WHITE)
        rl.draw_rectangle_lines_ex(dst_canvas, 1, rl.Color(160, 160, 160, 255))
        rl.draw_text("Canvas — GPU Oklab pigment blend", CX, CY - 22, 15,
                     rl.Color(100, 100, 100, 255))

        rl.draw_rectangle(PANEL_X, 10, PANEL_W, SH - 20, rl.WHITE)
        rl.draw_rectangle_lines(PANEL_X, 10, PANEL_W, SH - 20, rl.Color(200, 200, 200, 255))

        for layer in range(2):
            base_y = 50 + layer * 290
            rl.draw_text(layer_labels[layer], PANEL_X + 10, base_y, 17, layer_ui_cols[layer])
            for i in range(8):
                sy = base_y + 28 + i * 28
                pc = get_pigment_color(i)
                v  = layer_weights[layer][i]
                rl.draw_rectangle(SLIDER_X, sy, SLIDER_W, 16, rl.Color(220, 220, 220, 255))
                if v > 0.001:
                    rl.draw_rectangle(SLIDER_X, sy, int(v * SLIDER_W), 16, pc)
                rl.draw_rectangle_lines(SLIDER_X, sy, SLIDER_W, 16, rl.Color(160, 160, 160, 255))
                rl.draw_text(PIGMENT_DEFS[i][0], SLIDER_X + SLIDER_W + 6, sy + 1, 13,
                             rl.Color(50, 50, 50, 255))

        rl.end_drawing()

    rl.unload_shader(shader)
    rl.unload_texture(white_tex)
    rl.unload_render_texture(canvas_rt)
    rl.close_window()

if __name__ == "__main__":
    main()
