import raylib as rl
import numpy as np
from PIL import Image
import sys, os, ctypes
from scipy.ndimage import distance_transform_edt, zoom

W, H = 1920, 1080
FONT_SIZE = 32

# --- OpenGL proc loading (Windows) ---
_gl32 = ctypes.WinDLL("opengl32.dll")
_wglGetProcAddress = _gl32.wglGetProcAddress
_wglGetProcAddress.restype = ctypes.c_void_p
_wglGetProcAddress.argtypes = [ctypes.c_char_p]

def _gl_func(name, restype, argtypes):
    ptr = _wglGetProcAddress(name.encode())
    if not ptr:
        raise RuntimeError(f"wglGetProcAddress failed for {name}")
    proto = ctypes.WINFUNCTYPE(restype, *argtypes)
    return proto(ptr)

# lazily resolved after window+context creation
_glActiveTexture = None
_glBindTexture   = _gl32.glBindTexture
_glBindTexture.restype  = None
_glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]

GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0   = 0x84C0
GL_TEXTURE1   = 0x84C1

def resolve_gl():
    global _glActiveTexture
    _glActiveTexture = _gl_func("glActiveTexture", None, [ctypes.c_uint])

def gl_active_texture(slot):
    _glActiveTexture(slot)

def gl_bind_texture(target, id_):
    _glBindTexture(target, id_)

# --- SDF generation ---
def flood_fill_colors(arr):
    alpha = arr[:, :, 3]
    opaque = alpha > 0
    _, nearest = distance_transform_edt(~opaque, return_indices=True)
    out = arr.copy()
    for c in range(3):
        out[:, :, c] = arr[:, :, c][nearest[0], nearest[1]]
    return out

def generate_sdf(alpha):
    inside = alpha >= 128
    dist_in  = distance_transform_edt(inside)
    dist_out = distance_transform_edt(~inside)
    signed = dist_in - dist_out
    lo, hi = signed.min(), signed.max()
    sdf = (signed - lo) / (hi - lo + 1e-8)
    edge_t = float(-lo / (hi - lo + 1e-8))
    return sdf.astype(np.float32), edge_t

def build_sdf_texture(img_path, downscale_to=128, supersample=4):
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    src_w, src_h = img.size
    arr = flood_fill_colors(arr)

    fw, fh = arr.shape[1], arr.shape[0]

    super_size = downscale_to * supersample
    pil_alpha = Image.fromarray(arr[:, :, 3], 'L')
    super_alpha = np.array(pil_alpha.resize((super_size, super_size), Image.LANCZOS), dtype=np.uint8)
    sdf_super, _ = generate_sdf(super_alpha)

    scale = downscale_to / super_size
    sdf_small_f32 = np.clip(zoom(sdf_super, scale, order=1), 0.0, 1.0).astype(np.float32)

    small_alpha = np.array(pil_alpha.resize((downscale_to, downscale_to), Image.LANCZOS), dtype=np.uint8)
    _, edge_t_small = generate_sdf(small_alpha)

    pil_rgb = Image.fromarray(arr[:, :, :3], 'RGB')
    small_rgb = np.array(pil_rgb.resize((downscale_to, downscale_to), Image.LANCZOS), dtype=np.uint8)

    def pack_rg16(sdf_f32, h, w):
        u16 = (sdf_f32 * 65535.0).astype(np.uint32)
        rg = np.zeros((h, w, 4), dtype=np.uint8)
        rg[:, :, 0] = (u16 >> 8) & 0xFF
        rg[:, :, 1] = u16 & 0xFF
        rg[:, :, 3] = 255
        return rg

    small_rg = pack_rg16(sdf_small_f32, downscale_to, downscale_to)

    return (small_rg, small_rgb, downscale_to, downscale_to, edge_t_small, src_w, src_h)

def make_png_texture(arr_rgba, point_filter=True):
    tmp = "_tmp.png"
    Image.fromarray(arr_rgba, 'RGBA').save(tmp)
    tex = rl.LoadTexture(tmp.encode())
    os.remove(tmp)
    f = rl.TEXTURE_FILTER_POINT if point_filter else rl.TEXTURE_FILTER_BILINEAR
    rl.SetTextureFilter(tex, f)
    return tex

def make_rgb_texture(arr_rgb):
    tmp = "_tmp_rgb.png"
    Image.fromarray(arr_rgb, 'RGB').save(tmp)
    tex = rl.LoadTexture(tmp.encode())
    os.remove(tmp)
    return tex

def make_raw_texture(img_path):
    tmp = "_tmp_raw.png"
    Image.open(img_path).convert("RGBA").save(tmp)
    tex = rl.LoadTexture(tmp.encode())
    os.remove(tmp)
    return tex

# --- Shaders ---
VERT_SRC = b"""
#version 330 core
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec4 vertexColor;
uniform mat4 mvp;
out vec2 fragTexCoord;
out vec4 fragColor;
void main(){
    fragTexCoord = vertexTexCoord;
    fragColor    = vertexColor;
    gl_Position  = mvp * vec4(vertexPosition, 1.0);
}
"""

UNPACK_RG16 = b"""
float unpack_rg16(vec4 s) {
    return (s.r * 255.0 * 256.0 + s.g * 255.0) / 65535.0;
}
"""

FRAG_BILINEAR = UNPACK_RG16 + b"""
#version 330 core
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float threshold;
uniform float edge_width;
out vec4 finalColor;
void main(){
    float d  = unpack_rg16(texture(texture0, fragTexCoord));
    vec3  rgb = texture(texture1, fragTexCoord).rgb;
    float hw = max(edge_width, 0.0001);
    float a  = smoothstep(threshold - hw, threshold + hw, d);
    if(a < 0.001) discard;
    finalColor = vec4(rgb, a) * fragColor;
}
"""

FRAG_BICUBIC = UNPACK_RG16 + b"""
#version 330 core
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float threshold;
uniform float edge_width;
uniform vec2  tex_size;
out vec4 finalColor;

vec4 cubic_w(float t){
    float t2=t*t, t3=t2*t;
    return vec4(-0.5*t3+t2-0.5*t, 1.5*t3-2.5*t2+1.0, -1.5*t3+2.0*t2+0.5*t, 0.5*t3-0.5*t2);
}
float bicubic_sdf(sampler2D tex, vec2 uv, vec2 ts){
    vec2 p=uv*ts-0.5, f=fract(p), ip=floor(p);
    vec4 wx=cubic_w(f.x), wy=cubic_w(f.y);
    float rows[4];
    for(int j=0;j<4;j++){
        float py=(ip.y+float(j-1)+0.5)/ts.y;
        rows[j]=0.0;
        for(int i=0;i<4;i++){
            float px=(ip.x+float(i-1)+0.5)/ts.x;
            rows[j]+=wx[i]*unpack_rg16(texture(tex,vec2(px,py)));
        }
    }
    return wy[0]*rows[0]+wy[1]*rows[1]+wy[2]*rows[2]+wy[3]*rows[3];
}
void main(){
    float d  = bicubic_sdf(texture0, fragTexCoord, tex_size);
    vec3  rgb = texture(texture1, fragTexCoord).rgb;
    float hw = max(edge_width,0.0001);
    float a  = smoothstep(threshold-hw, threshold+hw, d);
    if(a<0.001) discard;
    finalColor = vec4(rgb,a)*fragColor;
}
"""

FRAG_LANCZOS = UNPACK_RG16 + b"""
#version 330 core
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float threshold;
uniform float edge_width;
uniform vec2  tex_size;
out vec4 finalColor;

#define PI 3.14159265358979
float sinc(float x){ if(abs(x)<0.0001)return 1.0; float px=PI*x; return sin(px)/px; }
float lanczos2(float x){ if(abs(x)>=2.0)return 0.0; return sinc(x)*sinc(x*0.5); }

float lanczos_sdf(sampler2D tex, vec2 uv, vec2 ts){
    vec2 p=uv*ts-0.5, f=fract(p), ip=floor(p);
    float wx[5],wy[5],sx=0.0,sy=0.0;
    for(int i=0;i<5;i++){
        wx[i]=lanczos2(float(i-2)-f.x); sx+=wx[i];
        wy[i]=lanczos2(float(i-2)-f.y); sy+=wy[i];
    }
    for(int i=0;i<5;i++){ wx[i]/=sx; wy[i]/=sy; }
    float result=0.0;
    for(int j=0;j<5;j++){
        float py=(ip.y+float(j-2)+0.5)/ts.y, row=0.0;
        for(int i=0;i<5;i++){
            float px=(ip.x+float(i-2)+0.5)/ts.x;
            row+=wx[i]*unpack_rg16(texture(tex,vec2(px,py)));
        }
        result+=wy[j]*row;
    }
    return result;
}
void main(){
    float d  = lanczos_sdf(texture0, fragTexCoord, tex_size);
    vec3  rgb = texture(texture1, fragTexCoord).rgb;
    float hw = max(edge_width,0.0001);
    float a  = smoothstep(threshold-hw, threshold+hw, d);
    if(a<0.001) discard;
    finalColor = vec4(rgb,a)*fragColor;
}
"""

FRAG_RAW = b"""
#version 330 core
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
out vec4 finalColor;
void main(){
    vec4 s=texture(texture0,fragTexCoord);
    if(s.a<0.01) discard;
    finalColor=s*fragColor;
}
"""

# Fix: version directive must be first line — prepend it properly
def prepend_version(src):
    return b"#version 330 core\n" + src

FRAG_BILINEAR = prepend_version(UNPACK_RG16 + b"""
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float threshold;
uniform float edge_width;
out vec4 finalColor;
void main(){
    float d   = unpack_rg16(texture(texture0, fragTexCoord));
    vec3  rgb = texture(texture1, fragTexCoord).rgb;
    float hw  = max(edge_width, 0.0001);
    float a   = smoothstep(threshold - hw, threshold + hw, d);
    if(a < 0.001) discard;
    finalColor = vec4(rgb, a) * fragColor;
}
""")

FRAG_BICUBIC = prepend_version(UNPACK_RG16 + b"""
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float threshold;
uniform float edge_width;
uniform vec2  tex_size;
out vec4 finalColor;
vec4 cubic_w(float t){
    float t2=t*t,t3=t2*t;
    return vec4(-0.5*t3+t2-0.5*t,1.5*t3-2.5*t2+1.0,-1.5*t3+2.0*t2+0.5*t,0.5*t3-0.5*t2);
}
float bicubic_sdf(sampler2D tex,vec2 uv,vec2 ts){
    vec2 p=uv*ts-0.5,f=fract(p),ip=floor(p);
    vec4 wx=cubic_w(f.x),wy=cubic_w(f.y);
    float rows[4];
    for(int j=0;j<4;j++){
        float py=(ip.y+float(j-1)+0.5)/ts.y;
        rows[j]=0.0;
        for(int i=0;i<4;i++){
            float px=(ip.x+float(i-1)+0.5)/ts.x;
            rows[j]+=wx[i]*unpack_rg16(texture(tex,vec2(px,py)));
        }
    }
    return wy[0]*rows[0]+wy[1]*rows[1]+wy[2]*rows[2]+wy[3]*rows[3];
}
void main(){
    float d   = bicubic_sdf(texture0, fragTexCoord, tex_size);
    vec3  rgb = texture(texture1, fragTexCoord).rgb;
    float hw  = max(edge_width,0.0001);
    float a   = smoothstep(threshold-hw,threshold+hw,d);
    if(a<0.001) discard;
    finalColor = vec4(rgb,a)*fragColor;
}
""")

FRAG_LANCZOS = prepend_version(UNPACK_RG16 + b"""
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float threshold;
uniform float edge_width;
uniform vec2  tex_size;
out vec4 finalColor;
#define PI 3.14159265358979
float sinc(float x){if(abs(x)<0.0001)return 1.0;float px=PI*x;return sin(px)/px;}
float lanczos2(float x){if(abs(x)>=2.0)return 0.0;return sinc(x)*sinc(x*0.5);}
float lanczos_sdf(sampler2D tex,vec2 uv,vec2 ts){
    vec2 p=uv*ts-0.5,f=fract(p),ip=floor(p);
    float wx[5],wy[5],sx=0.0,sy=0.0;
    for(int i=0;i<5;i++){wx[i]=lanczos2(float(i-2)-f.x);sx+=wx[i];wy[i]=lanczos2(float(i-2)-f.y);sy+=wy[i];}
    for(int i=0;i<5;i++){wx[i]/=sx;wy[i]/=sy;}
    float result=0.0;
    for(int j=0;j<5;j++){
        float py=(ip.y+float(j-2)+0.5)/ts.y,row=0.0;
        for(int i=0;i<5;i++){float px=(ip.x+float(i-2)+0.5)/ts.x;row+=wx[i]*unpack_rg16(texture(tex,vec2(px,py)));}
        result+=wy[j]*row;
    }
    return result;
}
void main(){
    float d   = lanczos_sdf(texture0, fragTexCoord, tex_size);
    vec3  rgb = texture(texture1, fragTexCoord).rgb;
    float hw  = max(edge_width,0.0001);
    float a   = smoothstep(threshold-hw,threshold+hw,d);
    if(a<0.001) discard;
    finalColor = vec4(rgb,a)*fragColor;
}
""")

FRAG_RAW = b"""
#version 330 core
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
out vec4 finalColor;
void main(){
    vec4 s=texture(texture0,fragTexCoord);
    if(s.a<0.01)discard;
    finalColor=s*fragColor;
}
"""

# --- Helpers ---
def mkcolor(r,g,b,a=255):
    c=rl.ffi.new("Color *"); c.r=r;c.g=g;c.b=b;c.a=a; return c[0]
def mkvec2(x,y):
    v=rl.ffi.new("Vector2 *"); v.x=x;v.y=y; return v[0]
def mkrect(x,y,w,h):
    r=rl.ffi.new("Rectangle *"); r.x=x;r.y=y;r.width=w;r.height=h; return r[0]

def set_f(sh,loc,v):
    p=rl.ffi.new("float *",v); rl.SetShaderValue(sh,loc,p,rl.SHADER_UNIFORM_FLOAT)
def set_v2(sh,loc,x,y):
    p=rl.ffi.new("float[2]",[x,y]); rl.SetShaderValue(sh,loc,p,rl.SHADER_UNIFORM_VEC2)
def set_i(sh,loc,v):
    p=rl.ffi.new("int *",v); rl.SetShaderValue(sh,loc,p,rl.SHADER_UNIFORM_INT)

def draw_quad(tex,iw,ih,cx,cy,sz,tint):
    rl.DrawTexturePro(tex, mkrect(0,0,iw,ih), mkrect(cx-sz*0.5,cy-sz*0.5,sz,sz), mkvec2(0,0), 0.0, tint)

def draw_text_c(text,cx,y,size,col):
    b=text if isinstance(text,bytes) else text.encode()
    rl.DrawText(b, cx - rl.MeasureText(b,size)//2, y, size, col)

def main():
    img_path = "aimage.png"
    if not os.path.exists(img_path):
        print("aimage.png not found"); sys.exit(1)

    rl.InitWindow(W, H, b"SDF RG16 Demo")
    rl.SetTargetFPS(60)

    # resolve after context creation
    resolve_gl()

    SDF_SMALL_SIZE = 64
    SUPERSAMPLE    = 4

    print("Generating SDF...")
    (small_rg, small_rgb_arr, sw, sh_size, edge_t_small, src_w, src_h) = \
        build_sdf_texture(img_path, SDF_SMALL_SIZE, SUPERSAMPLE)
    print(f"Source: {src_w}x{src_h}  SDF small: {sw}x{sh_size} edge@{edge_t_small:.4f}")

    sdf_point_tex    = make_png_texture(small_rg, point_filter=True)
    sdf_bilinear_tex = make_png_texture(small_rg, point_filter=False)
    rgb_tex          = make_rgb_texture(small_rgb_arr)
    raw_tex          = make_raw_texture(img_path)

    sh_raw      = rl.LoadShaderFromMemory(VERT_SRC, FRAG_RAW)
    sh_bilinear = rl.LoadShaderFromMemory(VERT_SRC, FRAG_BILINEAR)
    sh_bicubic  = rl.LoadShaderFromMemory(VERT_SRC, FRAG_BICUBIC)
    sh_lanczos  = rl.LoadShaderFromMemory(VERT_SRC, FRAG_LANCZOS)

    # set texture1 = slot 1 for all SDF shaders
    for sh in [sh_bilinear, sh_bicubic, sh_lanczos]:
        loc = rl.GetShaderLocation(sh, b"texture1")
        set_i(sh, loc, 1)

    def locs(sh):
        return (rl.GetShaderLocation(sh, b"threshold"),
                rl.GetShaderLocation(sh, b"edge_width"),
                rl.GetShaderLocation(sh, b"tex_size"))

    bl_th, bl_ew, _     = locs(sh_bilinear)
    bi_th, bi_ew, bi_ts = locs(sh_bicubic)
    lz_th, lz_ew, lz_ts = locs(sh_lanczos)

    def draw_sdf(sh, sdf_tex, cx, cy, sz):
        gl_active_texture(GL_TEXTURE1)
        gl_bind_texture(GL_TEXTURE_2D, rgb_tex.id)
        gl_active_texture(GL_TEXTURE0)
        rl.BeginShaderMode(sh)
        draw_quad(sdf_tex, sw, sh_size, cx, cy, sz, WHITE)
        rl.EndShaderMode()
        gl_active_texture(GL_TEXTURE1)
        gl_bind_texture(GL_TEXTURE_2D, 0)
        gl_active_texture(GL_TEXTURE0)

    offset     = 0.0
    edge_width = 0.02

    WHITE   = mkcolor(255,255,255)
    SKYBLUE = mkcolor(135,206,235)
    YELLOW  = mkcolor(255,230,80)

    render_sizes = [64, 128, 256, 512]
    top_y = 80

    col_xs = [W//5, (W*2)//5, (W*3)//5, (W*4)//5]
    col_labels = [
        f"RAW {src_w}x{src_h}",
        f"bilinear {sw}x{sh_size}",
        f"bicubic 4x4 {sw}x{sh_size}",
        f"lanczos 5x5 {sw}x{sh_size}",
    ]

    while not rl.WindowShouldClose():
        dt = rl.GetFrameTime()
        if rl.IsKeyDown(rl.KEY_D): offset = min(0.5, offset + 0.2*dt)
        if rl.IsKeyDown(rl.KEY_A): offset = max(-0.5, offset - 0.2*dt)
        if rl.IsKeyDown(rl.KEY_W): edge_width = min(0.5, edge_width + 0.1*dt)
        if rl.IsKeyDown(rl.KEY_S): edge_width = max(0.0, edge_width - 0.1*dt)

        rl.BeginDrawing()
        rl.ClearBackground(SKYBLUE)

        rl.DrawText(f"offset:{offset:.3f}  ew:{edge_width:.4f}  [A/D offset] [W/S edge]".encode(), 20, 20, FONT_SIZE, WHITE)
        for cx, lbl in zip(col_xs, col_labels):
            draw_text_c(lbl, cx, top_y, FONT_SIZE, YELLOW)

        y = top_y + FONT_SIZE + 28
        for sz in render_sizes:
            rl.DrawText(f"{sz}px".encode(), 10, y + sz//2 - FONT_SIZE//2, FONT_SIZE, WHITE)

            rl.BeginShaderMode(sh_raw)
            draw_quad(raw_tex, src_w, src_h, col_xs[0], y+sz//2, sz, WHITE)
            rl.EndShaderMode()

            set_f(sh_bilinear, bl_th, edge_t_small + offset)
            set_f(sh_bilinear, bl_ew, edge_width)
            draw_sdf(sh_bilinear, sdf_bilinear_tex, col_xs[1], y+sz//2, sz)

            set_f(sh_bicubic, bi_th, edge_t_small + offset)
            set_f(sh_bicubic, bi_ew, edge_width)
            set_v2(sh_bicubic, bi_ts, float(sw), float(sh_size))
            draw_sdf(sh_bicubic, sdf_point_tex, col_xs[2], y+sz//2, sz)

            set_f(sh_lanczos, lz_th, edge_t_small + offset)
            set_f(sh_lanczos, lz_ew, edge_width)
            set_v2(sh_lanczos, lz_ts, float(sw), float(sh_size))
            draw_sdf(sh_lanczos, sdf_point_tex, col_xs[3], y+sz//2, sz)

            y += sz + 20

        rl.EndDrawing()

    for t in [sdf_point_tex, sdf_bilinear_tex, rgb_tex, raw_tex]:
        rl.UnloadTexture(t)
    for s in [sh_raw, sh_bilinear, sh_bicubic, sh_lanczos]:
        rl.UnloadShader(s)
    rl.CloseWindow()

if __name__ == "__main__":
    main()
