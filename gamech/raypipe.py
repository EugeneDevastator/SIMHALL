import pyray as rl
import math

SCREEN_W = 1920
SCREEN_H = 1080
FONT_SIZE = 32
LINE_WIDTH = 1.2

LIGHT_DIR = (0.4, 0.8, -0.3)

POINTS = [
    (-4, 0, 0),
    (-2, 2, 1),
    (0, 0, 2),
    (2, -2, 1),
    (4, 1, -1),
    (6, 0, -2),
]

VERT_SRC = """
#version 330 core
in vec3 vertexPosition;
in vec4 vertexColor;
out float vU;
uniform mat4 mvp;
void main() {
    vU = vertexColor.r;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
in float vU;
out vec4 fragColor;
uniform vec3 lightDir;
uniform vec3 rightDir;
uniform vec3 upDir;
void main() {
    // x in [-1,1] across tube width
    float x = vU * 2.0 - 1.0;

    // cylinder depth profile: sqrt(1 - x^2), this IS the z-component of surface normal
    float profile = sqrt(max(0.0, 1.0 - x * x));

    // discard outside cylinder silhouette
    if (profile < 0.001) discard;

    // surface normal in world space:
    // right direction = x axis of tube cross-section
    // profile = z component (toward camera)
    // no up component for a billboard cylinder
    vec3 normal = normalize(rightDir * x + upDir * 0.0 + normalize(lightDir) * 0.0);
    // actually: normal = x*rightDir + profile*(toward_cam)
    // but we only have rightDir per segment, reconstruct:
    // normal points: x along right, profile along view
    // for lighting we need world normal
    // approximate: normal = rightDir * x  (profile is view-facing, skip for diffuse)
    // better: treat profile as the "facing" weight
    normal = normalize(rightDir * x);
    // dot with light for diffuse, profile masks the silhouette
    float ndotl = dot(normal, normalize(lightDir));

    // red = spherized gradient (cylinder depth)
    float red = profile;

    // green = diffuse light masked by profile
    float green = max(0.0, ndotl) * profile;

    fragColor = vec4(red, green, 0.0, 1.0);
}
"""

def catmull_rom(p0, p1, p2, p3, t):
    t2=t*t; t3=t2*t
    x=0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
    y=0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
    z=0.5*((2*p1[2])+(-p0[2]+p2[2])*t+(2*p0[2]-5*p1[2]+4*p2[2]-p3[2])*t2+(-p0[2]+3*p1[2]-3*p2[2]+p3[2])*t3)
    return (x,y,z)

def build_spline(points, steps=30):
    result=[]
    n=len(points)
    for i in range(n-1):
        p0=points[max(i-1,0)]; p1=points[i]
        p2=points[i+1]; p3=points[min(i+2,n-1)]
        for s in range(steps):
            result.append(catmull_rom(p0,p1,p2,p3,s/steps))
    result.append(points[-1])
    return result

def vnorm(a):
    l=math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
    if l<1e-9: return (0,0,1)
    return (a[0]/l,a[1]/l,a[2]/l)

def vcross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def vsub(a,b): return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def vadd(a,b): return (a[0]+b[0],a[1]+b[1],a[2]+b[2])
def vscale(a,s): return (a[0]*s,a[1]*s,a[2]*s)

def draw_spline_shader(spline, shader, cam_pos):
    cam=(cam_pos.x, cam_pos.y, cam_pos.z)
    n=len(spline)

    light_dir_loc = rl.get_shader_location(shader, "lightDir")
    right_dir_loc = rl.get_shader_location(shader, "rightDir")
    up_dir_loc    = rl.get_shader_location(shader, "upDir")

    ld=vnorm(LIGHT_DIR)
    rl.set_shader_value(shader, light_dir_loc,
        rl.Vector3(ld[0],ld[1],ld[2]), rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)

    rl.begin_shader_mode(shader)

    for i in range(n-1):
        a=spline[i]; b=spline[i+1]
        fwd=vnorm(vsub(b,a))
        mid=vscale(vadd(a,b),0.5)
        to_cam=vnorm(vsub(cam,mid))
        right=vnorm(vcross(fwd,to_cam))
        up=vnorm(vcross(right,fwd))

        rl.set_shader_value(shader, right_dir_loc,
            rl.Vector3(right[0],right[1],right[2]), rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
        rl.set_shader_value(shader, up_dir_loc,
            rl.Vector3(up[0],up[1],up[2]), rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)

        half=vscale(right, LINE_WIDTH*0.5)
        v0=vsub(a,half)
        v1=vadd(a,half)
        v2=vadd(b,half)
        v3=vsub(b,half)

        c0=rl.Color(0,  0,0,255)
        c1=rl.Color(255,0,0,255)

        rl.rl_begin(rl.RL_QUADS)

        rl.rl_color4ub(c0.r,c0.g,c0.b,c0.a); rl.rl_vertex3f(*v0)
        rl.rl_color4ub(c1.r,c1.g,c1.b,c1.a); rl.rl_vertex3f(*v1)
        rl.rl_color4ub(c1.r,c1.g,c1.b,c1.a); rl.rl_vertex3f(*v2)
        rl.rl_color4ub(c0.r,c0.g,c0.b,c0.a); rl.rl_vertex3f(*v3)

        rl.rl_color4ub(c0.r,c0.g,c0.b,c0.a); rl.rl_vertex3f(*v3)
        rl.rl_color4ub(c1.r,c1.g,c1.b,c1.a); rl.rl_vertex3f(*v2)
        rl.rl_color4ub(c1.r,c1.g,c1.b,c1.a); rl.rl_vertex3f(*v1)
        rl.rl_color4ub(c0.r,c0.g,c0.b,c0.a); rl.rl_vertex3f(*v0)

        rl.rl_end()

    rl.end_shader_mode()

def main():
    rl.init_window(SCREEN_W,SCREEN_H,"Spline Pipe")
    rl.set_target_fps(60)
    rl.disable_cursor()

    shader=rl.load_shader_from_memory(VERT_SRC,FRAG_SRC)

    camera=rl.Camera3D(
        rl.Vector3(0,2,-8), rl.Vector3(0,0,0), rl.Vector3(0,1,0),
        60.0, rl.CameraProjection.CAMERA_PERSPECTIVE)

    yaw=0.0; pitch=0.0
    SPEED=5.0; SENS=0.002
    spline=build_spline(POINTS,steps=30)

    while not rl.window_should_close():
        dt=rl.get_frame_time()
        md=rl.get_mouse_delta()
        yaw-=md.x*SENS
        pitch=max(-1.5,min(1.5,pitch-md.y*SENS))

        fwd=(math.cos(pitch)*math.sin(yaw),math.sin(pitch),math.cos(pitch)*math.cos(yaw))
        right=(math.cos(yaw),0,-math.sin(yaw))

        move=[0.0,0.0,0.0]
        if rl.is_key_down(rl.KeyboardKey.KEY_W): move[0]+=fwd[0];move[1]+=fwd[1];move[2]+=fwd[2]
        if rl.is_key_down(rl.KeyboardKey.KEY_S): move[0]-=fwd[0];move[1]-=fwd[1];move[2]-=fwd[2]
        if rl.is_key_down(rl.KeyboardKey.KEY_A): move[0]+=right[0];move[1]+=right[1];move[2]+=right[2]
        if rl.is_key_down(rl.KeyboardKey.KEY_D): move[0]-=right[0];move[1]-=right[1];move[2]-=right[2]
        if rl.is_key_down(rl.KeyboardKey.KEY_E): move[1]+=1.0
        if rl.is_key_down(rl.KeyboardKey.KEY_Q): move[1]-=1.0

        camera.position.x+=move[0]*SPEED*dt
        camera.position.y+=move[1]*SPEED*dt
        camera.position.z+=move[2]*SPEED*dt
        camera.target.x=camera.position.x+fwd[0]
        camera.target.y=camera.position.y+fwd[1]
        camera.target.z=camera.position.z+fwd[2]

        rl.begin_drawing()
        rl.clear_background(rl.Color(20,20,20,255))
        rl.begin_mode_3d(camera)
        draw_spline_shader(spline,shader,camera.position)
        rl.draw_grid(20,1)
        rl.end_mode_3d()
        rl.draw_fps(10,10)
        rl.draw_text("WASD+QE move | Mouse look | DEBUG: R=depth G=diffuse",10,SCREEN_H-40,FONT_SIZE,rl.RAYWHITE)
        rl.end_drawing()

    rl.unload_shader(shader)
    rl.enable_cursor()
    rl.close_window()

if __name__=="__main__":
    main()
