"""Shadow-mapped renderer — adds depth-from-light shadows to the instanced cubes.

Two passes:
  1. render cube depth from the light's point of view into a depth texture (the shadow map),
     using an orthographic light projection that covers the scene;
  2. render the scene from the camera, projecting each fragment into light space and comparing
     its depth against the shadow map to decide if it's occluded.

A ground plane is added so the displaced cubes cast visible shadows onto something. This is
the TouchDesigner Light + Shadow stage, done with a standard LearnOpenGL-style shadow map.
"""
from __future__ import annotations

import numpy as np
import moderngl

from .render import _unit_cube, _perspective, _look_at


def _ortho(l, r, b, t, n, f) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (r - l)
    m[1, 1] = 2.0 / (t - b)
    m[2, 2] = -2.0 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    m[3, 3] = 1.0
    return m


_ROT_GLSL = """
mat3 rotX(float a){ float c=cos(a),s=sin(a); return mat3(vec3(1,0,0),vec3(0,c,s),vec3(0,-s,c)); }
mat3 rotY(float a){ float c=cos(a),s=sin(a); return mat3(vec3(c,0,-s),vec3(0,1,0),vec3(s,0,c)); }
mat3 rotZ(float a){ float c=cos(a),s=sin(a); return mat3(vec3(c,s,0),vec3(-s,c,0),vec3(0,0,1)); }
"""

DEPTH_VS = "#version 330\nuniform mat4 u_light_mvp;\nuniform float u_base_size;\n" + _ROT_GLSL + """
in vec3 in_pos;
in vec3 i_offset; in vec3 i_scale; in vec3 i_euler;
void main(){
    mat3 R = rotZ(i_euler.z)*rotY(i_euler.y)*rotX(i_euler.x);
    vec3 world = i_offset + R*(in_pos*i_scale*u_base_size);
    gl_Position = u_light_mvp * vec4(world,1.0);
}
"""
DEPTH_FS = "#version 330\nvoid main(){}\n"

_SHADOW_FN = """
uniform sampler2D u_shadow;
float shadow_factor(vec4 ls, float ndl){
    vec3 p = ls.xyz/ls.w * 0.5 + 0.5;
    if(p.z > 1.0) return 0.0;
    float bias = max(0.0025*(1.0-ndl), 0.0008);
    float s = 0.0; vec2 tx = 1.0/vec2(textureSize(u_shadow,0));
    for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++){
        float closest = texture(u_shadow, p.xy+vec2(x,y)*tx).r;
        s += (p.z - bias > closest) ? 1.0 : 0.0;
    }
    return s/9.0;
}
"""

CUBE_VS = "#version 330\nuniform mat4 u_mvp;\nuniform mat4 u_light_mvp;\nuniform float u_base_size;\n" + _ROT_GLSL + """
in vec3 in_pos; in vec3 in_normal;
in vec3 i_offset; in vec3 i_scale; in vec3 i_euler;
out vec3 v_normal; out float v_height; out vec4 v_lightspace;
void main(){
    mat3 R = rotZ(i_euler.z)*rotY(i_euler.y)*rotX(i_euler.x);
    vec3 world = i_offset + R*(in_pos*i_scale*u_base_size);
    v_normal = normalize(R*in_normal);
    v_height = i_offset.z;
    v_lightspace = u_light_mvp * vec4(world,1.0);
    gl_Position = u_mvp * vec4(world,1.0);
}
"""
CUBE_FS = "#version 330\nuniform vec3 u_light_dir;\n" + _SHADOW_FN + """
in vec3 v_normal; in float v_height; in vec4 v_lightspace;
out vec4 f_color;
void main(){
    float ndl = max(dot(normalize(v_normal), normalize(u_light_dir)),0.0);
    float sh = shadow_factor(v_lightspace, ndl);
    vec3 albedo = vec3(1.0)*(0.55+0.45*clamp(v_height,0.0,1.0));
    float ambient = 0.2;
    vec3 color = albedo*(ambient + (1.0-ambient)*ndl*(1.0-sh));
    f_color = vec4(color,1.0);
}
"""

GROUND_VS = "#version 330\nuniform mat4 u_mvp;\nuniform mat4 u_light_mvp;\n" + """
in vec3 in_pos;
out vec4 v_lightspace; out vec3 v_world;
void main(){
    v_world = in_pos;
    v_lightspace = u_light_mvp*vec4(in_pos,1.0);
    gl_Position = u_mvp*vec4(in_pos,1.0);
}
"""
GROUND_FS = "#version 330\nuniform vec3 u_light_dir;\n" + _SHADOW_FN + """
in vec4 v_lightspace; in vec3 v_world;
out vec4 f_color;
void main(){
    float ndl = max(dot(vec3(0,0,1), normalize(u_light_dir)),0.0);
    float sh = shadow_factor(v_lightspace, ndl);
    float base = 0.24;
    float lit = base*(0.25 + 0.75*ndl*(1.0-sh));
    f_color = vec4(vec3(lit),1.0);
}
"""


class ShadowRenderer:
    def __init__(self, width, height, n_instances, base_size,
                 depth_scale=1.2, extent=2.0, shadow_size=2048):
        self.width, self.height, self.n = width, height, n_instances
        ctx = self.ctx = moderngl.create_standalone_context()
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        cube = _unit_cube()
        self.cube_vbo = ctx.buffer(cube.tobytes())
        self.inst_vbo = ctx.buffer(reserve=n_instances * 9 * 4, dynamic=True)

        self.depth_prog = ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)
        self.cube_prog = ctx.program(vertex_shader=CUBE_VS, fragment_shader=CUBE_FS)
        inst_layout = (self.inst_vbo, "3f 3f 3f/i", "i_offset", "i_scale", "i_euler")
        self.depth_vao = ctx.vertex_array(self.depth_prog,
            [(self.cube_vbo, "3f 12x", "in_pos"), inst_layout])
        self.cube_vao = ctx.vertex_array(self.cube_prog,
            [(self.cube_vbo, "3f 3f", "in_pos", "in_normal"), inst_layout])

        # ground quad just below z=0
        G = extent
        z = -0.03
        ground = np.array([[-G, -G, z], [G, -G, z], [G, G, z],
                           [-G, -G, z], [G, G, z], [-G, G, z]], dtype=np.float32)
        self.ground_vbo = ctx.buffer(ground.tobytes())
        self.ground_prog = ctx.program(vertex_shader=GROUND_VS, fragment_shader=GROUND_FS)
        self.ground_vao = ctx.vertex_array(self.ground_prog, [(self.ground_vbo, "3f", "in_pos")])

        # shadow map (depth texture rendered from the light)
        self.shadow_tex = ctx.depth_texture((shadow_size, shadow_size))
        self.shadow_tex.compare_func = ""   # sample raw depth, not hardware compare
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)

        # main target
        color = ctx.texture((width, height), 4)
        depth = ctx.depth_texture((width, height))
        self.fbo = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)

        # camera + light matrices
        eye = (extent * 0.55, -extent * 1.15, extent * 1.0)
        proj = _perspective(45.0, width / height, 0.1, 60.0)
        view = _look_at(eye, (0.0, 0.0, depth_scale * 0.3), (0.0, 0.0, 1.0))
        self.mvp = (proj @ view).astype(np.float32)

        ld = np.array([0.62, -0.42, 0.5], dtype=np.float32); ld /= np.linalg.norm(ld)
        self.light_dir = ld
        light_pos = ld * (extent * 2.2)
        s = extent * 1.4
        lproj = _ortho(-s, s, -s, s, 0.1, extent * 6.0)
        lview = _look_at(light_pos, (0, 0, 0), (0, 0, 1))
        self.light_mvp = (lproj @ lview).astype(np.float32)

        self.base_size = float(base_size)
        for prog in (self.depth_prog, self.cube_prog, self.ground_prog):
            if "u_light_mvp" in prog:
                prog["u_light_mvp"].write(self.light_mvp.T.tobytes())
            if "u_mvp" in prog:
                prog["u_mvp"].write(self.mvp.T.tobytes())
            if "u_base_size" in prog:
                prog["u_base_size"].value = self.base_size
            if "u_light_dir" in prog:
                prog["u_light_dir"].value = tuple(float(x) for x in ld)

    def render(self, instance_buf: np.ndarray) -> np.ndarray:
        self.inst_vbo.write(instance_buf.astype("f4").tobytes())

        # pass 1: cube depth from light
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        self.depth_vao.render(moderngl.TRIANGLES, instances=self.n)

        # pass 2: camera view with shadow sampling
        self.shadow_tex.use(location=0)
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.cube_prog["u_shadow"].value = 0
        self.ground_prog["u_shadow"].value = 0
        self.ground_vao.render(moderngl.TRIANGLES)
        self.cube_vao.render(moderngl.TRIANGLES, instances=self.n)

        raw = self.fbo.read(components=3)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.flipud(img).copy()

    def release(self):
        self.ctx.release()
