"""Glow renderer — soft, additive, trailing particles. The "luminous flowing dust" look.

Each particle is an instanced camera-facing billboard quad with a soft radial falloff,
drawn with additive blending into an HDR (RGBA16F) accumulation buffer. A ping-pong feedback
pass fades the previous frame and adds the new particles, so motion leaves smooth glowing
trails. A final exposure tonemap maps the HDR buffer to an 8-bit image read back as NumPy.

GL_POINTS is avoided on purpose — macOS GL 4.1 core clamps point size and skips perspective
scaling. Instanced quads are reliable and give per-particle size for free.
"""
from __future__ import annotations

import numpy as np
import moderngl

_PARTICLE_VS = """
#version 330
in vec2 in_corner;
in vec2 i_pos;     // NDC
in float i_size;   // half-size in NDC-y units
in float i_bright;
in vec3 i_color;
uniform float u_aspect;
out vec2 v_coord; out float v_bright; out vec3 v_color;
void main(){
    v_coord = in_corner; v_bright = i_bright; v_color = i_color;
    vec2 off = in_corner * i_size * vec2(1.0/u_aspect, 1.0);
    gl_Position = vec4(i_pos + off, 0.0, 1.0);
}
"""
_PARTICLE_FS = """
#version 330
in vec2 v_coord; in float v_bright; in vec3 v_color;
out vec4 f_color;
void main(){
    float d = length(v_coord);
    if(d > 1.0) discard;
    float a = smoothstep(1.0, 0.0, d);
    a = pow(a, 1.6);
    float g = a * v_bright;
    f_color = vec4(v_color * g, g);
}
"""
_FS_VS = """
#version 330
in vec2 in_vert; out vec2 v_uv;
void main(){ v_uv = in_vert*0.5+0.5; gl_Position = vec4(in_vert,0.0,1.0); }
"""
_FADE_FS = """
#version 330
in vec2 v_uv; uniform sampler2D u_prev; uniform float u_fade; out vec4 f_color;
void main(){ f_color = texture(u_prev, v_uv) * u_fade; }
"""
_COMPOSITE_FS = """
#version 330
in vec2 v_uv; uniform sampler2D u_scene; uniform float u_exposure; out vec4 f_color;
void main(){
    vec3 hdr = texture(u_scene, v_uv).rgb * u_exposure;
    vec3 mapped = vec3(1.0) - exp(-hdr);   // exposure tonemap
    f_color = vec4(mapped, 1.0);
}
"""


class GlowRenderer:
    def __init__(self, width, height, n, fade=0.90, exposure=1.3):
        self.w, self.h, self.n = width, height, n
        self.fade, self.exposure = fade, exposure
        ctx = self.ctx = moderngl.create_standalone_context()

        self.p_prog = ctx.program(vertex_shader=_PARTICLE_VS, fragment_shader=_PARTICLE_FS)
        self.fade_prog = ctx.program(vertex_shader=_FS_VS, fragment_shader=_FADE_FS)
        self.comp_prog = ctx.program(vertex_shader=_FS_VS, fragment_shader=_COMPOSITE_FS)
        self.p_prog["u_aspect"].value = width / height

        corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.corner_vbo = ctx.buffer(corners.tobytes())
        self.inst_vbo = ctx.buffer(reserve=n * 7 * 4, dynamic=True)
        self.p_vao = ctx.vertex_array(self.p_prog, [
            (self.corner_vbo, "2f", "in_corner"),
            (self.inst_vbo, "2f 1f 1f 3f/i", "i_pos", "i_size", "i_bright", "i_color"),
        ])
        tri = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        self.tri_vbo = ctx.buffer(tri.tobytes())
        self.fade_vao = ctx.vertex_array(self.fade_prog, [(self.tri_vbo, "2f", "in_vert")])
        self.comp_vao = ctx.vertex_array(self.comp_prog, [(self.tri_vbo, "2f", "in_vert")])

        def hdr():
            t = ctx.texture((width, height), 4, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            return t, ctx.framebuffer(color_attachments=[t])
        self.tex_a, self.fbo_a = hdr()
        self.tex_b, self.fbo_b = hdr()
        self.out_tex = ctx.texture((width, height), 3)
        self.out_fbo = ctx.framebuffer(color_attachments=[self.out_tex])
        self.fbo_a.use(); self.ctx.clear(0, 0, 0, 1)
        self.fbo_b.use(); self.ctx.clear(0, 0, 0, 1)

    def render(self, instance_data: np.ndarray) -> np.ndarray:
        ctx = self.ctx
        self.inst_vbo.write(instance_data.astype("f4").tobytes())

        # 1) fade previous accumulation (tex_a) into fbo_b
        self.fbo_b.use()
        ctx.clear(0, 0, 0, 1)
        ctx.disable(moderngl.BLEND)
        self.tex_a.use(0); self.fade_prog["u_prev"].value = 0
        self.fade_prog["u_fade"].value = self.fade
        self.fade_vao.render(moderngl.TRIANGLES, vertices=3)

        # 2) additive particles on top
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False
        self.p_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.n)

        # 3) tonemap to 8-bit output
        self.out_fbo.use()
        ctx.disable(moderngl.BLEND)
        self.tex_b.use(0); self.comp_prog["u_scene"].value = 0
        self.comp_prog["u_exposure"].value = self.exposure
        self.comp_vao.render(moderngl.TRIANGLES, vertices=3)

        raw = self.out_fbo.read(components=3)
        img = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
        out = np.flipud(img).copy()

        # 4) swap ping-pong
        self.tex_a, self.tex_b = self.tex_b, self.tex_a
        self.fbo_a, self.fbo_b = self.fbo_b, self.fbo_a
        return out

    def release(self):
        self.ctx.release()
