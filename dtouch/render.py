"""Render operator — the Copy SOP (instance a box) + Light + Camera, on the GPU.

A headless moderngl renderer that draws N instanced cubes, each with a per-instance
offset / scale / Euler rotation, lit by one directional light with a depth test.
Renders to an offscreen framebuffer (CGL standalone context on macOS, no window) and
returns the frame as an (H, W, 3) uint8 NumPy array.

Shadows are intentionally left for a follow-up pass (a depth-from-light shadow map);
v1 is lights + depth so the end-to-end pipeline is reliable.
"""
from __future__ import annotations

import numpy as np
import moderngl


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, target, up) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    f = np.asarray(target, dtype=np.float32) - eye
    f /= np.linalg.norm(f)
    up = np.asarray(up, dtype=np.float32)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _unit_cube() -> np.ndarray:
    """36 vertices of a unit cube centered at origin, interleaved pos(3) + normal(3)."""
    faces = [
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
        ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1)]),
        ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]),
        ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, -1, -1), (-1, 1, 1), (-1, 1, -1)]),
        ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, 1), (1, 1, -1), (-1, 1, -1)]),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
    ]
    verts = []
    for normal, tris in faces:
        for v in tris:
            verts.append([c * 0.5 for c in v] + list(normal))  # half-extent -> unit side
    return np.array(verts, dtype=np.float32)


VERTEX_SHADER = """
#version 330
uniform mat4 u_mvp;
uniform float u_base_size;
in vec3 in_pos;
in vec3 in_normal;
in vec3 i_offset;
in vec3 i_scale;
in vec3 i_euler;
out vec3 v_normal;
out float v_height;

mat3 rotX(float a){ float c=cos(a),s=sin(a); return mat3(vec3(1,0,0),vec3(0,c,s),vec3(0,-s,c)); }
mat3 rotY(float a){ float c=cos(a),s=sin(a); return mat3(vec3(c,0,-s),vec3(0,1,0),vec3(s,0,c)); }
mat3 rotZ(float a){ float c=cos(a),s=sin(a); return mat3(vec3(c,s,0),vec3(-s,c,0),vec3(0,0,1)); }

void main() {
    mat3 R = rotZ(i_euler.z) * rotY(i_euler.y) * rotX(i_euler.x);
    vec3 local = R * (in_pos * i_scale * u_base_size);
    vec3 world = i_offset + local;
    v_normal = normalize(R * in_normal);
    v_height = i_offset.z;
    gl_Position = u_mvp * vec4(world, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform vec3 u_light_dir;   // direction TO the light, normalized
in vec3 v_normal;
in float v_height;
out vec4 f_color;
void main() {
    float diff = max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0);
    float ambient = 0.18;
    // white dots; brighter where displaced higher so the relief reads clearly
    vec3 albedo = vec3(1.0) * (0.55 + 0.45 * clamp(v_height, 0.0, 1.0));
    vec3 color = albedo * (ambient + (1.0 - ambient) * diff);
    f_color = vec4(color, 1.0);
}
"""


class Renderer:
    def __init__(self, width: int, height: int, n_instances: int,
                 base_size: float, depth_scale: float = 1.2, extent: float = 2.0):
        self.width, self.height, self.n = width, height, n_instances
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        cube = _unit_cube()
        self.cube_vbo = self.ctx.buffer(cube.tobytes())
        self.inst_vbo = self.ctx.buffer(reserve=n_instances * 9 * 4, dynamic=True)
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.cube_vbo, "3f 3f", "in_pos", "in_normal"),
            (self.inst_vbo, "3f 3f 3f/i", "i_offset", "i_scale", "i_euler"),
        ])

        color = self.ctx.texture((width, height), 4)
        depth = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)

        # camera looks down at the grid at an angle so Z-displacement reads as relief (Z up)
        eye = (0.0, -extent * 1.15, extent * 0.95)
        proj = _perspective(45.0, width / height, 0.1, 50.0)
        view = _look_at(eye, (0.0, 0.0, depth_scale * 0.35), (0.0, 0.0, 1.0))
        mvp = proj @ view
        self.prog["u_mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog["u_base_size"].value = float(base_size)
        ld = np.array([0.4, -0.5, 1.0], dtype=np.float32); ld /= np.linalg.norm(ld)
        self.prog["u_light_dir"].value = tuple(float(x) for x in ld)

    def render(self, instance_buf: np.ndarray) -> np.ndarray:
        self.inst_vbo.write(instance_buf.astype("f4").tobytes())
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.vao.render(moderngl.TRIANGLES, instances=self.n)
        raw = self.fbo.read(components=3)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.flipud(img).copy()  # GL origin is bottom-left -> flip to image top-left

    def release(self):
        self.ctx.release()
