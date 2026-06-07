"""dtouch — a modular, CLI/agent-drivable engine for TouchDesigner-style effects.

Operators are small pure functions over NumPy arrays / a shared context; the renderer is a
headless moderngl GPU pipeline. Compose them as code (see `pipeline.Graph`) instead of wiring
a GUI. Everything is headless-verifiable so an agent can one-shot an effect and check the
rendered frame without a display or device permissions.
"""
from .field import make_grid, displace_z, random_scale, random_euler, pack_instances
from .sources import SyntheticSource, ImageSource, VideoSource, make_source
from .render import Renderer
from .pipeline import Op, Graph

__all__ = [
    "make_grid", "displace_z", "random_scale", "random_euler", "pack_instances",
    "SyntheticSource", "ImageSource", "VideoSource", "make_source",
    "Renderer", "Op", "Graph",
]
