"""The node-graph-as-code spine.

TouchDesigner is a visual DAG of operators. Here a graph is a list of `Op`s, each a pure
function that reads named keys from a shared per-frame context and writes named keys back.
Running the graph threads one `ctx` dict through the ops in order. This is the push-through-
context approximation of TD's pull model: simple, inspectable, and trivial for an agent to
author and rewire (it's just a list of Ops).

Example
-------
    graph = Graph([
        Op("displace", lambda ctx: {"pos": displace_z(ctx["grid"], ctx["luma"], ctx["depth"])},
           reads=("grid", "luma", "depth"), writes=("pos",)),
        Op("pack", lambda ctx: {"inst": pack_instances(ctx["pos"], ctx["scale"], ctx["euler"])},
           reads=("pos", "scale", "euler"), writes=("inst",)),
    ])
    ctx = graph.run({"grid": grid, "luma": luma, "depth": 1.2, "scale": s, "euler": e})
    frame = renderer.render(ctx["inst"])

The `reads`/`writes` declarations are optional metadata used for validation and for printing
the graph; they make the data dependencies explicit the way wires do in TD.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Callable, Dict, Sequence, Any


@dataclass
class Op:
    """A single node: a pure function of the context that returns a dict of outputs."""
    name: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    reads: Sequence[str] = ()
    writes: Sequence[str] = ()

    def __call__(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        for k in self.reads:
            if k not in ctx:
                raise KeyError(f"op '{self.name}' reads missing key '{k}'")
        out = self.fn(ctx)
        if not isinstance(out, dict):
            raise TypeError(f"op '{self.name}' must return a dict, got {type(out).__name__}")
        if self.writes:
            missing = [k for k in self.writes if k not in out]
            if missing:
                raise KeyError(f"op '{self.name}' declared writes {self.writes} but omitted {missing}")
        return out


@dataclass
class Graph:
    """An ordered list of Ops evaluated against a shared context dict."""
    ops: Sequence[Op] = _dc_field(default_factory=list)

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx = dict(ctx)
        for op in self.ops:
            ctx.update(op(ctx))
        return ctx

    def describe(self) -> str:
        lines = ["Graph:"]
        for op in self.ops:
            r = ", ".join(op.reads) or "-"
            w = ", ".join(op.writes) or "-"
            lines.append(f"  {op.name}: [{r}] -> [{w}]")
        return "\n".join(lines)
