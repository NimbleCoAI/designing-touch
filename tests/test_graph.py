"""TDD for the node-graph-as-code spine."""
import numpy as np
import pytest

from dtouch import Op, Graph, make_grid, displace_z, pack_instances, random_scale, random_euler


def test_graph_threads_context_in_order():
    g = Graph([
        Op("a", lambda c: {"x": c["seed"] + 1}, reads=("seed",), writes=("x",)),
        Op("b", lambda c: {"y": c["x"] * 10}, reads=("x",), writes=("y",)),
    ])
    out = g.run({"seed": 2})
    assert out["x"] == 3 and out["y"] == 30


def test_op_raises_on_missing_read():
    op = Op("need", lambda c: {"z": 1}, reads=("missing",), writes=("z",))
    with pytest.raises(KeyError):
        op({})


def test_op_raises_when_declared_write_missing():
    op = Op("liar", lambda c: {"a": 1}, writes=("b",))
    with pytest.raises(KeyError):
        op({})


def test_graph_builds_a_real_displacement_field():
    gx, gy = 8, 6
    n = gx * gy
    graph = Graph([
        Op("displace",
           lambda c: {"pos": displace_z(c["grid"], c["luma"], c["depth"])},
           reads=("grid", "luma", "depth"), writes=("pos",)),
        Op("pack",
           lambda c: {"inst": pack_instances(c["pos"], c["scale"], c["euler"])},
           reads=("pos", "scale", "euler"), writes=("inst",)),
    ])
    out = graph.run({
        "grid": make_grid(gx, gy),
        "luma": np.ones((gy, gx), dtype=np.float32),
        "depth": 1.5,
        "scale": random_scale(n, seed=0),
        "euler": random_euler(n, seed=1),
    })
    assert out["inst"].shape == (n * 9,)
    # all luma == 1 -> every point displaced to depth
    assert np.allclose(out["pos"][:, 2], 1.5)
