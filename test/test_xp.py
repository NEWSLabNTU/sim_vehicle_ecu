"""sim_vehicle_ecu.xp, the array namespaces a batched component computes in (092 K2.2c): the
same operations on numpy and on torch give the same values, and a batch that does not declare
a namespace is refused it."""
import numpy as np
import pytest

from sim_vehicle_ecu import batch as B
from sim_vehicle_ecu import xp as X


def ops(xp):
    a = xp.asarray(np.array([1.5, -2.0, np.nan, np.inf]), xp.float64)
    m = xp.asarray(np.array([True, False, True, False]), xp.bool)
    out = [xp.where(m, a, 0.0), xp.minimum(a, 1.0), xp.maximum(-1.0, a), xp.abs(a),
           xp.floor(a), xp.astype(xp.isnan(a), xp.float64), xp.repeat_cols(a, 2).reshape(-1),
           xp.broadcast(2.5, 4), xp.full(3, 7.0), xp.zeros((2, 2)).reshape(-1)]
    return [np.asarray(xp.to_host(v)) for v in out], xp.nonzero(m), xp.any(m)


def test_numpy_and_torch_agree():
    torch = pytest.importorskip("torch")
    n_vals, n_idx, n_any = ops(X.namespace("numpy"))
    t_vals, t_idx, t_any = ops(X.namespace("torch", "cpu"))
    for a, b in zip(n_vals, t_vals):
        assert a.tobytes() == b.astype(a.dtype).tobytes()
    assert n_idx == t_idx == [0, 2] and n_any is t_any is True
    _ = torch


def test_a_namespace_is_one_of_numpy_and_torch():
    with pytest.raises(ValueError, match="one of numpy, torch"):
        X.namespace("jax")
    with pytest.raises(ValueError, match="numpy computes on the cpu"):
        X.namespace("numpy", "cuda")


def test_a_batch_without_it_is_refused():
    class NumpyOnly:
        commands, truth, array_api = "autoware", (), "numpy"

        def __init__(self, vehicle, params, seeds):
            pass

    class Car:
        batch = NumpyOnly
    assert X.supports(NumpyOnly, "numpy") and not X.supports(NumpyOnly, "torch")
    with pytest.raises(B.BatchError, match="no batched form on array_api 'torch'"):
        B.load_batch(Car, {}, {}, [1], array_api="torch")


def test_hypot_is_math_hypot_and_sin_cos_within_a_few_ulp():
    import math
    ns = X.namespace("numpy")
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 5, 20000), rng.normal(0, 5, 20000)
    a[:10] = 0.0
    want = np.array([math.hypot(x, y) for x, y in zip(a.tolist(), b.tolist())])
    assert X.hypot(ns, a, b).tobytes() == want.tobytes()
    x = np.concatenate([rng.uniform(-1.2, 1.2, 20000), [0.0, -0.0]])
    s, c = X.sin_cos(ns, x)
    for got, f in ((s, math.sin), (c, math.cos), (s / c, math.tan)):
        ref = np.array([f(v) for v in x.tolist()])
        assert np.max(np.abs(got - ref) / np.spacing(np.abs(ref) + 1e-300)) <= 4
    assert s[-2:].tobytes() == np.array([0.0, -0.0]).tobytes() and c[-2:].tolist() == [1.0, 1.0]


def test_hypot_and_sin_cos_are_the_same_bits_on_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 5, 5000), rng.uniform(-1.2, 1.2, 5000)
    ns, ts = X.namespace("numpy"), X.namespace("torch", "cpu")
    ta, tb = torch.as_tensor(a), torch.as_tensor(b)
    assert X.hypot(ns, a, b).tobytes() == X.hypot(ts, ta, tb).numpy().tobytes()
    for u, v in zip(X.sin_cos(ns, b), X.sin_cos(ts, tb)):
        assert u.tobytes() == v.numpy().tobytes()
