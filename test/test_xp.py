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
