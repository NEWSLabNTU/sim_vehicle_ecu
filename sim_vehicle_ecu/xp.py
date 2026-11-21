"""The array namespaces a batched component computes in (scensim roadmap 092 S3's `array_api`,
K2.2c): "numpy" (the reference) or "torch" (on a device, so a batch on a GPU never leaves it).

    xp = namespace("torch", "cuda")
    a = xp.zeros(n); b = xp.where(mask, a, xp.full(n, 1.0))

A batch class written once over `xp` computes the same numbers in either: only elementwise
+ - * /, abs, min/max, floor, where and comparisons, each one exactly rounded in IEEE double
(numpy, torch on CPU, and torch's CUDA kernels, one op per kernel: nothing is fused), so a
torch batch equals the numpy one bit for bit, on CPU and on a GPU alike (the tests check it).
numpy's and torch's own exp, log, tanh and atan2 are not in it: they differ in the last bit.

`any`, `nonzero` and `item` bring a value to the host: on a GPU each is a synchronisation, so a
batch's per-step path should need none (a mask applied with `where` costs none).

numpy is imported here only when asked for; torch likewise.
"""

NAMES = ("numpy", "torch")


def namespace(name="numpy", device=None):
    """The namespace `name` ("numpy" or "torch"), torch's on `device` (default: cpu)."""
    if name == "numpy":
        if device not in (None, "cpu"):
            raise ValueError(f"numpy computes on the cpu, not {device!r}")
        return _Numpy()
    if name == "torch":
        return _Torch(device or "cpu")
    raise ValueError(f"array namespace {name!r}: one of {', '.join(NAMES)}")


def supports(cls_or_batch, name):
    """Whether a batch (class) declares the namespace `name` in its `array_api` (a name or a
    tuple of names)."""
    api = getattr(cls_or_batch, "array_api", "numpy")
    return name in ((api,) if isinstance(api, str) else tuple(api))


class _Numpy:
    name = "numpy"
    device = "cpu"

    def __init__(self):
        import numpy as np
        self.np = np
        self.float64, self.int64, self.bool = np.float64, np.int64, np.bool_
        self.inf, self.nan = np.inf, np.nan

    def asarray(self, x, dtype=None):
        return self.np.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return self.np.zeros(shape, dtype or self.float64)

    def ones(self, shape, dtype=None):
        return self.np.ones(shape, dtype or self.float64)

    def full(self, shape, value, dtype=None):
        return self.np.full(shape, value, dtype or self.float64)

    def arange(self, n):
        return self.np.arange(n)

    def where(self, c, a, b):
        return self.np.where(c, a, b)

    def minimum(self, a, b):
        return self.np.minimum(a, b)

    def maximum(self, a, b):
        return self.np.maximum(a, b)

    def abs(self, a):
        return self.np.abs(a)

    def floor(self, a):
        return self.np.floor(a)

    def isnan(self, a):
        return self.np.isnan(a)

    def isinf(self, a):
        return self.np.isinf(a)

    def isfinite(self, a):
        return self.np.isfinite(a)

    def astype(self, a, dtype):
        return a.astype(dtype)

    def copy(self, a):
        return a.copy()

    def broadcast(self, a, n):
        """`a` (a number or [N]) as a fresh float64 array [n]."""
        return self.np.broadcast_to(self.np.asarray(a, dtype=self.float64), (n,)).copy()

    def repeat_cols(self, a, k):
        """[N] -> [N, k], each row `a`'s value."""
        return self.np.repeat(a[:, None], k, axis=1)

    def any(self, a):
        return bool(self.np.any(a))

    def nonzero(self, a):
        """The indices where `a` is true, as host integers (a sync on a device)."""
        return [int(i) for i in self.np.flatnonzero(a)]

    def index(self, idx):
        return self.np.asarray(idx, dtype=self.np.int64)

    def to_host(self, a):
        return self.np.asarray(a)


class _Torch:
    name = "torch"

    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.float64, self.int64, self.bool = torch.float64, torch.int64, torch.bool
        self.inf, self.nan = float("inf"), float("nan")

    def asarray(self, x, dtype=None):
        if self.torch.is_tensor(x):
            return x.to(device=self.device, dtype=dtype or x.dtype)
        import numpy as np   # a host array or a number: through numpy, which keeps its dtype
        a = np.asarray(x)
        t = self.torch.as_tensor(a, device=self.device)
        return t if dtype is None else t.to(dtype)

    def zeros(self, shape, dtype=None):
        return self.torch.zeros(shape, dtype=dtype or self.float64, device=self.device)

    def ones(self, shape, dtype=None):
        return self.torch.ones(shape, dtype=dtype or self.float64, device=self.device)

    def full(self, shape, value, dtype=None):
        return self.torch.full(_shape(shape), value, dtype=dtype or self.float64,
                               device=self.device)

    def arange(self, n):
        return self.torch.arange(n, device=self.device)

    def where(self, c, a, b):
        t = self.torch
        ref = next((x for x in (a, b) if t.is_tensor(x)), None)
        dt = ref.dtype if ref is not None else self.float64
        a = a if t.is_tensor(a) else t.full_like(c, a, dtype=dt)
        b = b if t.is_tensor(b) else t.full_like(c, b, dtype=dt)
        return t.where(c, a, b)

    def minimum(self, a, b):
        a, b = self._pair(a, b)
        return self.torch.minimum(a, b)

    def maximum(self, a, b):
        a, b = self._pair(a, b)
        return self.torch.maximum(a, b)

    def _pair(self, a, b):
        t = self.torch
        if not t.is_tensor(a):
            a = t.full_like(b, a)
        if not t.is_tensor(b):
            b = t.full_like(a, b)
        return a, b

    def abs(self, a):
        return self.torch.abs(a)

    def floor(self, a):
        return self.torch.floor(a)

    def isnan(self, a):
        return self.torch.isnan(a)

    def isinf(self, a):
        return self.torch.isinf(a)

    def isfinite(self, a):
        return self.torch.isfinite(a)

    def astype(self, a, dtype):
        return a.to(dtype)

    def copy(self, a):
        return a.clone()

    def broadcast(self, a, n):
        return self.asarray(a, self.float64).expand(n).clone()

    def repeat_cols(self, a, k):
        return a[:, None].expand(a.shape[0], k).clone()

    def any(self, a):
        return bool(a.any())

    def nonzero(self, a):
        return [int(i) for i in self.torch.nonzero(a).flatten().tolist()]

    def index(self, idx):
        return self.torch.as_tensor(idx, dtype=self.torch.int64, device=self.device)

    def to_host(self, a):
        return a.detach().cpu().numpy()


def _shape(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)
