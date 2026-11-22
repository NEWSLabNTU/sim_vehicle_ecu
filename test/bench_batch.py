"""Timing of the batched form (092 S3): mean ms per step and us per slot-step, N = 1, 256, 2048;
then `ev`: its LoopBatch (N scalar Ecu) against EvBatch on numpy and torch (CPU, and CUDA when
there is one) at N = 16 and 1024. Not a test (not collected); run: python3 test/bench_batch.py"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_vehicle_ecu import batch as B  # noqa: E402
from sim_vehicle_ecu import component as comp  # noqa: E402
from sim_vehicle_ecu.signals import Request, Sensors, VescCommand  # noqa: E402
from batch_cars import DT, Lag, LagScalar  # noqa: E402
from test_calibration import MODEL3  # noqa: E402


def bench(b, cmds, sens, steps, sync=None):
    b.reset_batch(np.ones(b.n, bool))
    b.step_batch(cmds, sens, DT)   # warm
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        b.step_batch(cmds, sens, DT)
    if sync:
        sync()
    return (time.perf_counter() - t0) / steps


def ev_rows():
    """`ev`: the scalar Ecu per slot (LoopBatch) against EvBatch, numpy and torch."""
    rows = []
    try:
        import torch
    except ImportError:
        torch = None
    for n in (16, 1024):
        seeds = list(range(n))
        ac = B.commands_to_batch("autoware", [Request(-3.0, 0.0, 0.05)] * n)
        asn = B.sensors_to_batch([Sensors((20.0, 20.0, 19.0, 19.0), 0.1, -2.5, 0.5)] * n)
        forms = [("LoopBatch of ev (scalar Ecu per slot)",
                  B.LoopBatch([comp.load("ev", MODEL3, None, s) for s in seeds], "ev"), None,
                  "numpy")] if n <= 16 else []
        forms.append(("EvBatch, numpy", B.load_batch("ev", MODEL3, None, seeds), None, "numpy"))
        if torch is not None:
            devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
            for dev in devs:
                forms.append((f"EvBatch, torch {dev}",
                              B.load_batch("ev", MODEL3, None, seeds, array_api="torch",
                                           device=dev),
                              torch.cuda.synchronize if dev == "cuda" else None, dev))
        for label, b, sync, dev in forms:
            c, s = ac, asn
            if dev != "numpy":
                to = lambda d: {k: (to(v) if isinstance(v, dict) else  # noqa: E731
                                    torch.as_tensor(np.asarray(v), device=dev))
                                for k, v in d.items()}
                c, s = to(ac), to(asn)
            t = bench(b, c, s, 20 if "Loop" in label else 100, sync)
            rows.append((label, n, f"{t * 1e3:.3f}", f"{t * 1e6 / n:.2f}"))
    return rows


def main():
    rows = [("form", "N", "ms/step", "us/slot-step")]
    for n in (1, 256, 2048):
        seeds = list(range(n))
        vc = B.commands_to_batch("vesc", [VescCommand("current", 10.0, 0.6, 0.0)] * n)
        vs = B.sensors_to_batch([Sensors((10.0,) * 4, 0.1, 0.2, 0.0, {"speed": 3.0})] * n,
                                ("speed",))
        steps = 200 if n < 2048 else 40
        for label, b, c, s in (
                ("vectorised (noise: a Python Random per slot)", B.load_batch(Lag, {}, {}, seeds), vc, vs),
                ("vectorised, no noise", B.load_batch(Lag, {}, {"noise": 0.0}, seeds), vc, vs),
                ("LoopBatch of the same car (scalar)", B.load_batch(LagScalar, {}, {}, seeds), vc, vs)):
            t = bench(b, c, s, steps)
            rows.append((label, n, f"{t * 1e3:.3f}", f"{t * 1e6 / n:.2f}"))
    rows += ev_rows()
    w = [max(len(str(r[i])) for r in rows) for i in range(4)]
    for r in rows:
        print("  ".join(str(x).ljust(k) for x, k in zip(r, w)))


if __name__ == "__main__":
    main()
