"""Timing of the batched form (092 S3): mean ms per step and us per slot-step, N = 1, 256, 2048.
Not a test (not collected); run: python3 test/bench_batch.py"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_vehicle_ecu import batch as B  # noqa: E402
from sim_vehicle_ecu.signals import Request, Sensors, VescCommand  # noqa: E402
from batch_cars import DT, Lag, LagScalar  # noqa: E402
from test_calibration import MODEL3  # noqa: E402


def bench(b, cmds, sens, steps):
    b.reset_batch(np.ones(b.n, bool))
    b.step_batch(cmds, sens, DT)   # warm
    t0 = time.perf_counter()
    for _ in range(steps):
        b.step_batch(cmds, sens, DT)
    return (time.perf_counter() - t0) / steps


def main():
    rows = [("form", "N", "ms/step", "us/slot-step")]
    for n in (1, 256, 2048):
        seeds = list(range(n))
        vc = B.commands_to_batch("vesc", [VescCommand("current", 10.0, 0.6, 0.0)] * n)
        vs = B.sensors_to_batch([Sensors((10.0,) * 4, 0.1, 0.2, 0.0, {"speed": 3.0})] * n,
                                ("speed",))
        ac = B.commands_to_batch("autoware", [Request(1.0, 8.0, 0.05)] * n)
        asn = B.sensors_to_batch([Sensors((20.0,) * 4, 0.01, 1.0, 0.0)] * n)
        steps = 200 if n < 2048 else 40
        for label, b, c, s in (
                ("vectorised (noise: a Python Random per slot)", B.load_batch(Lag, {}, {}, seeds), vc, vs),
                ("vectorised, no noise", B.load_batch(Lag, {}, {"noise": 0.0}, seeds), vc, vs),
                ("LoopBatch of the same car (scalar)", B.load_batch(LagScalar, {}, {}, seeds), vc, vs),
                ("LoopBatch of ev", B.load_batch("ev", MODEL3, None, seeds), ac, asn)):
            t = bench(b, c, s, steps if "ev" not in label else max(5, steps // 8))
            rows.append((label, n, f"{t * 1e3:.3f}", f"{t * 1e6 / n:.2f}"))
    w = [max(len(str(r[i])) for r in rows) for i in range(4)]
    for r in rows:
        print("  ".join(str(x).ljust(k) for x, k in zip(r, w)))


if __name__ == "__main__":
    main()
