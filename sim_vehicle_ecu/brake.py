"""The brake module's slip control: ABS and traction control share one modulator. Pure."""


class SlipModulator:
    """Slip-based release and reapply of one actuator, as ABS does a wheel's brake pressure
    and traction control the motor's torque: a factor in [0, 1] that falls at `release_rate`
    (1/s) while the slip is past `release`, climbs back at `apply_rate` once it is within
    `reapply`, and creeps back at a quarter of that in between (a pure hold there can park
    the factor below 1 for good while the tire works at its peak). The slip is a magnitude in
    the direction limited."""

    def __init__(self, release, reapply, release_rate, apply_rate):
        self.release, self.reapply = release, reapply
        self.release_rate, self.apply_rate = release_rate, apply_rate
        self.factor = 1.0

    @property
    def active(self):
        return self.factor < 1.0

    def step(self, slip, dt, enabled=True):
        if not enabled:
            self.factor = min(1.0, self.factor + self.apply_rate * dt)
        elif slip > self.release:
            self.factor = max(0.0, self.factor - self.release_rate * dt)
        elif slip < self.reapply:
            self.factor = min(1.0, self.factor + self.apply_rate * dt)
        else:
            self.factor = min(1.0, self.factor + 0.25 * self.apply_rate * dt)
        return self.factor

    def reset(self):
        self.factor = 1.0
