# sim_vehicle_ecu: a simulated car's onboard computers (VCU, brake module with ABS, TC and ESC,
# power steering), below its vehicle interface. Pure Python, the standard library only.

default:
    @just --list

# The unit tests (no simulator, no ROS)
test:
    python3 -m pytest -q -p no:cacheprovider test

# The tests, then lint (pyflakes)
check: test
    python3 -m pyflakes sim_vehicle_ecu test
