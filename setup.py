from setuptools import find_packages, setup

NAME = "sim_vehicle_ecu"

setup(
    name=NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[("share/ament_index/resource_index/packages", [f"resource/{NAME}"]),
                (f"share/{NAME}", ["package.xml"])],
    zip_safe=False,
    # the shipped control components (scensim roadmap 091); a spec names them by
    # powertrain.model, a user's own by its entry point in the same group
    entry_points={"scensim.vehicle_components": [
        "ideal = sim_vehicle_ecu.component:Ideal", "ev = sim_vehicle_ecu.component:Ev",
        "fmu = sim_vehicle_ecu.fmu:Fmu"]},
    install_requires=["setuptools"],
    tests_require=["pytest"],
)
