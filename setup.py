from setuptools import find_packages, setup

NAME = "sim_vehicle_ecu"

setup(
    name=NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[("share/ament_index/resource_index/packages", [f"resource/{NAME}"]),
                (f"share/{NAME}", ["package.xml"])],
    zip_safe=False,
    install_requires=["setuptools"],
    tests_require=["pytest"],
)
