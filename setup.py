from setuptools import find_packages, setup

setup(
    name="eluvio-vertical-focus",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "common-ml @ git+https://github.com/eluv-io/common-ml.git@5e112f615acb0a13fa044ea4dacc94ab1612d9ef",
        "loguru>=0.7,<1",
        "mediapipe>=0.10.14,<0.11",
        "numpy>=1.24,<2",
        "opencv-contrib-python>=4.9,<5",
        "PyYAML>=6,<7",
        "requests>=2.31,<3",
    ],
)
