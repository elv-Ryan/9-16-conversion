from setuptools import find_packages, setup

setup(
    name="eluvio-nba-yolo-shot-tagger",
    version="0.3.0",
    description="Eluvio-compatible NBA shot-to-normalized-X YOLO tagger",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.11",
)
