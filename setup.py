import sys
from distutils.core import setup
from os import path
from setuptools import find_namespace_packages

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

packages = find_namespace_packages(
    exclude=("docs", "build.*"))

version = "0.1"
install_requires = [
    "gym",
    "numpy<=1.19.3",
    "matplotlib",
    "pandas",
    "pygame",
    "tqdm",
    "yapf",
    "seaborn",
    "panda3d~=1.10.8",
    "panda3d-gltf",
    "panda3d-simplepbr",
    "pillow",
    "pytest",
    "opencv-python-headless",
]

setup(
    name="playground",
    version=version,
    packages=packages,
    install_requires=install_requires,
    include_package_data=True,
    license="Apache 2.0",
)

