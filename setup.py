# setup.py
from setuptools import setup, find_packages

setup(
    name="das_io",
    version="0.1.0",
    packages=find_packages(where="src"),  # look in src/
    package_dir={"": "src"},               # root for packages is src/
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires='>=3.8',
)