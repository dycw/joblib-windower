from __future__ import annotations

from pathlib import Path
from re import MULTILINE
from re import search

from setuptools import find_packages
from setuptools import setup


with open("README.md") as fd:
    long_description = fd.read()
    header, description = long_description.splitlines()
    name = search(r"^# ([\w-]+)$", header).group(1)
for path in Path(__file__).resolve().parent.iterdir():
    try:
        with open(str(path.joinpath("__init__.py"))) as fd:
            version = search(
                r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]$', fd.read(), MULTILINE,
            ).group(1)
    except (AttributeError, FileNotFoundError, NotADirectoryError):
        pass
with open("requirements/core.txt") as fd:
    install_requires = fd.read().strip().split("\n")


setup(
    name=name,
    version=version,
    author="Bao Wei",
    author_email="baowei.ur521@gmail.com",
    license="MIT",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    python_requires=">=3.7",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
)
