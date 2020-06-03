from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from re import MULTILINE
from re import search
from typing import List

from setuptools import find_packages
from setuptools import setup


@dataclass
class MetaData:
    name: str
    version: str
    description: str
    long_description: str
    install_requires: List[str]


def _get_metadata() -> MetaData:
    with open("README.md") as file:
        long_description = file.read()
    header, description = long_description.splitlines()
    if match := search(r"^# ([\w-]+)$", header):
        name = match.group(1)
    else:
        raise ValueError("Unable to determine 'name' from README.md")
    path = (
        Path(__file__)
        .resolve()
        .parent.joinpath(name.replace("-", "_"), "__init__.py")
    )
    with open(path) as file:
        if (
            match := search(
                r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]$',
                file.read(),
                MULTILINE,
            )
        ) :
            version = match.group(1)
        else:
            raise ValueError(f"Unable to determine 'version' from {path}")
    with open("requirements/core.txt") as file:
        install_requires = file.read().strip().split("\n")
    return MetaData(
        name=name,
        version=version,
        description=description,
        long_description=long_description,
        install_requires=install_requires,
    )


_METADATA = _get_metadata()


setup(
    name=_METADATA.name,
    version=_METADATA.version,
    author="Bao Wei",
    author_email="baowei.ur521@gmail.com",
    license="MIT",
    description=_METADATA.description,
    long_description=_METADATA.long_description,
    long_description_content_type="text/markdown",
    install_requires=_METADATA.install_requires,
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
)
