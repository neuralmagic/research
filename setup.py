from setuptools import setup, find_packages
import os
import sys

setup(
    name="automation",
    version="0.1.0",
    author="NM MLR",
    description="Automation tools",
    #url="https://github.com/neuralmagic/research",
    package_dir={"": "src"},
    packages=find_packages(
        "src", include=["automation", "automation.*"], exclude=["*.__pycache__.*"]
    ),
    install_requires=[
        "clearml==1.14.4",
        #"google-cloud-storage>=1.13.2",
        "datasets",
        "pyhocon",
        "build>=1.0.0",
        "setuptools>=61.0",
        "setuptools-git-versioning>=2.0,<3",
    ],
    python_requires=">=3.7",
)
