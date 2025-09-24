from setuptools import setup, find_packages

setup(
    name="automation",
    version="0.2.0",
    author="Red Hat AI Inference Research",
    description="Automation tools",
    url="https://github.com/neuralmagic/research",
    package_dir={"": "src"},
    packages=find_packages(
        "src", include=["automation", "automation.*"], exclude=["*.__pycache__.*"]
    ),
    install_requires=[
        "google-cloud-storage>=1.13.2",
        "datasets",
        "pyhocon",
    ],
    python_requires=">=3.7",
    extras_require={
        "clearml": ["clearml==1.14.4"],
    }
)
