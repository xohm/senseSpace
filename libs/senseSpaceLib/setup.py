from setuptools import setup

setup(
    name="senseSpaceLib",
    version="0.1.0",
    description="Shared protocol and data structures for senseSpace server and client.",
    author="Max Rheiner",
    author_email="max.rheiner@zhdk.ch",
    # Explicitly specify the package
    packages=["senseSpaceLib"],
    # Tell setuptools where to find it
    package_dir={"senseSpaceLib": "senseSpaceLib"},
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.7",
)
