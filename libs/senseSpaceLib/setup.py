from setuptools import setup, find_packages

setup(
    name="senseSpaceLib",
    version="0.1.0",
    description="Shared protocol and data structures for senseSpace server and client.",
    author="Max Rheiner",
    author_email="max.rheiner@zhdk.ch",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.7",
)
