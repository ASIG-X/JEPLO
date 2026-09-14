from setuptools import setup, find_packages

setup(
    name="deepstats",
    version="1.0.0",
    url="https://github.com/rbalestr-lab/deepstats",
    author="Randall Balestriero",
    author_email="randallbalestriero@gmail.com",
    description="ToDo",
    packages=find_packages(),
    install_requires=["torch", "numpy", "loguru", "pytest"],
)
