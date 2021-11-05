from setuptools import setup, find_packages

tests_require = ["pytest", "pytest-tornasync", "mypy"]

setup(
    name="outlink-transformer",
    version="0.1.0",
    author_email="ml@wikimedia.org",
    description="Transformer for Outlink Topic Model",
    python_requires=">=3.6",
    packages=find_packages("outlink-transformer"),
)
