from setuptools import setup, find_packages


setup(
    name="articlequality-transformer",
    version="0.1.0",
    author_email="ml@wikimedia.org",
    description="Transformer for articlequality models",
    python_requires=">=3.6",
    packages=find_packages("articlequality-transformer"),
)