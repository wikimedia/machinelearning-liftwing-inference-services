from setuptools import setup, find_packages


setup(
    name="topic-transformer",
    version="0.1.0",
    author_email="ml@wikimedia.org",
    description="Transformer for topic models",
    python_requires=">=3.6",
    packages=find_packages("topic-transformer"),
)
