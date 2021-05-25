from setuptools import setup, find_packages

tests_require = [
    'pytest',
    'pytest-tornasync',
    'mypy'
]

setup(
    name='outlink-transformer',
    version='0.1.0',
    author_email='acraze@wikimedia.org',
    license='../../COPYING',
    description='Transformer for Outlink Topic Model',
    long_description=open('README.md').read(),
    python_requires='>=3.6',
    packages=find_packages("outlink-transformer"),
    install_requires=[
        "kfserving>=0.2.1",
        "argparse>=1.4.0",
        "mwapi"
    ],
    tests_require=tests_require,
    extras_require={'test': tests_require}
)
