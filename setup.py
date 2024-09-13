from setuptools import setup, find_packages

setup(
    name='nbrefactor',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'jupyter-nbrefactor = nbrefactor.cli:main',
        ],
    },
    install_requires=[
        'graphviz',
        'nbformat',
        'tqdm',
    ],
)
