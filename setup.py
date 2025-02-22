import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='nbrefactor',
    version='0.1.6',
    author='Mohamed Shahawy',
    author_email='envious-citizen.0s@icloud.com',
    description='An automation tool to refactor Jupyter Notebooks to Python modules, with code dependency analysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ThunderStruct/nbrefactor',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Code Generators',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Framework :: Jupyter',
        'Framework :: IPython',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent'
    ],
    keywords='jupyter, notebook, refactor, python, cli',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.7',
    install_requires=[
        'graphviz',
        'nbformat',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'jupyter-nbrefactor = nbrefactor.cli:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/ThunderStruct/nbrefactor/issues',
        'Source': 'https://github.com/ThunderStruct/nbrefactor',
    }
)

