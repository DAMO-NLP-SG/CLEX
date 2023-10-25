from setuptools import setup, find_packages
setup(
    name="CLEX",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'transformers==4.29.1',
        'torchdiffeq',
    ],

)