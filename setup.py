from setuptools import find_packages, setup

install_requires = [
    'gpytorch >= 1.8.1', 
    'pyro-ppl >= 1.8.0',
    'scipy',
    'numpy',
    'pandas',
    'matplotlib',
    'gym',
    'ipykernel',
    'pyglet',
    'wandb'
]


setup(
    name='mcbo',
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires
)

