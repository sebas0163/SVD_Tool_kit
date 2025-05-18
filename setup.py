from setuptools import setup, find_packages

setup(
    name='SVD_Tool_Kit',
    version='0.1',
    description='Copilation of SVD methods (Compact SVD, Generalized SVD, High Order SVD, Joint SVD, Tensor SVD and Quaternion SVD)',
    author='Sebastián Moya Monge',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'quaternion'
    ],
)