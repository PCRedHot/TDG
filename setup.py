from setuptools import setup
from setuptools.discovery import PackageFinder, PEP420PackageFinder

find_packages = PackageFinder.find
find_namespace_packages = PEP420PackageFinder.find

setup(
    name='tdg',
    version='1.0',
    description='TDG Project',
    author='Parry Choi',
    author_email='parrychoi1109@gmail.com',
    packages=find_packages(include=("tdg", "tdg.*")),
    package_data={
        # "pysudoku.resources.images": ["*.png"],
        # "pysudoku.resources.fonts": ["*.tff"],
    },
    install_requires=[
        # "pygame>=2.2.0"        
    ]
)