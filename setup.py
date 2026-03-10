# read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="hfpytrace",
    version="0.0.4",
    packages=find_packages(include=["hfpytrace", "hfpytrace.*"]),
    package_data={
        "hfpytrace": [
            "cfg/*.json",
        ],
    },
    install_requires=[
        "loguru",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "matplotlib==3.9.2",
        "xarray==2024.9.0",
        "toml==0.10.2",
        "tqdm==4.66.5",
        "scipy==1.14.1",
        "SciencePlots==2.1.1",
        "pysolar==0.11",
        "requests==2.32.3",
        "beautifulsoup4==4.12.3",
        "lxml==5.3.0",
        "nrlmsise00==0.1.2",
        "PyIRI",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "coverage",
            "scipy",
            "numpy",
            "pandas",
            "matplotlib",
            "pytesseract",
            "loguru",
            "requests",
            "beautifulsoup4",
            "tqdm",
            "lxml",
            "toml",
            "pytz",
            "timezonefinder",
            "SciencePlots",
            "PyIRI",
        ]
    },
    include_package_data=True,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    author="Shibaji Chakraborty",
    author_email="chakras4@erau.edu",
    maintainer="Shibaji Chakraborty",
    maintainer_email="chakras4@erau.edu",
    license="MIT",
    license_files=["LICENSE"],
    description=long_description,
    long_description=long_description,
    keywords=["python", "HF propagation", "ray tracing", "atmospheric science"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    url="https://github.com/shibaji7/trace",
)
