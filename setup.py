from setuptools import setup, find_packages

# Read the contents of the README.md file
with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setup(
    name="QuASK",
    version="2.0.0-alpha1",
    author="Massimiliano Incudini, Francesco Di Marcantonio, Michele Grossi",
    author_email="massimiliano.incudini@univr.it",
    description="Quantum Advantage Seeker with Kernels (QuASK): a software framework to speed up the research in quantum machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CERN-IT-INNOVATION/QuASK",
    packages=find_packages(where='src'),
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[]
)