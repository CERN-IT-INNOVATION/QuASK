from setuptools import setup

# Read the contents of the README.md file
with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
)
