from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/EYH0602/csnum/issues",
    "Documentation": "https://github.com/EYH0602/csnum/blob/main/README.md",
    "Source Code": "https://github.com/EYH0602/csnum",
}

setup(
    name="csnum",
    description="csnum - a highly Composable and type Safe NUMerical analysis package ",
    author="Yifeng He",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls=PROJECT_URLS,
    author_email="yfhe@ucdavis.edu",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
