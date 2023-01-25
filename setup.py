from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/EYH0602/NumericalSolutions/issues',
    'Documentation': 'https://github.com/EYH0602/numerical_methods/blob/main/README.md',
    'Source Code': 'https://github.com/EYH0602/numerical_methods'
}

setup(name='numerical_methods',
      description='A Numerical Computing Package',
      author='Yifeng He',
      long_description=long_description,
      long_description_content_type="text/markdown",
      project_urls=PROJECT_URLS,
      author_email='yfhe@ucdavis.edu',
      version='0.0.1', 
      packages=find_packages(),
      python_requires='>=3.8')