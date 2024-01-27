# Timothy Geiger, acse-tfg22

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

setup(
    name='lung_cancer',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    license="MIT",
    author='Timothy Geiger',
    author_email='tfg22@imperial.ac.uk',
    description='This package comprises of utility methods for data' +
    'visualization, quantum and classical classifiers and data wrappers.',
    install_requires=[],
    packages=['lung_cancer']
)
