from setuptools import setup
from Cython.Build import cythonize

setup(
    name='cube traversal script',
    ext_modules=cythonize("cubeloop.pyx"),
    zip_safe=False,
)