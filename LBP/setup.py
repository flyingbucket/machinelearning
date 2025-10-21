from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

compile_args = ["-O3", "-march=native", "-fopenmp"]
link_args = ["-fopenmp"]

ext = Extension(
    name="lbp_cy",
    sources=["./CythonAcc/lbp_cy.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    language="c",
)

setup(
    name="lbp_cy",
    ext_modules=cythonize(
        [ext],
        language_level="3",
        compiler_directives=dict(
            boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
        ),
    ),
)
