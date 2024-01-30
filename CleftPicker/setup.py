from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("pick",  ["pick.py"]),
]

setup(
    name = 'Cleft Point Picker',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)