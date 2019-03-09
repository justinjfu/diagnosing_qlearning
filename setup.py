import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

def make_extension(package_path, name, include_dirs=None):
    if include_dirs is None:
        include_dirs = []
    cy_dir = os.path.join(*package_path)
    package_prefix = '.'.join(package_path)+'.'
    ext = Extension(package_prefix+name,
                [os.path.join(cy_dir, name+'.pyx')],
                include_dirs=include_dirs)
    return ext

def discover_extensions(root_dir):
    for (dirname, subdir, filenames) in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pyx'):
                pkg = dirname.split(os.sep)
                name = os.path.splitext(filename)[0]
                yield make_extension(pkg, name)
                #print(pkg, name)

extensions = list(discover_extensions('rlutil'))
extensions.extend(list(discover_extensions('debugq')))

setup(
    ext_modules=cythonize(extensions)
)

