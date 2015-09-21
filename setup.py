#!/usr/bin/env python
# -*- coding: utf-8
from __future__ import division, print_function
'''Setup script'''

import glob
import numpy
import os
import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

SOURCE = '.'
os.chdir(SOURCE)

#Uncomment on mac envs for openmp
os.environ["CC"] = "gcc-5"
os.environ["CXX"] = "gcc-5"

if sys.version_info[:2] < (2, 7):
    print('Requires Python version 2.7 or later (%d.%d detected).' %
          sys.version_info[:2])
    sys.exit(-1)

def get_packages():
    '''Appends all packages (based on recursive sub dirs)'''

    packages  = ['fpmc']

    for package in packages:
        base = os.path.join(package, '**/')
        sub_dirs = glob.glob(base)
        while len(sub_dirs) != 0:
            for sub_dir in sub_dirs:
                package_name = sub_dir.replace('/', '.')
                if package_name.endswith('.'):
                    package_name = package_name[:-1]

                packages.append(package_name)
        
            base = os.path.join(base, '**/')
            sub_dirs = glob.glob(base)

    return packages

def get_extensions():
    '''Get's all .pyx and.pxd files'''

    extensions = []
    packages = get_packages()

    for pkg in packages:
        pkg_folder = pkg.replace('.', '/')
        pyx_files = glob.glob(os.path.join(pkg_folder, '*.pyx'))
        include_dirs = ['fpmc/myrandom/', numpy.get_include()]
        for pyx in pyx_files:
            pxd = pyx.replace('pyx', 'pxd')
            module = pyx.replace('.pyx', '').replace('/', '.')

            if os.path.exists(pxd):
                ext_files = [pyx, pxd]
            else:
                ext_files = [pyx]
            
            if module == 'fpmc.myrandom.random':
                ext_files.append(os.path.join(pkg_folder, 'randomkit.c'))
            
            extension = Extension(module, ext_files,
                    include_dirs=include_dirs,
                    extra_compile_args=['-msse', '-msse2', '-mfpmath=sse', \
                            '-fopenmp', '-Wno-unused-function'], #cython warnings supress
                    extra_link_args=['-fopenmp'])

            extensions.append(extension)

    return extensions

if __name__ == "__main__":
    packages = get_packages()
    extensions = get_extensions()
    setup(cmdclass = {'build_ext': build_ext},
            name             = 'fpmc',
            packages         = packages,
            ext_modules      = extensions)
