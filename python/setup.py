
from distutils.core import setup
from distutils.extension import Extension
import os.path
import sys
import numpy

if sys.platform == 'win32' or sys.platform == 'win64':
    print 'Windows is not a supported platform.'
    quit()

else:
    try:
        COOLCVMFSROOT=os.environ['SROOT']
    except KeyError:
        COOLCVMFSROOT= "/usr/local/"
    include_dirs = [COOLCVMFSROOT+"/include",
                    numpy.get_include(),
                    '/usr/local/include',
                    '../inc/',
                    '.']
    libraries = ['python2.7','boost_python',
                 'SQuIDS','nuSQuIDS',
                 'gsl','gslcblas','m',
                 'hdf5','hdf5_hl','PhysTools']

    #if sys.platform.startswith('linux'):
    #  libraries.append('cxxrt')
    if sys.platform.startswith('linux'):
      libraries.append('supc++')#'cxxrt'

    library_dirs = ['/usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib/python2.7',
                    '/usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib/python2.7/../',
                    '/usr/local/lib',
                    '/usr/local/lib',
                    '/usr/local/Cellar/gsl/1.15/lib',
                    '/usr/local/opt/szip/lib',
                    '/usr/local/lib',
                    '/usr/local/lib','/home/carguelles/programs/SNOT/local/lib',
                    COOLCVMFSROOT+"/lib",COOLCVMFSROOT+"/lib64"]

files = ['lvsearchpy.cpp']

setup(name = 'lvsearchpy',
      ext_modules = [
          Extension('lvsearchpy',files,
              library_dirs=library_dirs,
              libraries=libraries,
              include_dirs=include_dirs,
              extra_objects=["../mains/lbfgsb.o","../mains/linpack.o"],
              extra_compile_args=['-O3','-fPIC','-std=c++11','-Wno-unused-local-typedef'],
              depends=[]),
          ]
      )

