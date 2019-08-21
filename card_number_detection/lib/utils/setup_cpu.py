import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Build import cythonize
import os
from os.path import join as pjoin

def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib')}
    for k, v in cudaconfig.items():
    #for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig

CUDA = locate_cuda()


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


setup(ext_modules=cythonize("bbox.pyx"),include_dirs=[numpy_include])
setup(ext_modules=cythonize("cython_nms.pyx"),include_dirs=[numpy_include])
#setup(ext_modules=cythonize('nms_kernel.cu', 'gpu_nms.pyx'),library_dirs=[CUDA['lib64']],libraries=['cudart'],language='c++',include_dirs = [numpy_include, CUDA['include']])
setup(ext_modules=cythonize('gpu_nms.pyx'),include_dirs = [numpy_include, CUDA['include']])
