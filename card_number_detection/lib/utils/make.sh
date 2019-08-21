cython bbox.pyx
cython cython_nms.pyx
cython gpu_nms.pyx
python setup_new.py build_ext --inplace
rm -rf build
