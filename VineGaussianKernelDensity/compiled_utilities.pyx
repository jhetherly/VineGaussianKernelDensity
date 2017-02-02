# distutils: language = c++
# distutils: libraries = ["m"]

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

# cdef data_to_numpy_array_with_spec_size_t(void * ptr, np.npy_intp N):
#     cdef np.ndarray[size_t, ndim=1] arr = np.PyArray_SimpleNewFromData(
#                                                     1, &N, np.NPY_UINT, ptr)
#     # NOTE: transfer ownership of array
#     PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
#     return arr

cdef extern from "WeightedGaussKDE_KD_Tree.hpp":
    cdef cppclass WeightedGaussKDE_KD_Tree_1D:
        WeightedGaussKDE_KD_Tree_1D() except +
        WeightedGaussKDE_KD_Tree_1D(const unsigned int&) except +
        void SetPointsAndWeightsAdaptive (const double *points,
                                          const double *weights,
                                          const double *max_var,
                                          const size_t &n_points,
                                          const double &tol) except +
        void SetPointsAndWeightsAdaptive (const double *points,
                                          const double *weights,
                                          const double *max_var,
                                          const size_t &n_points) except +
        void SetPointsAndWeightsGlobal (const double *points,
                                        const double *weights,
                                        const double &max_var,
                                        const size_t &n_points,
                                        const double &tol) except +
        void SetPointsAndWeightsGlobal (const double *points,
                                        const double *weights,
                                        const double &max_var,
                                        const size_t &n_points) except +
        size_t* RadiusSearch (size_t &match_size,
                              const double *sc,
                              const double &eps) except +
        size_t* RadiusSearch (size_t &match_size,
                              const double *sc) except +

cdef class PyWeightedGaussKDE_KD_Tree_1D:
    # hold a pointer to the C++ instance which we're wrapping
    cdef WeightedGaussKDE_KD_Tree_1D* c_wgkde
    def __cinit__(self, const unsigned int &max_leaf):
        self.c_wgkde = new WeightedGaussKDE_KD_Tree_1D(max_leaf)
    def __dealloc__(self):
        del self.c_wgkde
    def SetPointsAndWeights(self,
                            np.ndarray[double, ndim=1, mode='c'] points,
                            np.ndarray[double, ndim=1, mode='c'] weights,
                            np.ndarray[double, ndim=1, mode='c'] max_var,
                            double tol=1e-8):
        self.c_wgkde.SetPointsAndWeightsAdaptive(&points[0], &weights[0],
                                                 &max_var[0], points.size,
                                                 tol)
    def SetPointsAndWeights(self,
                            np.ndarray[double, ndim=1, mode='c'] points,
                            np.ndarray[double, ndim=1, mode='c'] weights,
                            double max_var,
                            double tol=1e-8):
        self.c_wgkde.SetPointsAndWeightsGlobal(&points[0], &weights[0],
                                               max_var, points.size,
                                               tol)
    # def GetAllWithinTolerance (self, )
