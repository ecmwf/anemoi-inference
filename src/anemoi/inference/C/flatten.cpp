#include <cstddef>  // for std::size_t
#include <omp.h>    // for OpenMP pragmas


template <class T> inline void par_ascountiguousarray_1d_impl(T* in_ptr, T* out_ptr, std::size_t stride, std::size_t itemsize, std::size_t n)
{
   std::size_t stride_unit = stride / sizeof(T);
   std::size_t itemsize_unit = itemsize / sizeof(T);

   T* __restrict__ in_rptr = in_ptr;
   T* __restrict__ out_rptr = out_ptr;

   #pragma omp parallel for simd
   for (std::size_t i = 0; i < n; ++i)
   {
       out_rptr[i] = in_rptr[i*stride_unit];
   }
}

extern "C" {
    void par_float(float* in_ptr, float* out_ptr, 
                                            std::size_t stride, std::size_t itemsize, std::size_t n)
    {
        par_ascountiguousarray_1d_impl<float>(in_ptr, out_ptr, stride, itemsize, n);
    }
}