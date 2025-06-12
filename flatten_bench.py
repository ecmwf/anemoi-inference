import numpy as np
import time
import ctypes

cflatten =ctypes.CDLL("/ec/res4/hpcperm/naco/raps/build/sources/anemoi-inference/src/anemoi/inference/C/flatten.so")
# Define the argument types
cflatten.par_float.argtypes = [
    ctypes.c_void_p,  # float* A (device pointer)
    ctypes.c_void_p,  # float* B (device pointer)
    ctypes.c_size_t,  # std::size_t stride
    ctypes.c_size_t,  # std::size_t itemsize
    ctypes.c_size_t   # std::size_t n
]
cflatten.par_float.restype = None

def flatten(arr: np.ndarray) -> np.ndarray:
        """
        Convert a strided array to a contiguous array using parallel processing.

        Args:
            arr: Input numpy array (can be strided)

        Returns:
            Contiguous copy of the input array
        """
        if arr.ndim != 1:
            raise ValueError("This function only supports 1D arrays")

        # Get array properties
        dtype_str = str(arr.dtype)
        itemsize = arr.itemsize
        stride = arr.strides[0] if arr.strides else itemsize
        n = arr.shape[0]

        # Map numpy dtypes to function names
        dtype_map = {
            'float32': 'float32',
            #'float64': 'float64',
            #'int32': 'int32',
            #'int64': 'int64',
        }

        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        # Create output array (contiguous)
        out_arr = np.empty(arr.shape, dtype=arr.dtype)

        # Get data pointers
        in_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        out_ptr = out_arr.ctypes.data_as(ctypes.c_void_p)

        # Call the C++ function
        cflatten.par_float(
            in_ptr,
            out_ptr,
            ctypes.c_size_t(stride),
            ctypes.c_size_t(itemsize),
            ctypes.c_size_t(n)
        )

        return out_arr

#a = np.arange(52428800, dtype=np.float32)[::2]  # Large non-contiguous 1D array

# Each element in the output is 89 elements apart in the base array
size = 5242880
stride_bytes = 356
dtype = np.float32
element_size = np.dtype(dtype).itemsize  # 4 bytes for float32
step = stride_bytes // element_size  # 89 elements

# Create a base array large enough to accommodate the strides
base = np.zeros(size * step, dtype=dtype)

# Fill the base array at intervals of 'step' with random values
base[::step] = np.random.rand(size).astype(dtype)


# Create the strided view
a = np.lib.stride_tricks.as_strided(
    base,
    shape=(size,),
    strides=(stride_bytes,)
)



print("input shape:",a.shape)  # Should be (50000000,)
print("input type:", a.dtype)  # Should be int64   
print("original stride input:", a.strides)  # Should be (16,)

start = time.time()
#b = fcont.ascontiguousarray_1d(a)
b = flatten(a)
print("stride b:", b.strides)
end = time.time()
tim_time = end - start
print("Cathal Time:", tim_time, "seconds")

start = time.time()
c = np.ascontiguousarray(a)
print("stride c:", c.strides)
end = time.time()
cont_time = end - start

print("np.contiguousarray Time:", cont_time, "seconds", "speedup", cont_time / tim_time)

start = time.time()
d = a.flatten()
print("stride d",d.strides)
end = time.time()
flatten_time = end - start
print("np.flatten Time:", flatten_time, "seconds", "speedup" ,flatten_time / tim_time) 

print("vector are identical b d:", np.array_equal(b, d))
print("vector are identical b c:", np.array_equal(b, c))











