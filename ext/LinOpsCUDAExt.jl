"""
CUDA extension for LinOps

"""
module LinOpsCUDAExt

using CUDA
using LinOps: CoordinateSpace

Base.rand(sp::CoordinateSpace{T, N, S}) where {T, N, S <: CuArray} = CUDA.rand(T, size(sp)...)
Base.randn(sp::CoordinateSpace{T, N, S}) where {T, N, S <: CuArray} = CUDA.randn(T, size(sp))

end
