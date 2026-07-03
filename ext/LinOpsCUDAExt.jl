"""
CUDA extension for LinOps

"""
module LinOpsCUDAExt

using CUDA
using LinOps: StoredCoordinateSpace

Base.rand(sp::StoredCoordinateSpace{T, N, S}) where {T, N, S <: CuArray} = CUDA.rand(T, size(sp)...)
Base.randn(sp::StoredCoordinateSpace{T, N, S}) where {T, N, S <: CuArray} = CUDA.randn(T, size(sp))

end
