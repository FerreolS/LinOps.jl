"""
CUDA extension for LinOps

"""
module LinOpsCUDAExt

using CUDA
using LinOps: DeviceTypedCoordinateSpace

Base.rand(sp::DeviceTypedCoordinateSpace{T, N, S}) where {T, N, S <: CuArray} = CUDA.rand(T, size(sp)...)
Base.randn(sp::DeviceTypedCoordinateSpace{T, N, S}) where {T, N, S <: CuArray} = CUDA.randn(T, size(sp))

end
