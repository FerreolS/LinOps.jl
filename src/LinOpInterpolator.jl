using InterpolationKernels: Kernel, CatmullRomSpline

"""
    LinOpInterpolator(knots::AbstractRange, p, kernel = CatmullRomSpline(); sparse = true)

Linear operator interpolating samples given at the uniform grid `knots` with `kernel` at the
positions `p`. The output has the shape of `p`. Samples beyond the ends of `knots` take the
end values.

With `sparse = true` the interpolation matrix is stored; otherwise the kernel weights are
recomputed at each application, which needs no extra memory. The stored form is returned as a `LinOpSparse`.

Methods are provided by the `InterpolationKernels` extension, which is loaded together with
that package.

# Examples
```julia
using LinOps, InterpolationKernels
L = LinOpInterpolator(0.0:0.5:4.0, [0.1, 1.3, 2.2])
y = L * sin.(0.0:0.5:4.0)
```
"""
struct LinOpInterpolator{I, O, R <: AbstractRange, P, K} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    knots::R
    positions::P
    kernel::K
end

function LinOps.LinOpInterpolator(
        knots::AbstractRange{T}, p::AbstractArray, kernel::Kernel = CatmullRomSpline{T}(); sparse::Bool = true
    ) where {T}
    Base.require_one_based_indexing(p)
    positions = vec(p)
    sparse && return LinOpSparse(interpolation_matrix(knots, positions, kernel), (length(knots),), size(p))
    return LinOpInterpolator(CoordinateSpace(length(knots)), CoordinateSpace(size(p)), knots, positions, kernel)
end

Base.eltype(A::LinOpInterpolator) = eltype(A.knots)

# Indices `j` of the samples for which `kernel(u - j)` may be nonzero.
interpolator_indices(kernel, u) = (floor(Int, u - length(kernel) / 2) + 1):floor(Int, u + length(kernel) / 2)

# Calls `f(l, j, v)` for every nonzero weight `v = kernel(u_l - j)`, with `j` clamped to the knot range.
@inline function foreach_interpolation_weight(f::F, knots, positions, kernel) where {F}
    T = eltype(knots)
    n = length(knots)
    for (l, pl) in enumerate(positions)
        u = T((pl - first(knots)) / step(knots) + 1)
        for j in interpolator_indices(kernel, u)
            f(l, clamp(j, 1, n), kernel(u - j))
        end
    end
    return nothing
end

function interpolation_matrix(knots, positions, kernel)
    T = eltype(knots)
    nnz = length(positions) * length(interpolator_indices(kernel, zero(T)))
    L = Vector{Int}(undef, nnz)
    C = Vector{Int}(undef, nnz)
    V = Vector{T}(undef, nnz)
    c = 0
    foreach_interpolation_weight(knots, positions, kernel) do l, j, v
        c += 1
        L[c] = l
        C[c] = j
        V[c] = v
    end
    return sparse(L, C, V, length(positions), length(knots))
end

function LinOps.apply_!(y, A::LinOpInterpolator{I, O, R, P, K}, x) where {I, O, R, P, K}
    fill!(y, zero(eltype(y)))
    ry, rx = reshape(y, :), reshape(x, :)
    foreach_interpolation_weight(A.knots, A.positions, A.kernel) do l, j, v
        ry[l] += v * rx[j]
    end
    return y
end

function LinOps.apply_adjoint_!(y, A::LinOpInterpolator{I, O, R, P, K}, x) where {I, O, R, P, K}
    fill!(y, zero(eltype(y)))
    ry, rx = reshape(y, :), reshape(x, :)
    foreach_interpolation_weight(A.knots, A.positions, A.kernel) do l, j, v
        ry[j] += v * rx[l]
    end
    return y
end
