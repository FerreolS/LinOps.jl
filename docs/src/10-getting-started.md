# Getting started

`LinOps.jl` represents linear maps as objects with explicit input and output
domains. Operators can be applied like functions, composed with `*`, and
adjointed with `'`.

## A diagonal operator

`LinOpDiag` constructs an elementwise operator from its diagonal coefficients:

```julia
using LinOps

A = LinOpDiag([1.0, 2.0, 3.0])
x = [10.0, 10.0, 10.0]

A * x
# [10.0, 20.0, 30.0]
```

An operator records the shape of the values it accepts and produces:

```julia
inputsize(A)   # (3,)
outputsize(A)  # (3,)
inputspace(A)  # CoordinateSpace{Number, 1, AbstractArray}((3,))
```

Applying an array with an incompatible shape raises an `ArgumentError` rather
than silently producing a result with a different domain.

## Composition and algebra

`B * A` represents the map that applies `A` first and then `B`. Compatible
operators can therefore be composed directly:

```julia
B = LinOpDiag([2.0, 2.0, 2.0])
C = B * A

C * x
# [20.0, 40.0, 60.0]
```

Operators also support scalar multiplication, addition, inversion when
available, and powers. `A \ x` and `A / x` are not general solve operations;
use the operator algebra documented in the reference when selecting an
operation for a particular operator type.

## Adjoint application

Use `A'` for the adjoint operator. For a diagonal operator, the adjoint uses
the complex conjugate of each diagonal coefficient:

```julia
adjoint(A) * x
# [10.0, 20.0, 30.0]
```

The adjoint of a composition reverses the order of application, as expected
for linear maps.

## In-place application

Use `mul!` when the output array is already allocated:

```julia
using LinearAlgebra: mul!

y = similar(x)
mul!(y, A, x)
# y == [10.0, 20.0, 30.0]
```

The output array must belong to the operator's output domain. This makes it
possible to reuse storage in iterative algorithms and to work with array types
provided by compatible backends.

## Optional operators

Some operator families are supplied by package extensions. Query availability
without loading a backend-specific constructor:

```julia
has_operator(:dft)
operator_backend(:dft)
```

The DFT operator requires FFTW, while the NFFT operator requires
NonuniformFFTs. See the reference page for the complete capability API and
the extension-specific operator constructors.
