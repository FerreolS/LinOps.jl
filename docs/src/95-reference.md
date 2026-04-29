# [Reference](@id reference)

## Contents

```@contents
Pages = ["95-reference.md"]
```

## Index

```@index
Pages = ["95-reference.md"]
```

```@autodocs
Modules = [LinOps]
```

## Optional Operators

Some operators are backend-dependent and become available only when optional
dependencies are loaded.

- DFT operator: requires FFTW
- NFFT operator: requires NonuniformFFTs

Check capability at runtime:

```julia
LinOps.has_operator(:dft)
LinOps.has_operator(:nfft)
LinOps.has_operator(LinOps.LinOpDFT)
LinOps.has_operator(LinOps.LinOpNFFT)
```

Inspect selected backend:

```julia
LinOps.operator_backend(:dft)
LinOps.operator_backend(:nfft)
LinOps.operator_backend(LinOps.LinOpDFT)
LinOps.operator_backend(LinOps.LinOpNFFT)
```
