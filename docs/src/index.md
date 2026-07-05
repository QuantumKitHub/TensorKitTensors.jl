```@meta
CurrentModule = TensorKitTensors
```

# TensorKitTensors

Documentation for [TensorKitTensors](https://github.com/QuantumKitHub/TensorKitTensors.jl).
This is a lightweight package that defines several commonly used tensors for TensorKit, with various symmetries.

```@contents
Pages = Main.operatorpages
```

## Symmetric operators through basis transformations

Each operator module defines its operators only once, in a non-symmetric reference basis.
The symmetric versions are generated automatically by rotating the reference operator with a
documented unitary basis transformation and projecting the result onto the symmetric tensor
structure. The basis transformations are exposed through each module's `basis_transform`
function (e.g. the Hadamard matrix that maps the ``S^z`` basis onto the ``ℤ₂`` spin-flip
eigenbasis for `SpinOperators`), and the projection is available as [`symmetrize`](@ref) for
symmetrizing custom operators. Operators that are incompatible with a given symmetry throw
an `ArgumentError`.

```@docs
symmetrize
desymmetrize
```
