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

### Custom operators

Each operator module exports its own `custom` function as the high-level entry point for constructing custom operators from plain arrays.
Import the module for the relevant physical system, then pass the desired symmetry types explicitly:

```julia
using TensorKit
using TensorKitTensors.SpinOperators
A = [1.0 0.0; 0.0 -1.0]
Z = custom(A, U1Irrep)
```

An `N`-site input is a rank-`2N` array ordered as `(out₁, …, outₙ, in₁, …, inₙ)`.
Hubbard and t-J operators take separate particle and spin symmetries.
For t-J, `slave_fermion=true` transforms an array supplied in the normal t-J reference basis into the slave-fermion representation.

```@docs
symmetrize
desymmetrize
```
