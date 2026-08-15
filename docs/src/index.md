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
The symmetric versions are generated automatically by rotating the reference operator with a documented unitary basis transformation and projecting the result onto the symmetric tensor structure.
The basis transformations are exposed through each module's `basis_transform` function, and each module exports `symmetrize_operator` to apply the appropriate transformation and symmetry projection to an operator on its desymmetrized form.
Operators that are incompatible with a given symmetry throw an `ArgumentError`.

### Custom operators

Construct a `TensorMap` on the relevant module's `Trivial` reference space, then pass it to that module's `symmetrize_operator` with the desired symmetry types:

```julia
using TensorKit
using TensorKitTensors.SpinOperators
A = [1.0 0.0; 0.0 -1.0]
V = spin_space(Trivial)
O = TensorMap(A, V ← V)
Z = symmetrize_operator(O, U1Irrep)
```

The caller is responsible for constructing a square homogeneous-site operator with the intended local space and array-axis ordering.

```@docs
symmetrize
desymmetrize
```
