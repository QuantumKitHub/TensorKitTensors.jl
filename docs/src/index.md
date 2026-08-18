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

For each operator module, one can define a custom operator in `TensorMap` format in the non-symmetric reference basis, and then impose the symmetries with that module's `symmetrize_operator` function:

```julia
using TensorKit
using TensorKitTensors.SpinOperators
V = spin_space(Trivial)
O = TensorMap([1.0 0.0; 0.0 -1.0], V ← V)
Z = symmetrize_operator(O, U1Irrep)
```

```@docs
symmetrize
desymmetrize
```

## Adding symmetry charges

Use [`add_charge`](@ref) to attach symmetry charges to each site of the operator by fusing a one-dimensional auxiliary space to every local space of an operator.

For example, the following adds a uniform U(1) charge of `1 // 2` to the local spaces of a U(1)-symmetric spin-exchange operator:

```julia
using TensorKit
using TensorKitTensors
using TensorKitTensors.SpinOperators

O = S_exchange(U1Irrep)
O_shifted = add_charge(O, U1Irrep(1 // 2))
```

```@docs
add_charge
```
