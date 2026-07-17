```@meta
CollapsedDocStrings = true
CurrentModule = TensorKitTensors.SUNOperators
```

# SU(N) operators

Operators for SU(N) "spin" models, where each site carries a fixed SU(N) irrep. These
operators require [SUNRepresentations.jl](https://github.com/QuantumKitHub/SUNRepresentations.jl)
to be loaded (`using SUNRepresentations`); they are provided through a package extension.

## Conventions

### Specifying the irrep

The local irrep is selected with the `irrep` keyword, a **weight tuple** whose length fixes
the rank `N`. For example `irrep=(1, 0)` is the SU(2) fundamental (spin-½) and
`irrep=(1, 0, 0)` the SU(3) fundamental. Two-site operators may couple two different irreps
with `irreps=(irrep1, irrep2)`.

### Construction

Following the package-wide convention, each operator is defined in the non-symmetric
(`Trivial`) reference basis and the symmetric (`SUNIrrep`) version is obtained by
[`symmetrize`](@ref TensorKitTensors.symmetrize). The dense `exchange` operator is the
polarized quadratic Casimir ``\sum_a T^a \otimes T^a`` built from the SU(N) generators; SU(N)'s
Gelfand–Tsetlin basis coincides with the generator basis, so [`basis_transform`](@ref) is the
identity.

## Operator overview

| Function | Sites | Supported symmetries |
|---|---|---|
| [`sun_space`](@ref) | — | `Trivial`, `SUNIrrep` |
| [`basis_transform`](@ref) | — | `Trivial`, `SUNIrrep` |
| [`exchange`](@ref) | 2 | `Trivial`, `SUNIrrep` |
| [`swap`](@ref) | 2 | `Trivial`, `SUNIrrep` |
| [`biquadratic`](@ref) | 2 | `Trivial`, `SUNIrrep` |

## API

```@autodocs
Modules = [SUNOperators]
```
