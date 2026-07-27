```@meta
CollapsedDocStrings = true
CurrentModule = TensorKitTensors.FermionOperators
```

# Fermion operators

Operators for a single *spinless* fermionic mode per site, i.e. a two-dimensional local space that is either empty or occupied.

## Conventions

### Basis ordering

The local space is spanned by the empty and the occupied state, in that order:

```math
|0\rangle,\; |1\rangle \quad \text{(row/column 1 = empty, 2 = occupied)}
```

The space is *always* fermionically graded, `Vect[fℤ₂](0 => 1, 1 => 1)`, even for `Trivial` symmetry: the grading by the fermion parity ``(-1)^n`` is what makes TensorKit insert the anticommutation signs when operators on different sites are contracted, and is therefore not optional.
Here `Trivial` refers only to the absence of an *additional* symmetry.

Because a `TensorMap` on a graded space only has parity-preserving blocks, the parity-odd single-site operators ``f^+`` and ``f^-`` are not representable at all — they have no allowed block.
This module therefore provides no `f_plus` or `f_min`, only their parity-even two-site combinations and the number operator ``n = f^+ f^- = \mathrm{diag}(0, 1)``.

The two-site operators carry the signs picked up by anticommuting the fermionic operators past each other, with ``|1,1\rangle = f^+_1 f^+_2 |0,0\rangle`` as the reference state.
Their only nonzero matrix elements are

```math
f^+_1 f^-_2 : \; +|1,0\rangle \leftarrow |0,1\rangle, \qquad
f^-_1 f^+_2 : \; -|0,1\rangle \leftarrow |1,0\rangle
```

```math
f^+_1 f^+_2 : \; +|1,1\rangle \leftarrow |0,0\rangle, \qquad
f^-_1 f^-_2 : \; -|0,0\rangle \leftarrow |1,1\rangle
```

that is, `f_min_f_plus == -adjoint(f_plus_f_min)` and `f_min_f_min == -adjoint(f_plus_f_plus)`.
Consequently the hermitian hopping operator is the *difference*

```math
f_\mathrm{hop} = f^+_1 f^-_2 - f^-_1 f^+_2 = f^+_1 f^-_2 + (f^+_1 f^-_2)^\dagger .
```

The corresponding bosonic operator of [`BosonOperators`](bosonoperators.md) is a *sum*, because ``b^-_1 b^+_2`` is the plain adjoint of ``b^+_1 b^-_2`` and does not pick up a sign.

### Symmetry sectors

| Symmetry | Physical meaning | Sector label | Single-site space |
|---|---|---|---|
| `Trivial` | fermion parity only (always present) | ``fℤ₂`` charge ``n \bmod 2`` | `Vect[fℤ₂](0 => 1, 1 => 1)` |
| `U1Irrep` | particle-number conservation | ``fℤ₂ ⊠ U(1)`` charge ``(n \bmod 2,\, n)`` | `Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1)` |

!!! note "The U(1) refinement is free"
    The particle number refines the parity grading without reordering the basis, so the basis transformation onto the symmetric space is the identity and every symmetric operator is elementwise equal to its `Trivial` counterpart.

    [`f_num`](@ref), [`f_plus_f_min`](@ref), [`f_min_f_plus`](@ref) and [`f_hopping`](@ref) conserve the particle number and are available for both symmetries.
    The pair operators [`f_plus_f_plus`](@ref) and [`f_min_f_min`](@ref) change it by ``\pm 2``: they preserve the parity, and hence exist as `Trivial` operators, but throw an `ArgumentError` when requested with `U1Irrep`.

## Operator overview

| Function | Alias | Sites | Supported symmetries |
|---|---|---|---|
| [`fermion_space`](@ref) | — | — | `Trivial`, `U1Irrep` |
| [`basis_transform`](@ref) | — | — | `Trivial`, `U1Irrep` |
| [`f_num`](@ref) | `n` | 1 | `Trivial`, `U1Irrep` |
| [`f_plus_f_min`](@ref) | `f⁺f⁻` | 2 | `Trivial`, `U1Irrep` |
| [`f_min_f_plus`](@ref) | `f⁻f⁺` | 2 | `Trivial`, `U1Irrep` |
| [`f_plus_f_plus`](@ref) | `f⁺f⁺` | 2 | `Trivial` |
| [`f_min_f_min`](@ref) | `f⁻f⁻` | 2 | `Trivial` |
| [`f_hopping`](@ref) | `f_hop` | 2 | `Trivial`, `U1Irrep` |

There is deliberately no single-site `f_plus`/`f_min`, see the basis-ordering section above.

## API

```@autodocs
Modules = [FermionOperators]
```
