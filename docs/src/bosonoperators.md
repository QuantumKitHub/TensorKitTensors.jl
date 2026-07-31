```@meta
CollapsedDocStrings = true
CurrentModule = TensorKitTensors.BosonOperators
```

# Boson operators

Operators for a truncated bosonic mode, keeping at most `cutoff` bosons per site.
The truncation is not optional: every function in this module takes `cutoff` as a *required* keyword argument, e.g. `b_num(; cutoff = 4)`.

## Conventions

### Basis ordering

The local space is spanned by the occupation-number states in *ascending* order, so that its dimension is ``\mathrm{cutoff} + 1``:

```math
|0⟩,\; |1⟩,\; …,\; |\mathrm{cutoff}⟩ \quad \text{(row/column 1 = vacuum)}
```

In this basis the creation and annihilation operators have the usual matrix elements, with the square root set by the occupation of the *higher* of the two states,

```math
⟨n-1 | b^- | n⟩ = ⟨n | b^+ | n-1⟩ = √n,
\qquad n = 1, …, \mathrm{cutoff}
```

so that ``b^- |0⟩ = 0`` and ``b^+ |\mathrm{cutoff}⟩ = 0``, while ``n = b^+ b^-`` is diagonal with eigenvalues ``0, 1, …, \mathrm{cutoff}``.

!!! warning "The truncated commutator"
    Truncating the mode breaks the canonical commutation relation in the top state:

    ```math
    [b^-, b^+] = 1 - (\mathrm{cutoff}{+}1)\,|\mathrm{cutoff}⟩⟨\mathrm{cutoff}|
    ```

    i.e. ``⟨\mathrm{cutoff} | [b^-, b^+] | \mathrm{cutoff}⟩ = -\mathrm{cutoff}`` instead of ``1``.
    Every operator of this module is exact *within* the truncated space; only relations that involve states above the cutoff are affected.

### Symmetry sectors

| Symmetry | Physical meaning | Sector label | Single-site block structure |
|---|---|---|---|
| `Trivial` | none | — | full ``(\mathrm{cutoff}{+}1)×(\mathrm{cutoff}{+}1)`` matrix |
| `U1Irrep` | boson-number conservation | charge ``n ∈ \{0, …, \mathrm{cutoff}\}`` | ``\mathrm{cutoff}{+}1`` one-dimensional blocks; ``b^+``, ``b^-`` not individually representable |

!!! note "U(1) charge = occupation number"
    The ``U(1)`` charge is the boson number itself, and the charge sectors are ordered as `0:cutoff`, which coincides with the occupation-number basis.
    The basis transformation onto the symmetric space is therefore the identity, and every symmetric operator is elementwise equal to its `Trivial` counterpart.

    Only the boson-number conserving operators are representable: [`b_num`](@ref), [`b_plus_b_min`](@ref), [`b_min_b_plus`](@ref) and [`b_hopping`](@ref).
    The operators that change the boson number — [`b_plus`](@ref), [`b_min`](@ref), [`b_plus_b_plus`](@ref) and [`b_min_b_min`](@ref) — throw an `ArgumentError` when requested with `U1Irrep`.

## Operator overview

| Function | Alias | Sites | Supported symmetries |
|---|---|---|---|
| [`boson_space`](@ref) | — | — | `Trivial`, `U1Irrep` |
| [`basis_transform`](@ref) | — | — | `Trivial`, `U1Irrep` |
| [`b_plus`](@ref) | `b⁺` | 1 | `Trivial` |
| [`b_min`](@ref) | `b⁻` | 1 | `Trivial` |
| [`b_num`](@ref) | `n` | 1 | `Trivial`, `U1Irrep` |
| [`b_plus_b_plus`](@ref) | `b⁺b⁺` | 2 | `Trivial` |
| [`b_plus_b_min`](@ref) | `b⁺b⁻` | 2 | `Trivial`, `U1Irrep` |
| [`b_min_b_plus`](@ref) | `b⁻b⁺` | 2 | `Trivial`, `U1Irrep` |
| [`b_min_b_min`](@ref) | `b⁻b⁻` | 2 | `Trivial` |
| [`b_hopping`](@ref) | `b_hop` | 2 | `Trivial`, `U1Irrep` |

Note that the bosonic hopping operator is the *sum* ``b^+_1 b^-_2 + b^-_1 b^+_2``, in contrast with the fermionic one of [`FermionOperators`](fermionoperators.md), where the anticommutation sign turns the same hermitian combination into a difference.

## API

```@autodocs
Modules = [BosonOperators]
```
