```@meta
CollapsedDocStrings = true
CurrentModule = TensorKitTensors.HubbardOperators
```

# Hubbard operators

Creation, annihilation, number and spin operators for Hubbard-type models, defined on the four-dimensional local space of a spinful fermion, with independent symmetries for the particle number and for the spin.

## Conventions

### Basis ordering

A Hubbard site carries four states,

```math
|0⟩, \quad |↑⟩ = e^†_↑ |0⟩, \quad
|↓⟩ = e^†_↓ |0⟩, \quad
|↑↓⟩ = e^†_↑ e^†_↓ |0⟩,
```

of which the two singly occupied ones are fermionic.
The local space is therefore always graded by `FermionParity`, and since a graded space groups its basis vectors per sector, in the order of `sectors(V)`, the *index* order is not the order in which the states are listed above but the parity-sorted one:

```math
\underbrace{|0⟩,\; |↑↓⟩}_{\text{parity even}}, \quad
\underbrace{|↑⟩,\; |↓⟩}_{\text{parity odd}}
```

This is the order in which `block(op, FermionParity(0))` and `block(op, FermionParity(1))` list their rows and columns — with `Trivial` particle and spin symmetry both blocks are ``2 × 2``, so there is no single ``4 × 4`` matrix — and it is the reference order of [`basis_transform`](@ref), which is consequently the identity for `Trivial` symmetry.

For the symmetric versions the dense order follows the sector order of the target space and is in general *not* the reference order.
Rather than relying on it, read it off from the basis transformation: numbering the reference states ``(1, 2, 3, 4) = (|0⟩, |↑↓⟩, |↑⟩, |↓⟩)``, column ``j`` of `basis_transform(P, S)` has its single nonzero entry in the row that is the dense index of reference state ``j``:

```julia
U = convert(Array, basis_transform(U1Irrep, SU2Irrep))
i_up = findfirst(==(1), U[:, 3])   # dense index of |↑⟩
```

### Symmetry sectors

The particle-number and the spin symmetry are chosen independently.
The local sector type is always `FermionParity ⊠ (particle) ⊠ (spin)` with the `Trivial` factors omitted and the spin factor last, so e.g. `hubbard_space(U1Irrep, SU2Irrep)` is `Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1//2) => 1, (0, 2, 0) => 1)`.
The supported sectors are `Trivial`, `U1Irrep` and `SU2Irrep` on both axes; anything else, `Z2Irrep` included, throws an `ArgumentError`.

The particle symmetry labels the parity-even pair (empty, doubly occupied):

| Symmetry | Physical meaning | Sector label | Charges of the local states |
|---|---|---|---|
| `Trivial` | none | — | empty and doubly occupied share the parity-even sector, with multiplicity 2 |
| `U1Irrep` | particle-number conservation | ``n ∈ \{0, 1, 2\}`` | empty ``↦ 0``, singly occupied ``↦ 1``, doubly occupied ``↦ 2`` |
| `SU2Irrep` | ``η``-pairing SU(2) | ``η ∈ \{0, 1/2\}`` | (doubly occupied, empty) is the ``η = 1/2`` doublet, ordered by descending ``η^z = (n-1)/2``; singly occupied has ``η = 0`` |

The spin symmetry labels the parity-odd pair (the two singly occupied states):

| Symmetry | Physical meaning | Sector label | Charges of the local states |
|---|---|---|---|
| `Trivial` | none | — | the two singly occupied states share the parity-odd sector, with multiplicity 2 |
| `U1Irrep` | ``S^z`` conservation | ``m ∈ \{-1/2, 0, +1/2\}`` | ``↑ ↦ +1/2``, ``↓ ↦ -1/2``, empty and doubly occupied ``↦ 0`` |
| `SU2Irrep` | full spin SU(2) | ``s ∈ \{0, 1/2\}`` | the singly occupied states form the ``s = 1/2`` doublet (descending ``m``); empty and doubly occupied are ``s = 0`` |

!!! note "Charge conventions"
    The `U1Irrep` particle charge is the particle number ``n``, not the particle-hole symmetric ``n - 1``, and the `U1Irrep` spin charge is ``S^z = ±1/2``, not ``2S^z = ±1``.
    The ``η``-pairing `SU2Irrep` is built on ``η^z = (n-1)/2`` instead, so the particle number itself is *not* an ``η``-SU(2) scalar: this is why [`e_num`](@ref) and [`ud_num`](@ref) are unavailable with `SU2Irrep` particle symmetry, while the particle-hole symmetric [`half_ud_num`](@ref) ``= (n_↑ - 1/2)(n_↓ - 1/2)`` — which equals ``+1/4`` on the ``η`` doublet and ``-1/4`` on the singly occupied states — is available for every symmetry combination.

!!! note "Staggered gauge for SU(2) particle symmetry"
    The ``η``-pairing SU(2) only commutes with the Hubbard Hamiltonian after the staggered gauge transformation ``e_{j,σ} → i^j e_{j,σ}`` on a bipartite lattice.
    Accordingly, with `SU2Irrep` particle symmetry the operators of this module act on site ``k`` with the additional gauge factor ``G^{k-1}``, where ``G = i^n = \mathrm{diag}(1, -1, i, i)`` in the reference order above.
    Operators that commute with ``G`` are returned without it, so they stay real-representable and remain elementwise equal to their `Trivial` counterparts.
    Of the operators available with `SU2Irrep` particle symmetry, [`e_hopping`](@ref) is the only one that does not commute with ``G``: it is therefore genuinely complex, requires a complex `eltype`, and is the gauge-transformed version of its `Trivial` counterpart — it generates the same physics on a bipartite lattice, but is **not** elementwise equal to it.

    The ``η``-pairing SU(2) and the gauge transformation that makes it a symmetry of the Hubbard model are due to C. N. Yang and S. C. Zhang, *SO₄ symmetry in a Hubbard model*, Mod. Phys. Lett. B **4**, 759 (1990), [doi:10.1142/S0217984990000933](https://doi.org/10.1142/S0217984990000933); see also C. N. Yang, *η pairing and off-diagonal long-range order in a Hubbard model*, Phys. Rev. Lett. **63**, 2144 (1989), [doi:10.1103/PhysRevLett.63.2144](https://doi.org/10.1103/PhysRevLett.63.2144).

## Operator overview

The two symmetry columns are independent conditions: an operator is available for exactly those combinations that satisfy both.
*any* means all three supported sectors, `Trivial`, `U1Irrep` and `SU2Irrep`; every other request throws an `ArgumentError`.

| Function | Alias | Sites | Particle symmetry | Spin symmetry |
|---|---|---|---|---|
| [`hubbard_space`](@ref) | — | — | any | any |
| [`basis_transform`](@ref) | — | — | any | any |
| [`e_num`](@ref) | `n` | 1 | `Trivial`, `U1Irrep` | any |
| [`u_num`](@ref) | `nꜛ` | 1 | `Trivial`, `U1Irrep` | `Trivial`, `U1Irrep` |
| [`d_num`](@ref) | `nꜜ` | 1 | `Trivial`, `U1Irrep` | `Trivial`, `U1Irrep` |
| [`ud_num`](@ref) | `nꜛꜜ` | 1 | `Trivial`, `U1Irrep` | any |
| [`half_ud_num`](@ref) | — | 1 | any | any |
| [`h_num`](@ref) | `nʰ` | 1 | `Trivial`, `U1Irrep` | any |
| [`S_x`](@ref) | `Sˣ` | 1 | any | `Trivial` |
| [`S_y`](@ref) | `Sʸ` | 1 | any | `Trivial` |
| [`S_z`](@ref) | `Sᶻ` | 1 | any | `Trivial`, `U1Irrep` |
| [`S_plus`](@ref) | `S⁺` | 1 | any | `Trivial` |
| [`S_min`](@ref) | `S⁻` | 1 | any | `Trivial` |
| [`u_plus_u_min`](@ref) | `u⁺u⁻` | 2 | `Trivial`, `U1Irrep` | `Trivial`, `U1Irrep` |
| [`u_min_u_plus`](@ref) | `u⁻u⁺` | 2 | `Trivial`, `U1Irrep` | `Trivial`, `U1Irrep` |
| [`d_plus_d_min`](@ref) | `d⁺d⁻` | 2 | `Trivial`, `U1Irrep` | `Trivial`, `U1Irrep` |
| [`d_min_d_plus`](@ref) | `d⁻d⁺` | 2 | `Trivial`, `U1Irrep` | `Trivial`, `U1Irrep` |
| [`e_plus_e_min`](@ref) | `e⁺e⁻` | 2 | `Trivial`, `U1Irrep` | any |
| [`e_min_e_plus`](@ref) | `e⁻e⁺` | 2 | `Trivial`, `U1Irrep` | any |
| [`e_hopping`](@ref) | `e_hop` | 2 | any | any |
| [`u_min_d_min`](@ref) | `u⁻d⁻` | 2 | `Trivial` | `Trivial`, `U1Irrep` |
| [`d_min_u_min`](@ref) | `d⁻u⁻` | 2 | `Trivial` | `Trivial`, `U1Irrep` |
| [`u_plus_d_plus`](@ref) | `u⁺d⁺` | 2 | `Trivial` | `Trivial`, `U1Irrep` |
| [`d_plus_u_plus`](@ref) | `d⁺u⁺` | 2 | `Trivial` | `Trivial`, `U1Irrep` |
| [`u_min_u_min`](@ref) | `u⁻u⁻` | 2 | `Trivial` | `Trivial` |
| [`u_plus_u_plus`](@ref) | `u⁺u⁺` | 2 | `Trivial` | `Trivial` |
| [`d_min_d_min`](@ref) | `d⁻d⁻` | 2 | `Trivial` | `Trivial` |
| [`d_plus_d_plus`](@ref) | `d⁺d⁺` | 2 | `Trivial` | `Trivial` |
| [`singlet_plus`](@ref) | `singlet⁺` | 2 | `Trivial` | any |
| [`singlet_min`](@ref) | `singlet⁻` | 2 | `Trivial` | any |
| [`S_plus_S_min`](@ref) | `S⁺S⁻` | 2 | any | `Trivial`, `U1Irrep` |
| [`S_min_S_plus`](@ref) | `S⁻S⁺` | 2 | any | `Trivial`, `U1Irrep` |
| [`S_exchange`](@ref) | `SS` | 2 | any | any |
| [`singlet_plus_singlet_min_3site`](@ref) | `Δ⁺ij_Δjk` | 3 | `Trivial`, `U1Irrep` | any |
| [`singlet_plus_singlet_min_4site`](@ref) | `Δ⁺ij_Δkl` | 4 | `Trivial`, `U1Irrep` | any |

Note that the singlet-pair terms [`singlet_plus_singlet_min_3site`](@ref) and [`singlet_plus_singlet_min_4site`](@ref) are available with `U1Irrep` particle symmetry even though [`singlet_plus`](@ref) and [`singlet_min`](@ref) individually are not: only the product ``Δ^† Δ`` conserves the particle number.
[`S_y`](@ref) requires a complex `eltype` for every symmetry combination, and [`e_hopping`](@ref) requires one with `SU2Irrep` particle symmetry; all other operators honour any `eltype`, and the basis transformations have exact integer entries so that they never degrade the precision of the result.

The t-J restriction of this module, obtained by projecting out the doubly occupied state, is [`TJOperators`](tjoperators.md).

## API

```@autodocs
Modules = [HubbardOperators]
```
