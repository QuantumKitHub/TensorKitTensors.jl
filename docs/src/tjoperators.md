```@meta
CurrentModule = TensorKitTensors.TJOperators
CollapsedDocStrings = true
```

# t-J operators

Creation, annihilation, number and spin operators for t-J-type models, i.e. the Hubbard model with the doubly occupied state projected out, with independent symmetries for the electron number and for the spin.

## Conventions

### Basis ordering

A t-J site carries three states — one empty and two singly occupied — with no doubly occupied state:

```math
|0⟩, \quad |↑⟩, \quad |↓⟩
```

Here ``|0⟩`` labels the *empty* state rather than the vacuum ``|∅⟩`` of the underlying operators, a distinction that matters in the slave-fermion basis below.
The absence of double occupancy is exact: the product of [`u_num`](@ref) and [`d_num`](@ref) vanishes identically.

The local space is graded by `FermionParity`, and a graded space groups its basis vectors per sector, in the order of `sectors(V)`.
In the default t-J basis the empty state is fermion-parity even and the singly occupied states are odd, so the parity-sorted index order coincides with the natural one.
The slave-fermion basis flips the parity of every state and hence reverses the two groups:

```math
\text{t-J:}\quad |0⟩,\; |↑⟩,\; |↓⟩
\qquad\qquad
\text{slave-fermion:}\quad |↑⟩,\; |↓⟩,\; |0⟩
```

Each of these is the reference order of [`basis_transform`](@ref) in the corresponding basis, so `basis_transform(Trivial, Trivial; slave_fermion)` is the identity for either value of `slave_fermion`.

For the symmetric versions the dense order follows the sector order of the target space and is in general *not* the reference order.
Rather than relying on it, read it off from the basis transformation: numbering the reference states ``(1, 2, 3) = (|0⟩, |↑⟩, |↓⟩)``, column ``j`` of `basis_transform(P, S; slave_fermion)` has its single nonzero entry in the row that is the dense index of reference state ``j``:

```julia
U = convert(Array, basis_transform(U1Irrep, SU2Irrep))
i_up = findfirst(==(1), U[:, 2])   # dense index of |↑⟩
```

### Slave-fermion basis

Every function of this module takes an optional `slave_fermion::Bool = false` keyword that selects the basis it is expressed in.
In the slave-fermion representation the hole is created by a *fermionic* holon operator ``h``, and the spins by *bosonic* spinon operators `bꜛ` and `bꜜ`, acting on the vacuum ``|∅⟩``:

```
| label | tJ basis | slave-fermion |
| ----- | -------- | ------------- |
|  |0⟩  |   |∅⟩    |      h⁺|∅⟩    |
|  |↑⟩  |  u⁺|∅⟩   |     bꜛ⁺|∅⟩    |
|  |↓⟩  |  d⁺|∅⟩   |     bꜜ⁺|∅⟩    |
```

[`transform_slave_fermion`](@ref) performs this change of basis, on a space or on an operator, by fusing in a single auxiliary fermionic charge.
This flips the fermion parity of every state and thereby the statistics of the operator: braiding the auxiliary legs of an ``N``-site operator through the physical ones generates a staggered sign ``(-1)^{(k-1)p_k}`` on site ``k``, with ``p_k`` the parity of the state.

Which operators exist does not depend on `slave_fermion`; only the basis they are expressed in does.

!!! warning "The slave-fermion transformation does not distribute over `⊗`"
    Because of that staggered sign, [`transform_slave_fermion`](@ref) has to be applied to a *complete* operator: in general `transform_slave_fermion(A ⊗ B)` differs from `transform_slave_fermion(A) ⊗ transform_slave_fermion(B)`.
    Build a multi-site term in the plain t-J basis and transform once, at the very end, which is exactly what the operators of this module do — so `op(elt, P, S; slave_fermion = true)` always agrees with `transform_slave_fermion(op(elt, P, S))`, but assembling it site by site from `slave_fermion = true` single-site operators does not.

### Relation to the Hubbard model

The t-J local space is the Hubbard local space with the doubly occupied state removed.
[`tj_projector`](@ref) is the corresponding ``3 ← 4`` isometry, with `Int` entries so that it introduces no floating-point error, and every operator of this module is the projection of its [`HubbardOperators`](hubbardoperators.md) namesake:

```julia
proj = reduce(⊗, ntuple(Returns(tj_projector(P, S)), N))
TJOperators.op(elt, P, S) ≈ proj * HubbardOperators.op(elt, P, S) * proj'
```

for an ``N``-site operator `op`.
The projector is defined in the plain t-J basis only: the slave-fermion basis has no Hubbard counterpart.

Because double occupancy is removed rather than merely energetically penalized, this module exports no on-site interaction: there is no `ud_num` and no `half_ud_num`.
For the same reason [`h_num`](@ref) and [`e_num`](@ref) are complementary here, ``n^h + n = 1``, whereas in the Hubbard model they overshoot the identity by the double occupancy.

### Symmetry sectors

The electron-number and the spin symmetry are chosen independently.
The local sector type is always `FermionParity ⊠ (particle) ⊠ (spin)` with the `Trivial` factors omitted and the spin factor last, so e.g. `tj_space(U1Irrep, SU2Irrep)` is `Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1//2) => 1)`.

The particle symmetry labels the empty state against the singly occupied ones:

| Symmetry | Physical meaning | Sector label | Charges of the local states |
|---|---|---|---|
| `Trivial` | none | — | the fermion-parity grading alone separates the empty state from the singly occupied ones |
| `U1Irrep` | electron-number conservation | ``n ∈ \{0, 1\}`` | empty ``↦ 0``, singly occupied ``↦ 1`` |
| `SU2Irrep` | — | — | not available, see the note below |

The spin symmetry labels the two singly occupied states:

| Symmetry | Physical meaning | Sector label | Charges of the local states |
|---|---|---|---|
| `Trivial` | none | — | the two singly occupied states share one sector, with multiplicity 2 |
| `U1Irrep` | ``S^z`` conservation | ``m ∈ \{-1/2, 0, +1/2\}`` | ``↑ ↦ +1/2``, ``↓ ↦ -1/2``, empty ``↦ 0`` |
| `SU2Irrep` | full spin SU(2) | ``s ∈ \{0, 1/2\}`` | the singly occupied states form the ``s = 1/2`` doublet (descending ``m``); the empty state is ``s = 0`` |

!!! note "No SU(2) particle symmetry"
    Unlike [`HubbardOperators`](hubbardoperators.md), the t-J model admits no `SU2Irrep` particle symmetry.
    The ``η``-pairing doublet of the Hubbard model is (doubly occupied, empty), and the doubly occupied state is precisely what the t-J projection removes, so there is no doublet left to carry the ``η``-spin.
    `tj_space(SU2Irrep, S)` and every operator requested with `SU2Irrep` particle symmetry throw an `ArgumentError`, as does any unsupported sector such as `Z2Irrep` on either axis.

!!! note "Slave-fermion sectors"
    The slave-fermion transformation fuses in a single auxiliary `FermionParity(1)` charge together with the identity charge of every other factor, so it leaves the particle and spin labels untouched and only flips the parity factor.
    The two bases therefore carry the same sector labels up to that flip, but their sectors come out in a different *order*: in general `basis_transform(P, S; slave_fermion = true)` is a different permutation from `basis_transform(P, S)`.

## Operator overview

The two symmetry columns are independent conditions: an operator is available for exactly those combinations that satisfy both.
*any* means every supported sector, i.e. `Trivial` and `U1Irrep` for the electron number, and `Trivial`, `U1Irrep` and `SU2Irrep` for the spin.
Every other request throws an `ArgumentError`.
Availability does not depend on `slave_fermion`, and every entry below accepts that keyword.

| Function | Alias | Sites | Particle symmetry | Spin symmetry |
|---|---|---|---|---|
| [`tj_space`](@ref) | — | — | any | any |
| [`basis_transform`](@ref) | — | — | any | any |
| [`tj_projector`](@ref) | — | — | any | any |
| [`transform_slave_fermion`](@ref) | — | — | — | — |
| [`e_num`](@ref) | `n` | 1 | any | any |
| [`u_num`](@ref) | `nꜛ` | 1 | any | `Trivial`, `U1Irrep` |
| [`d_num`](@ref) | `nꜜ` | 1 | any | `Trivial`, `U1Irrep` |
| [`h_num`](@ref) | `nʰ` | 1 | any | any |
| [`S_x`](@ref) | `Sˣ` | 1 | any | `Trivial` |
| [`S_y`](@ref) | `Sʸ` | 1 | any | `Trivial` |
| [`S_z`](@ref) | `Sᶻ` | 1 | any | `Trivial`, `U1Irrep` |
| [`S_plus`](@ref) | `S⁺` | 1 | any | `Trivial` |
| [`S_min`](@ref) | `S⁻` | 1 | any | `Trivial` |
| [`u_plus_u_min`](@ref) | `u⁺u⁻` | 2 | any | `Trivial`, `U1Irrep` |
| [`u_min_u_plus`](@ref) | `u⁻u⁺` | 2 | any | `Trivial`, `U1Irrep` |
| [`d_plus_d_min`](@ref) | `d⁺d⁻` | 2 | any | `Trivial`, `U1Irrep` |
| [`d_min_d_plus`](@ref) | `d⁻d⁺` | 2 | any | `Trivial`, `U1Irrep` |
| [`e_plus_e_min`](@ref) | `e⁺e⁻` | 2 | any | any |
| [`e_min_e_plus`](@ref) | `e⁻e⁺` | 2 | any | any |
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
| [`singlet_plus_singlet_min_3site`](@ref) | `Δ⁺ij_Δjk` | 3 | any | any |
| [`singlet_plus_singlet_min_4site`](@ref) | `Δ⁺ij_Δkl` | 4 | any | any |

Note that the singlet-pair terms [`singlet_plus_singlet_min_3site`](@ref) and [`singlet_plus_singlet_min_4site`](@ref) are available for every symmetry combination even though [`singlet_plus`](@ref) and [`singlet_min`](@ref) individually require `Trivial` particle symmetry: only the product ``Δ^† Δ`` conserves the electron number.
[`S_y`](@ref) requires a complex `eltype`; all other operators honour any `eltype`, and the basis transformations have exact integer entries so that they never degrade the precision of the result.

## API

Every operator docstring below is generated from that of its `HubbardOperators` namesake, so
the two modules cannot drift apart; see [Relation to the Hubbard model](@ref).

```@autodocs
Modules = [TJOperators]
```
