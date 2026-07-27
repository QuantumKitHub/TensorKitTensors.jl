module TJOperators

using LinearAlgebra: diagind
using TensorKit
import ..HubbardOperators
import ..TensorKitTensors: symmetrize, desymmetrize, fuse_local_operators, @operator

export tj_space, basis_transform, tj_projector
export e_num, u_num, d_num, h_num
export S_x, S_y, S_z, S_plus, S_min
export u_plus_u_min, d_plus_d_min
export u_min_u_plus, d_min_d_plus
export u_min_d_min, d_min_u_min
export u_plus_d_plus, d_plus_u_plus
export u_min_u_min, d_min_d_min
export u_plus_u_plus, d_plus_d_plus
export e_plus_e_min, e_min_e_plus, e_hopping
export singlet_plus, singlet_min
export singlet_plus_singlet_min_3site, singlet_plus_singlet_min_4site
export S_plus_S_min, S_min_S_plus, S_exchange

export nꜛ, nꜜ, nʰ, n
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻
export u⁺u⁻, d⁺d⁻, u⁻u⁺, d⁻d⁺
export u⁻d⁻, d⁻u⁻, u⁺d⁺, d⁺u⁺
export u⁻u⁻, u⁺u⁺, d⁻d⁻, d⁺d⁺
export e⁺e⁻, e⁻e⁺, e_hop
export singlet⁺, singlet⁻
export Δ⁺ij_Δjk, Δ⁺ij_Δkl
export S⁻S⁺, S⁺S⁻, SS

export transform_slave_fermion

const _docs_basis_table = """
```
| label | tJ basis | slave-fermion |
| ----- | -------- | ------------- |
|  |0⟩  |   |∅⟩    |      h⁺|∅⟩    |
|  |↑⟩  |  u⁺|∅⟩   |     bꜛ⁺|∅⟩    |
|  |↓⟩  |  d⁺|∅⟩   |     bꜜ⁺|∅⟩    |
```
"""

# Spaces
# ------
@doc """
    tj_space([particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the local hilbert space for a t-J-type model with the given particle and spin symmetries.
The basis consists of the following states:

$_docs_basis_table

- `|∅⟩` is the vacuum state;
- `u` and `d` denote fermionic spin-up and spin-down operators;
- in the slave-fermion representation, ``h`` is the fermionic holon operator, and ``bꜛ``, ``bꜜ`` are bosonic spinon operators.

The possible symmetries are:
- Particle number : `Trivial`, `U1Irrep`
- Spin            : `Trivial`, `U1Irrep`, `SU2Irrep`.

Use `slave_fermion = true` to switch to the slave-fermion basis, which flips the fermion
parity of every state; see [`transform_slave_fermion`](@ref).
""" tj_space
function tj_space(
        particle_symmetry::Type{<:Sector} = Trivial,
        spin_symmetry::Type{<:Sector} = Trivial;
        slave_fermion::Bool = false
    )
    V = _tj_space(particle_symmetry, spin_symmetry)
    return slave_fermion ? transform_slave_fermion(V) : V
end

# the t-J space in the plain basis; dispatched on the symmetries, with the `slave_fermion`
# keyword handled once by `tj_space` above
_tj_space(::Type{Trivial}, ::Type{Trivial}) = Vect[FermionParity](0 => 1, 1 => 2)
function _tj_space(::Type{Trivial}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
end
function _tj_space(::Type{Trivial}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 1, (1, 1 // 2) => 1)
end
function _tj_space(::Type{U1Irrep}, ::Type{Trivial})
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2)
end
function _tj_space(::Type{U1Irrep}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep](
        (0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1
    )
end
function _tj_space(::Type{U1Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1)
end
# the η-pairing doublet of the Hubbard model is (|↑↓⟩, |0⟩), which has no t-J counterpart
function _tj_space(::Type{SU2Irrep}, spin_symmetry::Type{<:Sector})
    throw(ArgumentError("t-J model does not have ``SU(2)`` particle symmetry."))
end
function _tj_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    throw(ArgumentError("invalid symmetry `($particle_symmetry, $spin_symmetry)`"))
end

"""
    basis_transform(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the unitary basis transformation that maps the basis of
`tj_space(Trivial, Trivial; slave_fermion)` onto the basis of
`tj_space(particle_symmetry, spin_symmetry; slave_fermion)`, as a `TensorMap` between the
desymmetrized versions of these spaces (see
[`desymmetrize`](@ref TensorKitTensors.desymmetrize)), as required by
[`symmetrize`](@ref TensorKitTensors.symmetrize).

For all symmetry combinations the transformation is a permutation, determined by the sector
order of the target space, where the states are identified as follows:

- For `U1Irrep` particle symmetry, the number of electrons is used as charge, distinguishing
  ``|0⟩`` (charge 0) from ``|↑⟩`` and ``|↓⟩`` (charge 1).
- For `U1Irrep` spin symmetry, the ``S^z`` eigenvalue ``±1/2`` is used as charge,
  distinguishing ``|↑⟩`` from ``|↓⟩``.
- For `SU2Irrep` spin symmetry, ``(|↑⟩, |↓⟩)`` forms the spin doublet (descending ``m``).

Both bases order the states by fermion parity, so the reference basis differs between them:

```
| basis          | reference order |
| -------------- | --------------- |
| t-J            | |0⟩, |↑⟩, |↓⟩   |
| slave-fermion  | |↑⟩, |↓⟩, |0⟩   |
```

The transformations have exact integer entries and are therefore returned with integer
scalar type, such that they promote to any scalar type without loss of precision.
"""
function basis_transform(
        particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        slave_fermion::Bool = false
    )
    reference = _dense_state_order(Trivial, Trivial; slave_fermion)
    U = zeros(Int, 3, 3)
    for (row, state) in enumerate(_dense_state_order(particle_symmetry, spin_symmetry; slave_fermion))
        U[row, findfirst(==(state), reference)] = 1
    end
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return TensorMap(U, desymmetrize(V) ← desymmetrize(tj_space(Trivial, Trivial; slave_fermion)))
end

# the states (|0⟩, |↑⟩, |↓⟩) = (1, 2, 3) in the dense order of
# `tj_space(particle_symmetry, spin_symmetry; slave_fermion)`
function _dense_state_order(
        particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        slave_fermion::Bool = false
    )
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    states = Int[]
    for c in sectors(V)
        # a one-dimensional auxiliary space maps `c → c ⊠ aux` while preserving the
        # multiplicity order, so fusing in the auxiliary charge again (it is self-inverse)
        # recovers the plain-basis sector that identifies the states
        c′ = slave_fermion ? only(c ⊗ slave_fermion_auxiliary_charge(typeof(c))) : c
        append!(states, _state_indices(c′, particle_symmetry, spin_symmetry))
    end
    return states
end

# t-J basis indices (|0⟩, |↑⟩, |↓⟩) = (1, 2, 3) contained in sector `c`, in dense row order
function _state_indices(c, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    parity = c isa FermionParity ? c : c[1]
    return if !parity.isodd # parity even: |0⟩
        (1,)
    elseif spin_symmetry === Trivial || spin_symmetry === SU2Irrep
        (2, 3)
    else # U1Irrep: Sᶻ distinguishes the states (spin is the last sector factor)
        spin_sector = particle_symmetry === Trivial ? c[2] : c[3]
        spin_sector.charge > 0 ? (2,) : (3,)
    end
end

# Symmetrize a t-J operator through its basis transformation, in the requested basis. Note
# that `slave_fermion` is forwarded from the reference operator, which is already expressed
# in that basis (see `_maybe_slave_fermion`), such that only a permutation remains here.
function _symmetrize_operator(
        O::AbstractTensorMap, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}; kwargs...
    )
    return symmetrize(
        O, basis_transform(particle_symmetry, spin_symmetry; kwargs...),
        tj_space(particle_symmetry, spin_symmetry; kwargs...)
    )
end

"""
    tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Projection operator from Hubbard space to t-J space, which removes the doubly occupied state
``|↑↓⟩``. The t-J operators of this module are related to their `HubbardOperators`
counterparts by

```julia
proj = reduce(⊗, ntuple(Returns(tj_projector(P, S)), N))
TJOperators.op(elt, P, S) ≈ proj * HubbardOperators.op(elt, P, S) * proj'
```

for an `N`-site operator `op`. The scalartype is `Int` to avoid floating point errors.
"""
function tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    Vhub = HubbardOperators.hubbard_space(particle_symmetry, spin_symmetry)
    VtJ = tj_space(particle_symmetry, spin_symmetry)
    proj = zeros(Int, Vhub → VtJ)

    # the retained states are the leading rows of every shared sector: for `Trivial` particle
    # symmetry `|↑↓⟩` is the *second* even state of the Hubbard space, and for `U1Irrep` it
    # lives in a sector that has no t-J counterpart at all, so no block of `proj` contains it
    for (f1, f2) in fusiontrees(proj)
        proj[f1, f2][diagind(proj[f1, f2])] .= 1
    end
    return proj
end

# Slave-fermion basis
# -------------------
slave_fermion_auxiliary_charge(::Type{FermionParity}) = FermionParity(1)
slave_fermion_auxiliary_charge(::Type{ProductSector{T}}) where {T} =
    mapreduce(⊠, fieldtypes(T)) do I
    I === FermionParity ? FermionParity(1) : one(I)
end

@doc """
    transform_slave_fermion(O::AbstractTensorMap)
    transform_slave_fermion(V::ElementarySpace)

Transform the given operator to the slave-fermion basis, which is related to the usual t-J basis by

$_docs_basis_table

where ``h`` is the fermionic holon operator, and ``bꜛ``, ``bꜜ`` are bosonic spinon operators.

Fusing in the auxiliary fermionic charge flips the parity of every state, which changes the
statistics of the operator: braiding the auxiliary legs of an ``N``-site operator through the
physical ones generates a staggered sign ``(-1)^{(k-1)p_k}`` on site ``k``, with ``p_k`` the
parity of the state. Consequently this transformation has to be applied to a complete
operator, and does not commute with taking tensor products of single-site operators.
""" transform_slave_fermion
function transform_slave_fermion(O::AbstractTensorMap)
    (N = numin(O)) == numout(O) || throw(ArgumentError("not a valid operator"))
    aux_charge = slave_fermion_auxiliary_charge(sectortype(O))
    aux_space = spacetype(O)(aux_charge => 1)
    aux_operator = id(Int, aux_space^N)

    return fuse_local_operators(O, aux_operator)
end
function transform_slave_fermion(V::ElementarySpace)
    charge = slave_fermion_auxiliary_charge(sectortype(V))
    V_aux = spacetype(V)(charge => 1)
    return fuse(V, V_aux)
end

# Express a reference operator in the requested basis. Reference operators are always built up
# from single-site and multi-site operators in the plain t-J basis, and transformed only once,
# at the very end: the slave-fermion transformation does not distribute over `⊗`.
_maybe_slave_fermion(O::AbstractTensorMap, slave_fermion::Bool) =
    slave_fermion ? transform_slave_fermion(O) : O

function n_site_operator(::Val{N}, elt::Type{<:Number}) where {N}
    V = tj_space(Trivial, Trivial)
    return zeros(elt, V^N ← V^N)
end

# Single-site operators
# ---------------------
"""
    u_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    nꜛ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the one-body operator that counts the number of spin-up electrons.
"""
@operator nꜛ function u_num(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), I(1))][1, 1] = 1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    d_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    nꜜ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the one-body operator that counts the number of spin-down electrons.
"""
@operator nꜜ function d_num(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), I(1))][2, 2] = 1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    e_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    n([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the one-body operator that counts the number of electrons.
"""
@operator n function e_num(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = u_num(elt, Trivial, Trivial) + d_num(elt, Trivial, Trivial)
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    h_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    nʰ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the one-body operator that counts the number of holes, i.e. the number of
non-occupied sites.
"""
@operator nʰ function h_num(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = id(elt, tj_space(Trivial, Trivial)) - e_num(elt, Trivial, Trivial)
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    S_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    S⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the spin-plus operator `S⁺ = e†_↑ e_↓` (only compatible with `Trivial` spin symmetry).
"""
@operator S⁺ function S_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), I(1))][1, 2] = 1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    S_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    S⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the spin-minus operator `S⁻ = e†_↓ e_↑` (only compatible with `Trivial` spin symmetry).
"""
@operator S⁻ function S_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = copy(adjoint(S_plus(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    S_x([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    Sˣ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the one-body spin-1/2 x-operator on the electrons (only compatible with `Trivial` spin symmetry).
"""
@operator Sˣ function S_x(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = (S_plus(elt, Trivial, Trivial) + S_min(elt, Trivial, Trivial)) / 2
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    S_y([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    Sʸ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the one-body spin-1/2 y-operator on the electrons (only compatible with `Trivial` spin symmetry).
"""
@operator Sʸ function S_y(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    # explicit error to avoid infinite recursion:
    elt <: Real && throw(ArgumentError("S_y requires `elt <: Complex`"))
    t = (S_plus(elt, Trivial, Trivial) - S_min(elt, Trivial, Trivial)) / (2im)
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    S_z([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    Sᶻ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the one-body spin-1/2 z-operator on the electrons.
"""
@operator Sᶻ function S_z(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = (u_num(elt, Trivial, Trivial) - d_num(elt, Trivial, Trivial)) / 2
    return _maybe_slave_fermion(t, slave_fermion)
end

# Two site operators
# ------------------
"""
    u_plus_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    u⁺u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↑} e_{2,↑}`` that creates a spin-up electron at the
first site and annihilates a spin-up electron at the second. The only nonzero matrix element is
```
    +|↑,0⟩ ↤ |0,↑⟩
```
"""
@operator u⁺u⁻ function u_plus_u_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 1, 1] = 1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    d_plus_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    d⁺d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↓} e_{2,↓}`` that creates a spin-down electron at the
first site and annihilates a spin-down electron at the second. The only nonzero matrix element is
```
    +|↓,0⟩ ↤ |0,↓⟩
```
"""
@operator d⁺d⁻ function d_plus_d_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 1, 2] = 1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    u_min_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    u⁻u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↑} e†_{2,↑}`` that annihilates a spin-up electron at the
first site and creates a spin-up electron at the second.
"""
@operator u⁻u⁺ function u_min_u_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = -copy(adjoint(u_plus_u_min(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    d_min_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    d⁻d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↓} e†_{2,↓}`` that annihilates a spin-down electron at the
first site and creates a spin-down electron at the second.
"""
@operator d⁻d⁺ function d_min_d_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = -copy(adjoint(d_plus_d_min(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    e_plus_e_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    e⁺e⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator that creates an electron at the first site and annihilates an
electron at the second. This is the sum of `u_plus_u_min` and `d_plus_d_min`.
"""
@operator e⁺e⁻ function e_plus_e_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = u_plus_u_min(elt, Trivial, Trivial) + d_plus_d_min(elt, Trivial, Trivial)
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    e_min_e_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    e⁻e⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator that annihilates an electron at the first site and creates an
electron at the second. This is the sum of `u_min_u_plus` and `d_min_d_plus`.
"""
@operator e⁻e⁺ function e_min_e_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = -copy(adjoint(e_plus_e_min(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    e_hopping([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    e_hop([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator that describes an electron that hops between the first and the
second site.
"""
@operator e_hop function e_hopping(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = e_plus_e_min(elt, Trivial, Trivial) - e_min_e_plus(elt, Trivial, Trivial)
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    u_min_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    u⁻d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↑} e_{2,↓}`` that annihilates a spin-up electron at the
first site and a spin-down electron at the second site. The only nonzero matrix element is
```
    -|0,0⟩ ↤ |↑,↓⟩
```
This operator does not conserve the number of electrons, and is therefore only compatible
with `Trivial` particle symmetry.
"""
@operator u⁻d⁻ function u_min_d_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 1, 2] = -1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    u_plus_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    u⁺d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↑} e†_{2,↓}`` that creates a spin-up electron at the
first site and a spin-down electron at the second site.
"""
@operator u⁺d⁺ function u_plus_d_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = -copy(adjoint(u_min_d_min(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    d_min_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    d⁻u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↓} e_{2,↑}`` that annihilates a spin-down electron at the
first site and a spin-up electron at the second site. The only nonzero matrix element is
```
    -|0,0⟩ ↤ |↓,↑⟩
```
This operator does not conserve the number of electrons, and is therefore only compatible
with `Trivial` particle symmetry.
"""
@operator d⁻u⁻ function d_min_u_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 2, 1] = -1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    d_plus_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    d⁺u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↓} e†_{2,↑}`` that creates a spin-down electron at the
first site and a spin-up electron at the second site.
"""
@operator d⁺u⁺ function d_plus_u_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = -copy(adjoint(d_min_u_min(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    u_min_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    u⁻u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↑} e_{2,↑}`` that annihilates a spin-up electron at both
sites. The only nonzero matrix element is
```
    -|0,0⟩ ↤ |↑,↑⟩
```
This operator conserves neither the number of electrons nor ``S^z``, and is therefore only
compatible with `Trivial` particle and spin symmetry.
"""
@operator u⁻u⁻ function u_min_u_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 1, 1] = -1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    u_plus_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    u⁺u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↑} e†_{2,↑}`` that creates a spin-up electron at both sites.
"""
@operator u⁺u⁺ function u_plus_u_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = -copy(adjoint(u_min_u_min(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    d_min_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    d⁻d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↓} e_{2,↓}`` that annihilates a spin-down electron at both
sites. The only nonzero matrix element is
```
    -|0,0⟩ ↤ |↓,↓⟩
```
This operator conserves neither the number of electrons nor ``S^z``, and is therefore only
compatible with `Trivial` particle and spin symmetry.
"""
@operator d⁻d⁻ function d_min_d_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 2, 2] = -1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    d_plus_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    d⁺d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↓} e†_{2,↓}`` that creates a spin-down electron at both sites.
"""
@operator d⁺d⁺ function d_plus_d_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = -copy(adjoint(d_min_d_min(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    singlet_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    singlet⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body singlet operator ``(e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``,
which creates the singlet state when acting on vacuum.
"""
@operator singlet⁺ function singlet_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = (u_plus_d_plus(elt, Trivial, Trivial) - d_plus_u_plus(elt, Trivial, Trivial)) /
        sqrt(2)
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    singlet_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    singlet⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the adjoint of `singlet_plus` operator, which is
``(-e_{1,↑} e_{2,↓} + e_{1,↓} e_{2,↑}) / \\sqrt{2}``.
"""
@operator singlet⁻ function singlet_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = copy(adjoint(singlet_plus(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    singlet_plus_singlet_min_3site([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    Δ⁺ij_Δjk([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Returns the 3-site term ``O_{ijk} = Δ^†_{ij} Δ_{jk}``, where ``Δ^†_{ij} = (e^†_{i,↑} e^†_{j,↓} - e^†_{i,↓} e^†_{j,↑}) / \\sqrt{2}``.
It describes the hopping of a singlet pair from bond `(j,k)` to a nearest neighbor bond `(i,j)` sharing site `j`.
"""
@operator Δ⁺ij_Δjk function singlet_plus_singlet_min_3site(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    #=
                -5      -6
            ┌---┴-------┴---┐
            |     A_{jk}    |
            └---┬-------┬---┘
        -4      1       -3
    ┌---┴-------┴---┐
    |    A†_{ij}    |
    └---┬-------┬---┘
        -1      -2
        i       j       k
    =#
    singp = singlet_plus(elt, Trivial, Trivial)
    singm = singp'
    @tensor t[-1 -2 -3; -4 -5 -6] := singp[-1 -2; -4 1] * singm[1 -3; -5 -6]
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    singlet_plus_singlet_min_4site([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    Δ⁺ij_Δkl([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Returns the 4-site term ``O_{ijkl} = Δ^†_{ij} Δ_{kl}``, where ``Δ^†_{ij} = (e^†_{i,↑} e^†_{j,↓} - e^†_{i,↓} e^†_{j,↑}) / \\sqrt{2}``.
It measures the singlet pair correlation between two bonds `(i,j)` and `(k,l)`.
"""
@operator Δ⁺ij_Δkl function singlet_plus_singlet_min_4site(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    singp = singlet_plus(elt, Trivial, Trivial)
    return _maybe_slave_fermion(singp ⊗ singp', slave_fermion)
end

"""
    S_plus_S_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    S⁺S⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator S⁺S⁻.
The only nonzero matrix element corresponds to `|↑,↓⟩ <-- |↓,↑⟩`.
"""
@operator S⁺S⁻ function S_plus_S_min(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(1)), dual(I(1)))][1, 2, 2, 1] = 1
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    S_min_S_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    S⁻S⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator S⁻S⁺.
The only nonzero matrix element corresponds to `|↓,↑⟩ <-- |↑,↓⟩`.
"""
@operator S⁻S⁺ function S_min_S_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    t = copy(adjoint(S_plus_S_min(elt, Trivial, Trivial)))
    return _maybe_slave_fermion(t, slave_fermion)
end

"""
    S_exchange([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    SS([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the spin exchange operator S⋅S.
"""
@operator SS function S_exchange(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
        slave_fermion::Bool = false
    )
    Sz = S_z(elt, Trivial, Trivial)
    t = Sz ⊗ Sz +
        (S_plus_S_min(elt, Trivial, Trivial) + S_min_S_plus(elt, Trivial, Trivial)) / 2
    return _maybe_slave_fermion(t, slave_fermion)
end

end
