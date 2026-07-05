module HubbardOperators

using TensorKit
using LinearAlgebra: I
import ..TensorKitTensors: symmetrize, desymmetrize, _restrict_scalartype

export hubbard_space, basis_transform
export e_num, u_num, d_num, ud_num, half_ud_num, h_num
export S_x, S_y, S_z, S_plus, S_min
export u_plus_u_min, d_plus_d_min
export u_min_u_plus, d_min_d_plus
export u_min_d_min, d_min_u_min
export u_plus_d_plus, d_plus_u_plus
export u_min_u_min, d_min_d_min
export u_plus_u_plus, d_plus_d_plus
export e_plus_e_min, e_min_e_plus, e_hopping
export singlet_plus, singlet_min
export S_plus_S_min, S_min_S_plus, S_exchange

export n, nꜛ, nꜜ, nꜛꜜ, nʰ
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻
export u⁺u⁻, d⁺d⁻, u⁻u⁺, d⁻d⁺
export u⁻d⁻, d⁻u⁻, u⁺d⁺, d⁺u⁺
export u⁻u⁻, u⁺u⁺, d⁻d⁻, d⁺d⁺
export e⁺e⁻, e⁻e⁺, e_hop
export singlet⁺, singlet⁻
export S⁻S⁺, S⁺S⁻

"""
    hubbard_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the local hilbert space for a Hubbard-type model with the given particle and spin symmetries. The four basis states are
```
    |0⟩ (vacuum), |↑⟩ = (c↑)†|0⟩, |↓⟩ = (c↓)†|0⟩, |↑↓⟩ = (c↑)†(c↓)†|0⟩.
```
The possible symmetries are `Trivial`, `U1Irrep`, and `SU2Irrep`, for both particle number and spin.
"""
function hubbard_space((::Type{Trivial}) = Trivial, (::Type{Trivial}) = Trivial)
    return Vect[FermionParity](0 => 2, 1 => 2)
end
function hubbard_space(::Type{Trivial}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
end
function hubbard_space(::Type{Trivial}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 2, (1, 1 // 2) => 1)
end
function hubbard_space(::Type{U1Irrep}, ::Type{Trivial})
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1)
end
function hubbard_space(::Type{U1Irrep}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep](
        (0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1, (0, 2, 0) => 1
    )
end
function hubbard_space(::Type{U1Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep](
        (0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (0, 2, 0) => 1
    )
end
function hubbard_space(::Type{SU2Irrep}, ::Type{Trivial})
    return Vect[FermionParity ⊠ SU2Irrep]((0, 1 // 2) => 1, (1, 0) => 2)
end
function hubbard_space(::Type{SU2Irrep}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ SU2Irrep ⊠ U1Irrep](
        (0, 1 // 2, 0) => 1, (1, 0, -1 // 2) => 1, (1, 0, 1 // 2) => 1
    )
end
function hubbard_space(::Type{SU2Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ SU2Irrep ⊠ SU2Irrep](
        (0, 1 // 2, 0) => 1, (1, 0, 1 // 2) => 1
    )
end
function hubbard_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    throw(ArgumentError("invalid symmetry `($particle_symmetry, $spin_symmetry)`"))
end

"""
    basis_transform(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the unitary basis transformation that maps the basis ``\\{|0⟩, |↑↓⟩, |↑⟩, |↓⟩\\}`` of
`hubbard_space(Trivial, Trivial)` (fermion-parity even states first) onto the basis of
`hubbard_space(particle_symmetry, spin_symmetry)`, as a `TensorMap` between the
desymmetrized versions of these spaces (see
[`desymmetrize`](@ref TensorKitTensors.desymmetrize)), as required by
[`symmetrize`](@ref TensorKitTensors.symmetrize).

For all symmetry combinations the transformation is a permutation, determined by the sector
order of the target space, where the states are identified as follows:

- For `U1Irrep` particle symmetry, the particle number is used as charge, distinguishing
  ``|0⟩`` (charge 0) from ``|↑↓⟩`` (charge 2).
- For `SU2Irrep` particle symmetry, ``(|↑↓⟩, |0⟩)`` forms the ``η``-pairing doublet, ordered
  by descending ``η^z = (n - 1)/2``.
- For `U1Irrep` spin symmetry, the ``S^z`` eigenvalue ``±1/2`` is used as charge,
  distinguishing ``|↑⟩`` from ``|↓⟩``.
- For `SU2Irrep` spin symmetry, ``(|↑⟩, |↓⟩)`` forms the spin doublet (descending ``m``).

!!! note "Staggered gauge for SU2Irrep particle symmetry"
    The ``η``-pairing SU(2) symmetry only commutes with the Hubbard model after a staggered
    gauge transformation ``c_{j,σ} → i^j c_{j,σ}`` on a bipartite lattice. Accordingly, for
    `SU2Irrep` particle symmetry the operators of this module act on site ``k`` with the
    additional gauge factor ``G^{k-1}``, where ``G = i^n = \\mathrm{diag}(1, -1, i, i)`` in
    the basis above. The resulting operators (e.g. [`e_hopping`](@ref)) are the
    gauge-transformed versions of their `Trivial` counterparts: they generate the same
    physics on bipartite lattices but are not elementwise equal to them.

The transformations have exact integer entries and are therefore returned with integer
scalar type, such that they promote to any scalar type without loss of precision.
"""
function basis_transform(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    V = hubbard_space(particle_symmetry, spin_symmetry)
    U = zeros(Int, 4, 4)
    row = 1
    for c in sectors(V)
        for j in _state_indices(c, particle_symmetry, spin_symmetry)
            U[row, j] = 1
            row += 1
        end
    end
    return TensorMap(U, desymmetrize(V) ← desymmetrize(hubbard_space(Trivial, Trivial)))
end

# trivial-basis indices (|0⟩, |↑↓⟩, |↑⟩, |↓⟩) = (1, 2, 3, 4) contained in sector `c`,
# in dense row order
function _state_indices(c, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    parity = c isa FermionParity ? c : c[1]
    return if !parity.isodd # parity even: |0⟩ and/or |↑↓⟩
        if particle_symmetry === Trivial
            return (1, 2)
        elseif particle_symmetry === SU2Irrep
            return (2, 1) # η-doublet, descending ηᶻ
        else # U1Irrep: particle number distinguishes the states
            return c[2].charge == 0 ? (1,) : (2,)
        end
    else # parity odd: |↑⟩ and/or |↓⟩
        if spin_symmetry === Trivial || spin_symmetry === SU2Irrep
            return (3, 4)
        else # U1Irrep: Sᶻ distinguishes the states (spin is the last sector factor)
            spin_sector = particle_symmetry === Trivial ? c[2] : c[3]
            return spin_sector.charge > 0 ? (3,) : (4,)
        end
    end
end

# staggered gauge i^n for the η-pairing SU(2) symmetry, in the trivial basis order;
# exact (Gaussian) integer entries promote to any scalar type without loss of precision
function _particle_gauge(::Type{SU2Irrep})
    return Complex{Int}[1 0 0 0; 0 -1 0 0; 0 0 im 0; 0 0 0 im]
end
_particle_gauge(::Type{<:Sector}) = Matrix{Int}(I, 4, 4)

function n_site_operator(::Val{N}, elt::Type{<:Number}) where {N}
    V = hubbard_space(Trivial, Trivial)
    return zeros(elt, V^N ← V^N)
end

# Single-site operators
# ---------------------
@doc """
    u_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜛ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of spin-up particles.
""" u_num
function u_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), I(1))][1, 1] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end
const nꜛ = u_num

@doc """
    d_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜜ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of spin-down particles.
""" d_num
function d_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), I(1))][2, 2] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end
const nꜜ = d_num

@doc """
    e_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    n([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of particles.
""" e_num
function e_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return u_num(elt, Trivial, Trivial) + d_num(elt, Trivial, Trivial)
end
const n = e_num

@doc """
    ud_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜛꜜ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of doubly occupied sites.
""" ud_num
function ud_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return u_num(elt, Trivial, Trivial) * d_num(elt, Trivial, Trivial)
end
const nꜛꜜ = ud_num

@doc """
    half_ud_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the the one-body operator that is equivalent to `(nꜛ - 1/2)(nꜜ - 1/2)`, which respects the particle-hole symmetry.
""" half_ud_num
function half_ud_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    I = id(hubbard_space(Trivial, Trivial))
    return (u_num(elt, Trivial, Trivial) - I / 2) * (d_num(elt, Trivial, Trivial) - I / 2)
end

@doc """
    h_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nʰ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of holes, i.e. the number of non-occupied sites.
""" h_num
function h_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return id(elt, hubbard_space(Trivial, Trivial)) - e_num(elt, Trivial, Trivial)
end
const nʰ = h_num

@doc """
    S_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    S⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the spin-plus operator `S⁺ = e†_↑ e_↓` (only compatible with `Trivial` spin symmetry).
""" S_plus
function S_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), dual(I(1)))][1, 2] = 1.0
    return t
end
const S⁺ = S_plus

@doc """
    S_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    S⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the spin-minus operator `S⁻ = e†_↓ e_↑` (only compatible with `Trivial` spin symmetry).
""" S_min
function S_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return copy(adjoint(S_plus(elt, Trivial, Trivial)))
end
const S⁻ = S_min

@doc """
    S_x([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    Sˣ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body spin-1/2 x-operator on the electrons (only compatible with `Trivial` spin symmetry).
""" S_x
function S_x(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return (S_plus(elt, Trivial, Trivial) + S_min(elt, Trivial, Trivial)) / 2
end
const Sˣ = S_x

@doc """
    S_y([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    Sʸ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body spin-1/2 y-operator on the electrons (only compatible with `Trivial` spin symmetry).
""" S_y
function S_y(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return (S_plus(elt, Trivial, Trivial) - S_min(elt, Trivial, Trivial)) / (2im)
end
const Sʸ = S_y

@doc """
    S_z([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    Sᶻ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body spin-1/2 z-operator on the electrons.
""" S_z
function S_z(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return (u_num(elt, Trivial, Trivial) - d_num(elt, Trivial, Trivial)) / 2
end
const Sᶻ = S_z

# Two site operators
# ------------------
@doc """
    u_plus_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁺u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↑}, e_{2,↑}`` that creates a spin-up particle at the first site and annihilates a spin-up particle at the second.
""" u_plus_u_min
function u_plus_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 1, 1] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 1, 2] = 1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 2, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 2, 2, 2] = -1
    return t
end
const u⁺u⁻ = u_plus_u_min

@doc """
    d_plus_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁺d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↓}, e_{2,↓}`` that creates a spin-down particle at the first site and annihilates a spin-down particle at the second.
""" d_plus_d_min
function d_plus_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 1, 2] = -1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 1, 1, 2] = -1
    return t
end
const d⁺d⁻ = d_plus_d_min

@doc """
    u_min_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁻u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↑}, e†_{2,↑}`` that annihilates a spin-up particle at the first site and creates a spin-up particle at the second.
""" u_min_u_plus
function u_min_u_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(u_plus_u_min(elt, Trivial, Trivial)))
end
const u⁻u⁺ = u_min_u_plus

@doc """
    d_min_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁻d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↓}, e†_{2,↓}`` that annihilates a spin-down particle at the first site and creates a spin-down particle at the second.
""" d_min_d_plus
function d_min_d_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(d_plus_d_min(elt, Trivial, Trivial)))
end
const d⁻d⁺ = d_min_d_plus

@doc """
    e_plus_e_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e⁺e⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
This is the sum of `u_plus_u_min` and `d_plus_d_min`.
""" e_plus_e_min
function e_plus_e_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return u_plus_u_min(elt, Trivial, Trivial) + d_plus_d_min(elt, Trivial, Trivial)
end
const e⁺e⁻ = e_plus_e_min

@doc """
    e_min_e_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e⁻e⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
This is the sum of `u_min_u_plus` and `d_min_d_plus`.
""" e_min_e_plus
function e_min_e_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(e_plus_e_min(elt, Trivial, Trivial)))
end
const e⁻e⁺ = e_min_e_plus

@doc """
    e_hopping([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e_hop([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that describes a particle that hops between the first and the second site.

For `SU2Irrep` particle symmetry, the hopping operator is expressed in the staggered gauge
``c_{j,σ} → i^j c_{j,σ}`` and requires a complex scalar type; see
[`basis_transform`](@ref HubbardOperators.basis_transform) for details.
""" e_hopping
function e_hopping(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return e_plus_e_min(elt, Trivial, Trivial) - e_min_e_plus(elt, Trivial, Trivial)
end
const e_hop = e_hopping

@doc """
    u_min_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁻d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↑} e_{2,↓}`` that annihilates a spin-up particle at the first site and a spin-down particle at the second site.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↑,↓⟩,     +|0,↑⟩ ↤ |↑,↑↓⟩,
    +|↓,0⟩ ↤ |↑↓,↓⟩,    -|↓,↑⟩ ↤ |↑↓,↑↓⟩
```
""" u_min_d_min
function u_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 1, 2] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 1, 1, 2] = 1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 2, 2] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 2, 2] = -1
    return t
end
const u⁻d⁻ = u_min_d_min

@doc """
    u_plus_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁺d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↑} e†_{2,↓}`` that creates a spin-up particle at the first site and a spin-down particle at the second site.
""" u_plus_d_plus
function u_plus_d_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(u_min_d_min(elt, Trivial, Trivial)))
end
const u⁺d⁺ = u_plus_d_plus

@doc """
    d_min_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁻u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↓} e_{2,↑}`` that annihilates a spin-down particle at the first site and a spin-up particle at the second site.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↓,↑⟩,     -|0,↓⟩ ↤ |↓,↑↓⟩
    -|↑,0⟩ ↤ |↑↓,↑⟩,    -|↑,↓⟩ ↤ |↑↓,↑↓⟩
```
""" d_min_u_min
function d_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 2, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 2, 2, 2] = -1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 2, 1] = -1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 2, 2] = -1
    return t
end
const d⁻u⁻ = d_min_u_min

@doc """
    d_plus_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁺u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↓} e†_{2,↑}`` that creates a spin-down particle at the first site and a spin-up particle at the second site.
""" d_plus_u_plus
function d_plus_u_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(d_min_u_min(elt, Trivial, Trivial)))
end
const d⁺u⁺ = d_plus_u_plus

@doc """
    u_min_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁻u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↑} e_{2,↑}`` that annihilates a spin-up particle at both sites.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↑,↑⟩,     -|0,↓⟩ ↤ |↑,↑↓⟩
    +|↓,0⟩ ↤ |↑↓,↑⟩,    +|↓,↓⟩ ↤ |↑↓,↑↓⟩
```
""" u_min_u_min
function u_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 1, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 2, 1, 2] = -1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 2, 1] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 2, 2, 2] = 1
    return t
end
const u⁻u⁻ = u_min_u_min

@doc """
    u_plus_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁺u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↑} e†_{2,↑}`` that creates a spin-up particle at both sites.
""" u_plus_u_plus
function u_plus_u_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(u_min_u_min(elt, Trivial, Trivial)))
end
const u⁺u⁺ = u_plus_u_plus

@doc """
    d_min_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁻d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↓} e_{2,↓}`` that annihilates a spin-down particle at both sites.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↓,↓⟩,     +|0,↑⟩ ↤ |↓,↑↓⟩
    -|↑,0⟩ ↤ |↑↓,↓⟩,    +|↑,↑⟩ ↤ |↑↓,↑↓⟩
```
""" d_min_d_min
function d_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 2, 2] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 1, 2, 2] = 1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 2, 2] = -1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 1, 2, 2] = 1
    return t
end
const d⁻d⁻ = d_min_d_min

@doc """
    d_plus_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁺d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↓} e†_{2,↓}`` that creates a spin-down particle at both sites.
""" d_plus_d_plus
function d_plus_d_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(d_min_d_min(elt, Trivial, Trivial)))
end
const d⁺d⁺ = d_plus_d_plus

@doc """
    singlet_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    singlet⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body singlet operator ``(e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``,
which creates the singlet state when acting on vaccum.
""" singlet_plus
function singlet_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return (u_plus_d_plus(elt, Trivial, Trivial) - d_plus_u_plus(elt, Trivial, Trivial)) /
        sqrt(2)
end
const singlet⁺ = singlet_plus

@doc """
    singlet_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    singlet⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the adjoint of `singlet_plus` operator, which is
``(-e_{1,↑} e_{2,↓} + e_{1,↓} e_{2,↑}) / \\sqrt{2}``.
""" singlet_min
function singlet_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return copy(adjoint(singlet_plus(elt, Trivial, Trivial)))
end
const singlet⁻ = singlet_min

@doc """
    S_plus_S_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    S⁺S⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator S⁺S⁻.
The only nonzero matrix element corresponds to `|↑,↓⟩ <-- |↓,↑⟩`.
""" S_plus_S_min
function S_plus_S_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(1)), dual(I(1)))][1, 2, 2, 1] = 1
    return t
end
const S⁺S⁻ = S_plus_S_min

@doc """
    S_min_S_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    S⁻S⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator S⁻S⁺.
The only nonzero matrix element corresponds to `|↓,↑⟩ <-- |↑,↓⟩`.
""" S_min_S_plus
function S_min_S_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return copy(adjoint(S_plus_S_min(elt, Trivial, Trivial)))
end
const S⁻S⁺ = S_min_S_plus

@doc """
    S_exchange([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the spin exchange operator S⋅S.
""" S_exchange
function S_exchange(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    Sz = S_z(elt, Trivial, Trivial)
    return Sz ⊗ Sz +
        (S_plus_S_min(elt, Trivial, Trivial) + S_min_S_plus(elt, Trivial, Trivial)) / 2
end

# Symmetric operators and default arguments
# -----------------------------------------
# The symmetric operators are automatically generated from their `(Trivial, Trivial)`
# counterparts through `symmetrize` and `basis_transform` (including the staggered gauge for
# `SU2Irrep` particle symmetry). Operators that are incompatible with a given symmetry
# combination throw an `ArgumentError`.
for opname in (
        :e_num, :u_num, :d_num, :ud_num, :half_ud_num, :h_num,
        :S_x, :S_y, :S_z, :S_plus, :S_min,
        :u_plus_u_min, :d_plus_d_min, :u_min_u_plus, :d_min_d_plus,
        :u_min_d_min, :d_min_u_min, :u_plus_d_plus, :d_plus_u_plus,
        :u_min_u_min, :d_min_d_min, :u_plus_u_plus, :d_plus_d_plus,
        :e_plus_e_min, :e_min_e_plus, :e_hopping,
        :singlet_plus, :singlet_min,
        :S_plus_S_min, :S_min_S_plus, :S_exchange,
    )
    @eval begin
        function $opname(P::Type{<:Sector} = Trivial, S::Type{<:Sector} = Trivial)
            return $opname(ComplexF64, P, S)
        end
        $opname(elt::Type{<:Number}) = $opname(elt, Trivial, Trivial)
        function $opname(
                elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                spin_symmetry::Type{<:Sector}
            )
            O = $opname(elt, Trivial, Trivial)
            U = basis_transform(particle_symmetry, spin_symmetry)
            G = _particle_gauge(particle_symmetry)
            Vref = domain(U)[1]
            Us = ntuple(k -> U * TensorMap(G^(k - 1), Vref ← Vref), numout(O))
            V = hubbard_space(particle_symmetry, spin_symmetry)
            O′ = symmetrize(O, Us, V)
            return _restrict_scalartype(elt, O′)
        end
    end
end

end
