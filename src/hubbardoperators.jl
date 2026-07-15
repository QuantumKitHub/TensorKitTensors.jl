module HubbardOperators

using TensorKit
import ..TensorKitTensors: symmetrize, desymmetrize, @operator

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
export singlet_plus_singlet_min_3site
export singlet_plus_singlet_min_4site
export S_plus_S_min, S_min_S_plus, S_exchange

export n, nꜛ, nꜜ, nꜛꜜ, nʰ
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻
export u⁺u⁻, d⁺d⁻, u⁻u⁺, d⁻d⁺
export u⁻d⁻, d⁻u⁻, u⁺d⁺, d⁺u⁺
export u⁻u⁻, u⁺u⁺, d⁻d⁻, d⁺d⁺
export e⁺e⁻, e⁻e⁺, e_hop
export singlet⁺, singlet⁻, Δ⁺ij_Δjk, Δ⁺ij_Δkl
export S⁻S⁺, S⁺S⁻, SS

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
    the basis above. Operators that commute with the staggered gauge are unaffected by it;
    the remaining ones (e.g. [`e_hopping`](@ref)) are genuinely complex, require a complex
    scalar type, and are the gauge-transformed versions of their `Trivial` counterparts:
    they generate the same physics on bipartite lattices but are not elementwise equal to
    them.

    The ``η``-pairing SU(2) and the gauge transformation making it a symmetry of the Hubbard
    model are due to C. N. Yang and S. C. Zhang, *SO₄ symmetry in a Hubbard model*, Mod. Phys.
    Lett. B **4**, 759 (1990), [doi:10.1142/S0217984990000933](https://doi.org/10.1142/S0217984990000933);
    see also C. N. Yang, *η pairing and off-diagonal long-range order in a Hubbard model*,
    Phys. Rev. Lett. **63**, 2144 (1989), [doi:10.1103/PhysRevLett.63.2144](https://doi.org/10.1103/PhysRevLett.63.2144).

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
const _PARTICLE_GAUGE = Complex{Int}[1 0 0 0; 0 -1 0 0; 0 0 im 0; 0 0 0 im]

# Symmetrize a Hubbard operator, inserting the staggered gauge for `SU2Irrep` particle symmetry
# If the staggered gauge commutes we don't incorporate it to retain the option for real tensors.
function _symmetrize_hubbard(
        O::AbstractTensorMap, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    U = basis_transform(particle_symmetry, spin_symmetry)
    V = hubbard_space(particle_symmetry, spin_symmetry)
    particle_symmetry === SU2Irrep || return symmetrize(O, U, V)

    Vref = domain(U)[1]
    Gs = ntuple(k -> TensorMap(_PARTICLE_GAUGE^(k - 1), Vref ← Vref), numout(O))
    W = reduce(⊗, Gs)
    Od = desymmetrize(O)

    # check if commutes with the staggered gauge
    W * Od ≈ Od * W && return symmetrize(O, U, V)

    scalartype(O) <: Real &&
        throw(ArgumentError("operator with `SU2Irrep` particle symmetry that does not commute with the staggered gauge requires a complex scalar type"))
    return symmetrize(O, map(g -> U * g, Gs), V)
end

# Symmetrize a Hubbard operator through its (staggered-gauge) basis transformation
_symmetrize_operator(O::AbstractTensorMap, P::Type{<:Sector}, S::Type{<:Sector}) =
    _symmetrize_hubbard(O, P, S)

function n_site_operator(::Val{N}, elt::Type{<:Number}) where {N}
    V = hubbard_space(Trivial, Trivial)
    return zeros(elt, V^N ← V^N)
end

# Single-site operators
# ---------------------
"""
    u_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    nꜛ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of spin-up particles.
"""
@operator nꜛ function u_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), I(1))][1, 1] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end

"""
    d_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    nꜜ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of spin-down particles.
"""
@operator nꜜ function d_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), I(1))][2, 2] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end

"""
    e_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    n([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of particles.
"""
@operator n function e_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return u_num(elt, Trivial, Trivial) + d_num(elt, Trivial, Trivial)
end

"""
    ud_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    nꜛꜜ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of doubly occupied sites.
"""
@operator nꜛꜜ function ud_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return u_num(elt, Trivial, Trivial) * d_num(elt, Trivial, Trivial)
end

"""
    half_ud_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the the one-body operator that is equivalent to `(nꜛ - 1/2)(nꜜ - 1/2)`, which respects the particle-hole symmetry.
"""
@operator function half_ud_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    I = id(hubbard_space(Trivial, Trivial))
    return (u_num(elt, Trivial, Trivial) - I / 2) * (d_num(elt, Trivial, Trivial) - I / 2)
end

"""
    h_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    nʰ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of holes, i.e. the number of non-occupied sites.
"""
@operator nʰ function h_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return id(elt, hubbard_space(Trivial, Trivial)) - e_num(elt, Trivial, Trivial)
end

"""
    S_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    S⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the spin-plus operator `S⁺ = e†_↑ e_↓` (only compatible with `Trivial` spin symmetry).
"""
@operator S⁺ function S_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt)
    I = sectortype(t)
    t[(I(1), dual(I(1)))][1, 2] = 1.0
    return t
end

"""
    S_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    S⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the spin-minus operator `S⁻ = e†_↓ e_↑` (only compatible with `Trivial` spin symmetry).
"""
@operator S⁻ function S_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return copy(adjoint(S_plus(elt, Trivial, Trivial)))
end

"""
    S_x([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    Sˣ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the one-body spin-1/2 x-operator on the electrons (only compatible with `Trivial` spin symmetry).
"""
@operator Sˣ function S_x(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return (S_plus(elt, Trivial, Trivial) + S_min(elt, Trivial, Trivial)) / 2
end

"""
    S_y([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    Sʸ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the one-body spin-1/2 y-operator on the electrons (only compatible with `Trivial` spin symmetry).
"""
@operator Sʸ function S_y(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    # explicit error to avoid infinite recursion:
    elt <: Real && throw(ArgumentError("S_y requires `elt <: Complex`"))
    return (S_plus(elt, Trivial, Trivial) - S_min(elt, Trivial, Trivial)) / (2im)
end

"""
    S_z([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    Sᶻ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the one-body spin-1/2 z-operator on the electrons.
"""
@operator Sᶻ function S_z(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return (u_num(elt, Trivial, Trivial) - d_num(elt, Trivial, Trivial)) / 2
end

# Two site operators
# ------------------
"""
    u_plus_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    u⁺u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↑}, e_{2,↑}`` that creates a spin-up particle at the first site and annihilates a spin-up particle at the second.
"""
@operator u⁺u⁻ function u_plus_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 1, 1] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 1, 2] = 1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 2, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 2, 2, 2] = -1
    return t
end

"""
    d_plus_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    d⁺d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↓}, e_{2,↓}`` that creates a spin-down particle at the first site and annihilates a spin-down particle at the second.
"""
@operator d⁺d⁻ function d_plus_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 1, 2] = -1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 1, 1, 2] = -1
    return t
end

"""
    u_min_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    u⁻u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↑}, e†_{2,↑}`` that annihilates a spin-up particle at the first site and creates a spin-up particle at the second.
"""
@operator u⁻u⁺ function u_min_u_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(u_plus_u_min(elt, Trivial, Trivial)))
end

"""
    d_min_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    d⁻d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↓}, e†_{2,↓}`` that annihilates a spin-down particle at the first site and creates a spin-down particle at the second.
"""
@operator d⁻d⁺ function d_min_d_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(d_plus_d_min(elt, Trivial, Trivial)))
end

"""
    e_plus_e_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    e⁺e⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
This is the sum of `u_plus_u_min` and `d_plus_d_min`.
"""
@operator e⁺e⁻ function e_plus_e_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return u_plus_u_min(elt, Trivial, Trivial) + d_plus_d_min(elt, Trivial, Trivial)
end

"""
    e_min_e_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    e⁻e⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
This is the sum of `u_min_u_plus` and `d_min_d_plus`.
"""
@operator e⁻e⁺ function e_min_e_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(e_plus_e_min(elt, Trivial, Trivial)))
end

"""
    e_hopping([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    e_hop([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator that describes a particle that hops between the first and the second site.

For `SU2Irrep` particle symmetry, the hopping operator is expressed in the staggered gauge
``c_{j,σ} → i^j c_{j,σ}`` and requires a complex scalar type; see
[`basis_transform`](@ref HubbardOperators.basis_transform) for details.
"""
@operator e_hop function e_hopping(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return e_plus_e_min(elt, Trivial, Trivial) - e_min_e_plus(elt, Trivial, Trivial)
end

"""
    u_min_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    u⁻d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↑} e_{2,↓}`` that annihilates a spin-up particle at the first site and a spin-down particle at the second site.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↑,↓⟩,     +|0,↑⟩ ↤ |↑,↑↓⟩,
    +|↓,0⟩ ↤ |↑↓,↓⟩,    -|↓,↑⟩ ↤ |↑↓,↑↓⟩
```
"""
@operator u⁻d⁻ function u_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 1, 2] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 1, 1, 2] = 1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 2, 2] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 2, 2] = -1
    return t
end

"""
    u_plus_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    u⁺d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↑} e†_{2,↓}`` that creates a spin-up particle at the first site and a spin-down particle at the second site.
"""
@operator u⁺d⁺ function u_plus_d_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(u_min_d_min(elt, Trivial, Trivial)))
end

"""
    d_min_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    d⁻u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↓} e_{2,↑}`` that annihilates a spin-down particle at the first site and a spin-up particle at the second site.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↓,↑⟩,     -|0,↓⟩ ↤ |↓,↑↓⟩
    -|↑,0⟩ ↤ |↑↓,↑⟩,    -|↑,↓⟩ ↤ |↑↓,↑↓⟩
```
"""
@operator d⁻u⁻ function d_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 2, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 2, 2, 2] = -1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 2, 1] = -1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 2, 2] = -1
    return t
end

"""
    d_plus_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    d⁺u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↓} e†_{2,↑}`` that creates a spin-down particle at the first site and a spin-up particle at the second site.
"""
@operator d⁺u⁺ function d_plus_u_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(d_min_u_min(elt, Trivial, Trivial)))
end

"""
    u_min_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    u⁻u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↑} e_{2,↑}`` that annihilates a spin-up particle at both sites.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↑,↑⟩,     -|0,↓⟩ ↤ |↑,↑↓⟩
    +|↓,0⟩ ↤ |↑↓,↑⟩,    +|↓,↓⟩ ↤ |↑↓,↑↓⟩
```
"""
@operator u⁻u⁻ function u_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 1, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 2, 1, 2] = -1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 2, 1] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 2, 2, 2] = 1
    return t
end

"""
    u_plus_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    u⁺u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↑} e†_{2,↑}`` that creates a spin-up particle at both sites.
"""
@operator u⁺u⁺ function u_plus_u_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(u_min_u_min(elt, Trivial, Trivial)))
end

"""
    d_min_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    d⁻d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↓} e_{2,↓}`` that annihilates a spin-down particle at both sites.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↓,↓⟩,     +|0,↑⟩ ↤ |↓,↑↓⟩
    -|↑,0⟩ ↤ |↑↓,↓⟩,    +|↑,↑⟩ ↤ |↑↓,↑↓⟩
```
"""
@operator d⁻d⁻ function d_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 2, 2] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 1, 2, 2] = 1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 2, 2] = -1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 1, 2, 2] = 1
    return t
end

"""
    d_plus_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    d⁺d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↓} e†_{2,↓}`` that creates a spin-down particle at both sites.
"""
@operator d⁺d⁺ function d_plus_d_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return -copy(adjoint(d_min_d_min(elt, Trivial, Trivial)))
end

"""
    singlet_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    singlet⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body singlet operator ``(e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``,
which creates the singlet state when acting on vaccum.
"""
@operator singlet⁺ function singlet_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return (u_plus_d_plus(elt, Trivial, Trivial) - d_plus_u_plus(elt, Trivial, Trivial)) /
        sqrt(2)
end

"""
    singlet_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    singlet⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the adjoint of `singlet_plus` operator, which is
``(-e_{1,↑} e_{2,↓} + e_{1,↓} e_{2,↑}) / \\sqrt{2}``.
"""
@operator singlet⁻ function singlet_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return copy(adjoint(singlet_plus(elt, Trivial, Trivial)))
end

"""
    singlet_plus_singlet_min_3site([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Returns the 3-site term ``O_{ijk} = Δ^†_{ij} Δ_{jk}``, where
``Δ^†_{ij} = (e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``.
It describes the hopping of a singlet pair from bond `(j,k)`
to a nearest neighbor bond `(i,j)` sharing site `j`.
"""
@operator Δ⁺ij_Δjk function singlet_plus_singlet_min_3site(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
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
    return @tensor t[-1 -2 -3; -4 -5 -6] := singp[-1 -2; -4 1] * singm[1 -3; -5 -6]
end


"""
    singlet_plus_singlet_min_4site([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Returns the 4-site term ``O_{ijkl} = Δ^†_{ij} Δ_{kl}``, where
``Δ^†_{ij} = (e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``.
It measures the singlet pair correlation between two bonds `(i,j)` and `(k,l)`.
"""
@operator Δ⁺ij_Δkl function singlet_plus_singlet_min_4site(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    singp = singlet_plus(elt, Trivial, Trivial)
    return singp ⊗ singp'
end

"""
    S_plus_S_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    S⁺S⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator S⁺S⁻.
The only nonzero matrix element corresponds to `|↑,↓⟩ <-- |↓,↑⟩`.
"""
@operator S⁺S⁻ function S_plus_S_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(1)), dual(I(1)))][1, 2, 2, 1] = 1
    return t
end

"""
    S_min_S_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])
    S⁻S⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the two-body operator S⁻S⁺.
The only nonzero matrix element corresponds to `|↓,↑⟩ <-- |↑,↓⟩`.
"""
@operator S⁻S⁺ function S_min_S_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    return copy(adjoint(S_plus_S_min(elt, Trivial, Trivial)))
end

"""
    S_exchange([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}])

Return the spin exchange operator S⋅S.
"""
@operator SS function S_exchange(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    Sz = S_z(elt, Trivial, Trivial)
    return Sz ⊗ Sz +
        (S_plus_S_min(elt, Trivial, Trivial) + S_min_S_plus(elt, Trivial, Trivial)) / 2
end

end
