module HubbardOperators

using TensorKit

export hubbard_space
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

function n_site_operator(
        ::Val{N}, elt::Type{<:Number},
        particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    ) where {N}
    V = hubbard_space(particle_symmetry, spin_symmetry)
    return zeros(elt, V^N ← V^N)
end

# helper functions to convert Trivial ⊠ SU2Irrep operators
# to U1Irrep ⊠ SU2Irrep if it has U(1) particle symmetry
function _promote_basis_particle_u1(
        sect::ProductSector{Tuple{FermionParity, SU2Irrep}}, idx::Integer
    )
    p, j = sect[1].isodd, sect[2].j
    n = if (!p && j == 0 && idx == 1)
        0
    elseif (!p && j == 0 && idx == 2)
        2
    elseif (p && j == 1 // 2 && idx == 1)
        1
    else
        error("Invalid Hubbard state.")
    end
    return FermionParity(p) ⊠ U1Irrep(n) ⊠ SU2Irrep(j)
end
function _promote_particle_u1(
        op1::AbstractTensorMap{E, S1, N, N}
    ) where {E, S1 <: GradedSpace{ProductSector{Tuple{FermionParity, SU2Irrep}}}, N}
    S = FermionParity ⊠ U1Irrep ⊠ SU2Irrep
    op2 = n_site_operator(Val(N), E, U1Irrep, SU2Irrep)
    for (f1, f2) in fusiontrees(op1)
        blk = op1[f1, f2]
        nonzero_idxs = findall(!iszero, blk)
        for idxs in nonzero_idxs
            # output fusion (split) tree
            uncoupled1 = map(f1.uncoupled, idxs.I[1:N]) do sect, idx
                return _promote_basis_particle_u1(sect, idx)
            end
            coupled = S(
                f1.coupled[1].isodd,
                sum(sect[2].charge for sect in uncoupled1),
                f1.coupled[2].j
            )
            innerlines1 = tuple(
                map(enumerate(f1.innerlines)) do (i, sect)
                    ninner = sum(c[2].charge for c in uncoupled1[1:(i + 1)])
                    return S(sect[1].isodd, ninner, sect[2].j)
                end...
            )
            f1′ = FusionTree(uncoupled1, coupled, ntuple(Returns(false), N), innerlines1)
            # input fusion tree
            uncoupled2 = map(f2.uncoupled, idxs.I[(N + 1):end]) do sect, idx
                return _promote_basis_particle_u1(sect, idx)
            end
            innerlines2 = tuple(
                map(enumerate(f2.innerlines)) do (i, sect)
                    ninner = sum(c[2].charge for c in uncoupled2[1:(i + 1)])
                    return S(sect[1].isodd, ninner, sect[2].j)
                end...
            )
            f2′ = FusionTree(uncoupled2, coupled, ntuple(Returns(false), N), innerlines2)
            # tensor element
            @assert length(op2[f1′, f2′]) == 1
            op2[f1′, f2′] .= blk[idxs]
        end
    end
    return op2
end

# Single-site operators
# ---------------------
@doc """
    u_num([particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜛ([particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of spin-up particles.
""" u_num
u_num(P::Type{<:Sector}, S::Type{<:Sector}) = u_num(ComplexF64, P, S)
function u_num(elt::Type{<:Number}, (::Type{Trivial}) = Trivial, (::Type{Trivial}) = Trivial)
    t = n_site_operator(Val(1), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(1))][1, 1] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end
function u_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep})
    t = n_site_operator(Val(1), elt, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1 // 2), dual(I(1, 1 // 2)))][1, 1] = 1
    t[(I(0, 0), dual(I(0, 0)))][2, 2] = 1
    return t
end
function u_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt, U1Irrep, Trivial)
    I = sectortype(t)
    block(t, I(1, 1))[1, 1] = 1
    block(t, I(0, 2))[1, 1] = 1
    return t
end
function u_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = n_site_operator(Val(1), elt, U1Irrep, U1Irrep)
    I = sectortype(t)
    block(t, I(1, 1, 1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 1
    return t
end
function u_num(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function u_num(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function u_num(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜛ = u_num

@doc """
    d_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜜ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of spin-down particles.
""" d_num
d_num(P::Type{<:Sector}, S::Type{<:Sector}) = d_num(ComplexF64, P, S)
function d_num(elt::Type{<:Number}, (::Type{Trivial}) = Trivial, (::Type{Trivial}) = Trivial)
    t = n_site_operator(Val(1), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(1))][2, 2] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end
function d_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep})
    t = n_site_operator(Val(1), elt, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, -1 // 2), dual(I(1, -1 // 2)))][1, 1] = 1
    t[(I(0, 0), I(0, 0))][2, 2] = 1
    return t
end
function d_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt, U1Irrep, Trivial)
    I = sectortype(t)
    block(t, I(1, 1))[2, 2] = 1 # expected to be [1,2]
    block(t, I(0, 2))[1, 1] = 1
    return t
end
function d_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = n_site_operator(Val(1), elt, U1Irrep, U1Irrep)
    I = sectortype(t)
    block(t, I(1, 1, -1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 1
    return t
end
function d_num(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function d_num(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function d_num(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜜ = d_num

@doc """
    e_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    n([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of particles.
""" e_num
e_num(P::Type{<:Sector}, S::Type{<:Sector}) = e_num(ComplexF64, P, S)
function e_num(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return u_num(elt, particle_symmetry, spin_symmetry) +
        d_num(elt, particle_symmetry, spin_symmetry)
end
function e_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep})
    t = n_site_operator(Val(1), elt, Trivial, SU2Irrep)
    I = sectortype(t)
    block(t, I(1, 1 // 2))[1, 1] = 1
    block(t, I(0, 0))[2, 2] = 2
    return t
end
function e_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = n_site_operator(Val(1), elt, U1Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I(1, 1, 1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 2
    return t
end
const n = e_num

@doc """
    ud_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜛꜜ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of doubly occupied sites.
""" ud_num
ud_num(P::Type{<:Sector}, S::Type{<:Sector}) = ud_num(ComplexF64, P, S)
function ud_num(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return u_num(elt, particle_symmetry, spin_symmetry) *
        d_num(elt, particle_symmetry, spin_symmetry)
end
function ud_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep})
    t = n_site_operator(Val(1), elt, Trivial, SU2Irrep)
    I = sectortype(t)
    block(t, I(0, 0))[2, 2] = 1
    return t
end
function ud_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = n_site_operator(Val(1), elt, U1Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I(0, 2, 0)) .= 1
    return t
end
const nꜛꜜ = ud_num

@doc """
    half_ud_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the the one-body operator that is equivalent to `(nꜛ - 1/2)(nꜜ - 1/2)`, which respects the particle-hole symmetry.
"""
half_ud_num(P::Type{<:Sector}, S::Type{<:Sector}) = half_ud_num(ComplexF64, P, S)
function half_ud_num(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    I = id(hubbard_space(particle_symmetry, spin_symmetry))
    return (u_num(elt, particle_symmetry, spin_symmetry) - I / 2) *
        (d_num(elt, particle_symmetry, spin_symmetry) - I / 2)
end
function half_ud_num(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    t = n_site_operator(Val(1), elt, SU2Irrep, SU2Irrep)
    block(t, sectortype(t)(0, 1 // 2, 0)) .= 1 // 4
    block(t, sectortype(t)(1, 0, 1 // 2)) .= -1 // 4
    return t
end

@doc """
    h_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nʰ([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of holes, i.e. the number of non-occupied sites.
""" h_num
h_num(P::Type{<:Sector}, S::Type{<:Sector}) = h_num(ComplexF64, P, S)
h_num(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}) =
    id(elt, hubbard_space(particle_symmetry, spin_symmetry)) - e_num(elt, particle_symmetry, spin_symmetry)
const nʰ = h_num

@doc """
    S_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    S⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the spin-plus operator `S⁺ = e†_↑ e_↓` (only defined for `Trivial` spin symmetry).
""" S_plus
function S_plus(P::Type{<:Sector}, S::Type{<:Sector})
    return S_plus(ComplexF64, P, S)
end
function S_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), dual(I(1)))][1, 2] = 1.0
    return t
end
function S_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial})
    t = n_site_operator(Val(1), elt, U1Irrep, Trivial)
    I = sectortype(t)
    t[(I(1, 1), dual(I(1, 1)))][1, 2] = 1.0
    return t
end
function S_plus(::Type{<:Number}, ::Type{<:Sector}, ::Type{U1Irrep})
    throw(ArgumentError("`S_plus`, `S_min` are not symmetric under `U1Irrep` spin symmetry"))
end
function S_plus(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`S_plus`, `S_min` are not symmetric under `SU2Irrep` spin symmetry"))
end
const S⁺ = S_plus

@doc """
    S_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    S⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the spin-minus operator (only defined for `Trivial` spin symmetry).
""" S_min
function S_min(P::Type{<:Sector}, S::Type{<:Sector})
    return S_min(ComplexF64, P, S)
end
function S_min(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return copy(adjoint(S_plus(elt, particle_symmetry, spin_symmetry)))
end
const S⁻ = S_min

@doc """
    S_x(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    Sˣ(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the one-body spin-1/2 x-operator on the electrons (only defined for `Trivial` symmetry).
""" S_x
function S_x(P::Type{<:Sector} = Trivial, S::Type{<:Sector} = Trivial)
    return S_x(ComplexF64, P, S)
end
function S_x(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return (
        S_plus(elt, particle_symmetry, spin_symmetry) +
            S_min(elt, particle_symmetry, spin_symmetry)
    ) / 2
end
const Sˣ = S_x

@doc """
    S_y(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    Sʸ(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the one-body spin-1/2 y-operator on the electrons (only defined for `Trivial` symmetry).
""" S_y
function S_y(P::Type{<:Sector} = Trivial, S::Type{<:Sector} = Trivial)
    return S_y(ComplexF64, P, S)
end
function S_y(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return (
        S_plus(elt, particle_symmetry, spin_symmetry) -
            S_min(elt, particle_symmetry, spin_symmetry)
    ) / (2im)
end
const Sʸ = S_y

@doc """
    S_z(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    Sᶻ(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the one-body spin-1/2 z-operator on the electrons.
""" S_z
function S_z(P::Type{<:Sector} = Trivial, S::Type{<:Sector} = Trivial)
    return S_z(ComplexF64, P, S)
end
function S_z(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return (
        u_num(elt, particle_symmetry, spin_symmetry) -
            d_num(elt, particle_symmetry, spin_symmetry)
    ) / 2
end
const Sᶻ = S_z

# Two site operators
# ------------------
@doc """
    u_plus_u_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁺u⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↑}, e_{2,↑}`` that creates a spin-up particle at the first site and annihilates a spin-up particle at the second.
""" u_plus_u_min
u_plus_u_min(P::Type{<:Sector}, S::Type{<:Sector}) = u_plus_u_min(ComplexF64, P, S)
function u_plus_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 1, 1] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 1, 2] = 1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 2, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 2, 2, 2] = -1
    return t
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep})
    t = n_site_operator(Val(2), elt, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1 // 2), I(0, 0), dual(I(0, 0)), dual(I(1, 1 // 2)))][1, 1, 1, 1] = 1
    t[(I(1, 1 // 2), I(1, -1 // 2), dual(I(0, 0)), dual(I(0, 0)))][1, 1, 1, 2] = 1
    t[(I(0, 0), I(0, 0), dual(I(1, -1 // 2)), dual(I(1, 1 // 2)))][2, 1, 1, 1] = -1
    t[(I(0, 0), I(1, -1 // 2), dual(I(1, -1 // 2)), dual(I(0, 0)))][2, 1, 1, 2] = -1
    return t
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, U1Irrep, Trivial)
    I = sectortype(t)
    t[(I(1, 1), I(0, 0), dual(I(0, 0)), dual(I(1, 1)))][1, 1, 1, 1] = 1
    t[(I(1, 1), I(1, 1), dual(I(0, 0)), dual(I(0, 2)))][1, 2, 1, 1] = 1
    t[(I(0, 2), I(0, 0), dual(I(1, 1)), dual(I(1, 1)))][1, 1, 2, 1] = -1
    t[(I(0, 2), I(1, 1), dual(I(1, 1)), dual(I(0, 2)))][1, 2, 2, 1] = -1
    return t
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = n_site_operator(Val(2), elt, U1Irrep, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1, 1 // 2), I(0, 0, 0), dual(I(0, 0, 0)), dual(I(1, 1, 1 // 2)))] .= 1
    t[(I(1, 1, 1 // 2), I(1, 1, -1 // 2), dual(I(0, 0, 0)), dual(I(0, 2, 0)))] .= 1
    t[(I(0, 2, 0), I(0, 0, 0), dual(I(1, 1, -1 // 2)), dual(I(1, 1, 1 // 2)))] .= -1
    t[(I(0, 2, 0), I(1, 1, -1 // 2), dual(I(1, 1, -1 // 2)), dual(I(0, 2, 0)))] .= -1
    return t
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function u_plus_u_min(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_plus_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
const u⁺u⁻ = u_plus_u_min

@doc """
    d_plus_d_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁺d⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e†_{1,↓}, e_{2,↓}`` that creates a spin-down particle at the first site and annihilates a spin-down particle at the second.
""" d_plus_d_min
d_plus_d_min(P::Type{<:Sector}, S::Type{<:Sector}) = d_plus_d_min(ComplexF64, P, S)
function d_plus_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 1, 2] = -1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 1, 1, 2] = -1
    return t
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep})
    t = n_site_operator(Val(2), elt, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, -1 // 2), I(0, 0), dual(I(0, 0)), dual(I(1, -1 // 2)))][1, 1, 1, 1] = 1
    t[(I(1, -1 // 2), I(1, 1 // 2), dual(I(0, 0)), dual(I(0, 0)))][1, 1, 1, 2] = -1
    t[(I(0, 0), I(0, 0), dual(I(1, 1 // 2)), dual(I(1, -1 // 2)))][2, 1, 1, 1] = 1
    t[(I(0, 0), I(1, 1 // 2), dual(I(1, 1 // 2)), dual(I(0, 0)))][2, 1, 1, 2] = -1
    return t
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, U1Irrep, Trivial)
    I = sectortype(t)
    t[(I(1, 1), I(0, 0), dual(I(0, 0)), dual(I(1, 1)))][2, 1, 1, 2] = 1
    t[(I(1, 1), I(1, 1), dual(I(0, 0)), dual(I(0, 2)))][2, 1, 1, 1] = -1
    t[(I(0, 2), I(0, 0), dual(I(1, 1)), dual(I(1, 1)))][1, 1, 1, 2] = 1
    t[(I(0, 2), I(1, 1), dual(I(1, 1)), dual(I(0, 2)))][1, 1, 1, 1] = -1
    return t
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = n_site_operator(Val(2), elt, U1Irrep, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1, -1 // 2), I(0, 0, 0), dual(I(0, 0, 0)), dual(I(1, 1, -1 // 2)))] .= 1
    t[(I(1, 1, -1 // 2), I(1, 1, 1 // 2), dual(I(0, 0, 0)), dual(I(0, 2, 0)))] .= -1
    t[(I(0, 2, 0), I(0, 0, 0), dual(I(1, 1, 1 // 2)), dual(I(1, 1, -1 // 2)))] .= 1
    t[(I(0, 2, 0), I(1, 1, 1 // 2), dual(I(1, 1, 1 // 2)), dual(I(0, 2, 0)))] .= -1
    return t
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function d_plus_d_min(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_plus_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
const d⁺d⁻ = d_plus_d_min

@doc """
    u_min_u_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁻u⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↑}, e†_{2,↑}`` that annihilates a spin-up particle at the first site and creates a spin-up particle at the second.
""" u_min_u_plus
u_min_u_plus(P::Type{<:Sector}, S::Type{<:Sector}) = u_min_u_plus(ComplexF64, P, S)
function u_min_u_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return -copy(adjoint(u_plus_u_min(elt, particle_symmetry, spin_symmetry)))
end
const u⁻u⁺ = u_min_u_plus

@doc """
    d_min_d_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁻d⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``e_{1,↓}, e†_{2,↓}`` that annihilates a spin-down particle at the first site and creates a spin-down particle at the second.
""" d_min_d_plus
d_min_d_plus(P::Type{<:Sector}, S::Type{<:Sector}) = d_min_d_plus(ComplexF64, P, S)
function d_min_d_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return -copy(adjoint(d_plus_d_min(elt, particle_symmetry, spin_symmetry)))
end
const d⁻d⁺ = d_min_d_plus

@doc """
    e_plus_e_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e⁺e⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
This is the sum of `u_plus_u_min` and `d_plus_d_min`.
""" e_plus_e_min
e_plus_e_min(P::Type{<:Sector}, S::Type{<:Sector}) = e_plus_e_min(ComplexF64, P, S)
function e_plus_e_min(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return u_plus_u_min(elt, particle_symmetry, spin_symmetry) +
        d_plus_d_min(elt, particle_symmetry, spin_symmetry)
end
function e_plus_e_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep})
    t = n_site_operator(Val(2), elt, Trivial, SU2Irrep)
    I = sectortype(t)
    f1 = only(fusiontrees((I(0, 0), I(1, 1 // 2)), I(1, 1 // 2)))
    f2 = only(fusiontrees((I(1, 1 // 2), I(0, 0)), I(1, 1 // 2)))
    t[f1, f2][1, 1, 1, 1] = 1
    f3 = only(fusiontrees((I(1, 1 // 2), I(0, 0)), I(1, 1 // 2)))
    f4 = only(fusiontrees((I(0, 0), I(1, 1 // 2)), I(1, 1 // 2)))
    t[f3, f4][1, 2, 2, 1] = -1
    f5 = only(fusiontrees((I(0, 0), I(0, 0)), I(0, 0)))
    f6 = only(fusiontrees((I(1, 1 // 2), I(1, 1 // 2)), I(0, 0)))
    t[f5, f6][1, 2, 1, 1] = sqrt(2)
    f7 = only(fusiontrees((I(1, 1 // 2), I(1, 1 // 2)), I(0, 0)))
    f8 = only(fusiontrees((I(0, 0), I(0, 0)), I(0, 0)))
    t[f7, f8][1, 1, 2, 1] = sqrt(2)
    return t
end
function e_plus_e_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = n_site_operator(Val(2), elt, U1Irrep, SU2Irrep)
    I = sectortype(t)
    f1 = only(fusiontrees((I(0, 0, 0), I(1, 1, 1 // 2)), I(1, 1, 1 // 2)))
    f2 = only(fusiontrees((I(1, 1, 1 // 2), I(0, 0, 0)), I(1, 1, 1 // 2)))
    t[f1, f2] .= 1
    f3 = only(fusiontrees((I(1, 1, 1 // 2), I(0, 2, 0)), I(1, 3, 1 // 2)))
    f4 = only(fusiontrees((I(0, 2, 0), I(1, 1, 1 // 2)), I(1, 3, 1 // 2)))
    t[f3, f4] .= -1
    f5 = only(fusiontrees((I(0, 0, 0), I(0, 2, 0)), I(0, 2, 0)))
    f6 = only(fusiontrees((I(1, 1, 1 // 2), I(1, 1, 1 // 2)), I(0, 2, 0)))
    t[f5, f6] .= sqrt(2)
    f7 = only(fusiontrees((I(1, 1, 1 // 2), I(1, 1, 1 // 2)), I(0, 2, 0)))
    f8 = only(fusiontrees((I(0, 2, 0), I(0, 0, 0)), I(0, 2, 0)))
    t[f7, f8] .= sqrt(2)
    return t
end
const e⁺e⁻ = e_plus_e_min

@doc """
    e_min_e_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e⁻e⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
This is the sum of `u_min_u_plus` and `d_min_d_plus`.
""" e_min_e_plus
e_min_e_plus(P::Type{<:Sector}, S::Type{<:Sector}) = e_min_e_plus(ComplexF64, P, S)
function e_min_e_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return -copy(adjoint(e_plus_e_min(elt, particle_symmetry, spin_symmetry)))
end
const e⁻e⁺ = e_min_e_plus

@doc """
    e_hopping([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e_hop([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that describes a particle that hops between the first and the second site.
""" e_hopping
e_hopping(P::Type{<:Sector}, S::Type{<:Sector}) = e_hopping(ComplexF64, P, S)
function e_hopping(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return e_plus_e_min(elt, particle_symmetry, spin_symmetry) -
        e_min_e_plus(elt, particle_symmetry, spin_symmetry)
end
function e_hopping(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    elt <: Complex || throw(DomainError(elt, "SU₂ × SU₂ symmetry requires complex entries"))
    t = n_site_operator(Val(2), elt, SU2Irrep, SU2Irrep)
    I = sectortype(t)
    even = I(0, 1 // 2, 0)
    odd = I(1, 0, 1 // 2)
    f1 = only(fusiontrees((odd, odd), one(I)))
    f2 = only(fusiontrees((even, even), one(I)))
    t[f1, f2] .= 2im
    t[f2, f1] .= -2im
    f3 = only(fusiontrees((even, odd), I((1, 1 // 2, 1 // 2))))
    f4 = only(fusiontrees((odd, even), I((1, 1 // 2, 1 // 2))))
    t[f3, f4] .= im
    t[f4, f3] .= -im
    return t
end
const e_hop = e_hopping

@doc """
    u_min_d_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    u⁻d⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e_{1,↑} e_{2,↓}`` that annihilates a spin-up particle at the first site and a spin-down particle at the second site.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↑,↓⟩,     +|0,↑⟩ ↤ |↑,↑↓⟩,
    +|↓,0⟩ ↤ |↑↓,↓⟩,    -|↓,↑⟩ ↤ |↑↓,↑↓⟩
```
""" u_min_d_min
function u_min_d_min(P::Type{<:Sector}, S::Type{<:Sector})
    return u_min_d_min(ComplexF64, P, S)
end
function u_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 1, 2] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 1, 1, 2] = 1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 2, 2] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 2, 2] = -1
    return t
end
function u_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep})
    t = n_site_operator(Val(2), elt, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(0, 0), I(0, 0), dual(I(1, 1 // 2)), dual(I(1, -1 // 2)))][1, 1, 1, 1] = -1
    t[(I(0, 0), I(1, 1 // 2), dual(I(1, 1 // 2)), dual(I(0, 0)))][1, 1, 1, 2] = 1
    t[(I(1, -1 // 2), I(0, 0), dual(I(0, 0)), dual(I(1, -1 // 2)))][1, 1, 2, 1] = 1
    t[(I(1, -1 // 2), I(1, 1 // 2), dual(I(0, 0)), dual(I(0, 0)))][1, 1, 2, 2] = -1
    return t
end
function u_min_d_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{<:Sector})
    throw(ArgumentError("`u_min_d_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function u_min_d_min(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_min_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_min_d_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_min_d_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` spin symmetry"))
end
const u⁻d⁻ = u_min_d_min

@doc """
    u_plus_d_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    u⁺d⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e†_{1,↑} e†_{2,↓}`` that annihilates a spin-up particle at the first site and a spin-down particle at the second site.
""" u_plus_d_plus
function u_plus_d_plus(P::Type{<:Sector}, S::Type{<:Sector})
    return u_plus_d_plus(ComplexF64, P, S)
end
function u_plus_d_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return -copy(adjoint(u_min_d_min(elt, particle_symmetry, spin_symmetry)))
end
const u⁺d⁺ = u_plus_d_plus

@doc """
    d_min_u_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    d⁻u⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e_{1,↓} e_{2,↑}`` that annihilates a spin-down particle at the first site and a spin-up particle at the second site.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↓,↑⟩,     -|0,↓⟩ ↤ |↓,↑↓⟩
    -|↑,0⟩ ↤ |↑↓,↑⟩,    -|↑,↓⟩ ↤ |↑↓,↑↓⟩
```
""" d_min_u_min
function d_min_u_min(P::Type{<:Sector}, S::Type{<:Sector})
    return d_min_u_min(ComplexF64, P, S)
end
function d_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 2, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 2, 2, 2] = -1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 2, 1] = -1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 2, 2] = -1
    return t
end
function d_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep})
    t = n_site_operator(Val(2), elt, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(0, 0), I(0, 0), dual(I(1, -1 // 2)), dual(I(1, 1 // 2)))][1, 1, 1, 1] = -1
    t[(I(0, 0), I(1, -1 // 2), dual(I(1, -1 // 2)), dual(I(0, 0)))][1, 1, 1, 2] = -1
    t[(I(1, 1 // 2), I(0, 0), dual(I(0, 0)), dual(I(1, 1 // 2)))][1, 1, 2, 1] = -1
    t[(I(1, 1 // 2), I(1, -1 // 2), dual(I(0, 0)), dual(I(0, 0)))][1, 1, 2, 2] = -1
    return t
end
function d_min_u_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{<:Sector})
    throw(ArgumentError("`d_min_u_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function d_min_u_min(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_min_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_min_u_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_min_u_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` particle symmetry"))
end
const d⁻u⁻ = d_min_u_min

@doc """
    d_plus_u_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    d⁺u⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e†_{1,↓} e†_{2,↑}`` that creates a spin-down particle at the first site and a spin-up particle at the second site.
""" d_plus_u_plus
function d_plus_u_plus(P::Type{<:Sector}, S::Type{<:Sector})
    return d_plus_u_plus(ComplexF64, P, S)
end
function d_plus_u_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return -copy(adjoint(d_min_u_min(elt, particle_symmetry, spin_symmetry)))
end
const d⁺u⁺ = d_plus_u_plus

@doc """
    u_min_u_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    u⁻u⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e_{1,↑} e_{2,↑}`` that annihilates a spin-up particle at both sites.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↑,↑⟩,     -|0,↓⟩ ↤ |↑,↑↓⟩
    +|↓,0⟩ ↤ |↑↓,↑⟩,    +|↓,↓⟩ ↤ |↑↓,↑↓⟩
```
""" u_min_u_min
function u_min_u_min(P::Type{<:Sector}, S::Type{<:Sector})
    return u_min_u_min(ComplexF64, P, S)
end
function u_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 1, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 2, 1, 2] = -1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 2, 1] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 2, 2, 2] = 1
    return t
end
function u_min_u_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{<:Sector})
    throw(ArgumentError("`u_min_u_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function u_min_u_min(::Type{<:Number}, ::Type{<:Sector}, ::Type{U1Irrep})
    throw(ArgumentError("`u_min_u_min` is not symmetric under `U1Irrep` spin symmetry"))
end
function u_min_u_min(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_min_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_min_u_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
    throw(ArgumentError("`u_min_u_min` is not symmetric under `U1Irrep` particle symmetry or under `U1Irrep` spin symmetry"))
end
function u_min_u_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_min_u_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` spin symmetry"))
end
const u⁻u⁻ = u_min_u_min

@doc """
    u_plus_u_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    u⁺u⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e†_{1,↑} e†_{2,↑}`` that creates a spin-up particle at both sites.
""" u_plus_u_plus
function u_plus_u_plus(P::Type{<:Sector}, S::Type{<:Sector})
    return u_plus_u_plus(ComplexF64, P, S)
end
function u_plus_u_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return -copy(adjoint(u_min_u_min(elt, particle_symmetry, spin_symmetry)))
end
const u⁺u⁺ = u_plus_u_plus

@doc """
    d_min_d_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    d⁻d⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e_{1,↓} e_{2,↓}`` that annihilates a spin-down particle at both sites.
The nonzero matrix elements are
```
    -|0,0⟩ ↤ |↓,↓⟩,     +|0,↑⟩ ↤ |↓,↑↓⟩
    -|↑,0⟩ ↤ |↑↓,↓⟩,    +|↑,↑⟩ ↤ |↑↓,↑↓⟩
```
""" d_min_d_min
function d_min_d_min(P::Type{<:Sector}, S::Type{<:Sector})
    return d_min_d_min(ComplexF64, P, S)
end
function d_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][1, 1, 2, 2] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][1, 1, 2, 2] = 1
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 2, 2] = -1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 1, 2, 2] = 1
    return t
end
function d_min_d_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{<:Sector})
    throw(ArgumentError("`d_min_d_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function d_min_d_min(::Type{<:Number}, ::Type{<:Sector}, ::Type{U1Irrep})
    throw(ArgumentError("`d_min_d_min` is not symmetric under `U1Irrep` spin symmetry"))
end
function d_min_d_min(::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_min_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_min_d_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
    throw(ArgumentError("`d_min_d_min` is not symmetric under `U1Irrep` particle symmetry or under `U1Irrep` spin symmetry"))
end
function d_min_d_min(::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_min_d_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` spin symmetry"))
end
const d⁻d⁻ = d_min_d_min

@doc """
    d_plus_d_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    d⁺d⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e†_{1,↓} e†_{2,↓}`` that creates a spin-down particle at both sites.
The nonzero matrix elements are
""" d_plus_d_plus
function d_plus_d_plus(P::Type{<:Sector}, S::Type{<:Sector})
    return d_plus_d_plus(ComplexF64, P, S)
end
function d_plus_d_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return -copy(adjoint(d_min_d_min(elt, particle_symmetry, spin_symmetry)))
end
const d⁺d⁺ = d_plus_d_plus

@doc """
    singlet_plus(elt, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    singlet⁺(elt, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body singlet operator ``(e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``,
which creates the singlet state when acting on vaccum.
""" singlet_plus
function singlet_plus(P::Type{<:Sector}, S::Type{<:Sector})
    return singlet_plus(ComplexF64, P, S)
end
function singlet_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return (
        u_plus_d_plus(elt, particle_symmetry, spin_symmetry) -
            d_plus_u_plus(elt, particle_symmetry, spin_symmetry)
    ) / sqrt(2)
end
function singlet_plus(
        elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep}
    )
    t = n_site_operator(Val(2), elt, Trivial, SU2Irrep)
    for (s, f) in fusiontrees(t)
        l1 = s.uncoupled[1][2].j
        l2 = s.uncoupled[2][2].j
        l3 = f.uncoupled[1][2].j
        l4 = f.uncoupled[2][2].j
        if (l1 == l2 == 1 // 2) && (l3 == l4 == 0)
            t[s, f][1, 1, 1, 1] = 1
        end
        if (l1 == l2 == 0) && (l3 == l4 == 1 // 2)
            t[s, f][2, 2, 1, 1] = -1
        end
        if (l1 == 0 && l2 == 1 // 2) && (l3 == 1 // 2 && l4 == 0)
            t[s, f][2, 1, 1, 1] = -1 / sqrt(2)
        end
        if (l1 == 1 // 2 && l2 == 0) && (l3 == 0 && l4 == 1 // 2)
            t[s, f][1, 2, 1, 1] = -1 / sqrt(2)
        end
    end
    return t
end
const singlet⁺ = singlet_plus

@doc """
    singlet_min(elt, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    singlet⁻(elt, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the adjoint of `singlet_plus` operator, which is 
``(-e_{1,↑} e_{2,↓} + e_{1,↓} e_{2,↑}) / \\sqrt{2}``.
""" singlet_min
function singlet_min(P::Type{<:Sector}, S::Type{<:Sector})
    return singlet_min(ComplexF64, P, S)
end
function singlet_min(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return copy(adjoint(singlet_plus(elt, particle_symmetry, spin_symmetry)))
end
const singlet⁻ = singlet_min

@doc """
    S_plus_S_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    S⁺S⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator S⁺S⁻.
The only nonzero matrix element corresponds to `|↑,↓⟩ <-- |↓,↑⟩`.
""" S_plus_S_min
function S_plus_S_min(P::Type{<:Sector}, S::Type{<:Sector})
    return S_plus_S_min(ComplexF64, P, S)
end
function S_plus_S_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(1)), dual(I(1)))][1, 2, 2, 1] = 1
    return t
end
function S_plus_S_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep})
    t = n_site_operator(Val(2), elt, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1 // 2), I(1, -1 // 2), dual(I(1, -1 // 2)), dual(I(1, 1 // 2)))] .= 1
    return t
end
function S_plus_S_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial})
    t = n_site_operator(Val(2), elt, U1Irrep, Trivial)
    I = sectortype(t)
    t[(I(1, 1), I(1, 1), dual(I(1, 1)), dual(I(1, 1)))][1, 2, 2, 1] = 1
    return t
end
function S_plus_S_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = n_site_operator(Val(2), elt, U1Irrep, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1, 1 // 2), I(1, 1, -1 // 2), dual(I(1, 1, -1 // 2)), dual(I(1, 1, 1 // 2)))] .= 1
    return t
end
const S⁺S⁻ = S_plus_S_min

@doc """
    S_min_S_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    S⁻S⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator S⁻S⁺.
The only nonzero matrix element corresponds to `|↓,↑⟩ <-- |↑,↓⟩`.
""" S_min_S_plus
function S_min_S_plus(P::Type{<:Sector}, S::Type{<:Sector})
    return S_min_S_plus(ComplexF64, P, S)
end
function S_min_S_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    return copy(adjoint(S_plus_S_min(elt, particle_symmetry, spin_symmetry)))
end
const S⁻S⁺ = S_min_S_plus

@doc """
    S_exchange(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the spin exchange operator S⋅S.
""" S_exchange
function S_exchange(P::Type{<:Sector}, S::Type{<:Sector})
    return S_exchange(ComplexF64, P, S)
end
function S_exchange(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    Sz = S_z(elt, particle_symmetry, spin_symmetry)
    return Sz ⊗ Sz + (
        S_plus_S_min(elt, particle_symmetry, spin_symmetry) +
            S_min_S_plus(elt, particle_symmetry, spin_symmetry)
    ) / 2
end
function S_exchange(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep})
    t = n_site_operator(Val(2), elt, Trivial, SU2Irrep)
    for (s, f) in fusiontrees(t)
        l3 = f.uncoupled[1][2].j
        l4 = f.uncoupled[2][2].j
        k = f.coupled[2].j
        t[s, f] .= (k * (k + 1) - l3 * (l3 + 1) - l4 * (l4 + 1)) / 2
    end
    return t
end
function S_exchange(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = n_site_operator(Val(2), elt, U1Irrep, SU2Irrep)
    for (s, f) in fusiontrees(t)
        l3 = f.uncoupled[1][3].j
        l4 = f.uncoupled[2][3].j
        k = f.coupled[3].j
        t[s, f] .= (k * (k + 1) - l3 * (l3 + 1) - l4 * (l4 + 1)) / 2
    end
    return t
end

# Three site operators
# --------------------

@doc """
    singlet_plus_singlet_min_3site(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Returns the 3-site term ``O_{ijk} = A^†_{ij} A_{jk}``, where
``A^†_{ij} = (e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``.
It describes the hopping of a singlet pair from bond `(j,k)`
to a nearest neighbor bond `(i,j)` sharing site `j`.
""" singlet_plus_singlet_min_3site
function singlet_plus_singlet_min_3site(P::Type{<:Sector}, S::Type{<:Sector})
    return singlet_plus_singlet_min_3site(ComplexF64, P, S)
end
function singlet_plus_singlet_min_3site(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
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
    singp = singlet_plus(elt, particle_symmetry, spin_symmetry)
    singm = singp'
    return @tensor t[-1 -2 -3; -4 -5 -6] := singp[-1 -2; -4 1] * singm[1 -3; -5 -6]
end
function singlet_plus_singlet_min_3site(elt::Type{<:Number}, ::Type{U1Irrep}, spin_symmetry::Type{<:Sector})
    #= rewrite the operator as

    O_{ijk}
    = ∑_σ (c†_{iσ} c†_{jσ̄} c_{jσ̄} c_{kσ} - c†_{iσ̄} c†_{jσ} c_{jσ̄} c_{kσ})
    = ∑_σ [c†_{iσ} (c†_{jσ̄} c_{jσ̄}) c_{kσ} + (c†_{jσ} c_{kσ}) (c†_{iσ̄} c_{jσ̄})]

    also use the contraction

        -4          -5
    ┌---┴-----------┴---┐
    |   c†_{iσ̄} c_{jσ̄}  |
    └---┬-----------┬---┘
        -1          1           -6
                ┌---┴-----------┴---┐
                |   c†_{jσ} c_{kσ}  |
                └---┬-----------┬---┘
                    -2          -3
        i           j           k
    =#
    hop_up = u_plus_u_min(elt, U1Irrep, spin_symmetry)
    hop_down = d_plus_d_min(elt, U1Irrep, spin_symmetry)
    Nu = u_num(elt, U1Irrep, spin_symmetry)
    Nd = d_num(elt, U1Irrep, spin_symmetry)
    t = permute(hop_up ⊗ Nd + hop_down ⊗ Nu, ((1, 3, 2), (4, 6, 5)))
    t += @tensor t3[-1 -2 -3; -4 -5 -6] := hop_down[-2 -3; 1 -6] * hop_up[-1 1; -4 -5]
    t += @tensor t4[-1 -2 -3; -4 -5 -6] := hop_up[-2 -3; 1 -6] * hop_down[-1 1; -4 -5]
    return t
end
function singlet_plus_singlet_min_3site(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    op1 = singlet_plus_singlet_min_3site(elt, Trivial, SU2Irrep)
    return _promote_particle_u1(op1)
end

# Four site operators
# -------------------

@doc """
    singlet_plus_singlet_min_4site(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Returns the 4-site term ``O_{ijkl} = A^†_{ij} A_{kl}``, where
``A^†_{ij} = (e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``.
It measures the singlet pair correlation between two bonds `(i,j)` and `(k,l)`.
""" singlet_plus_singlet_min_4site
function singlet_plus_singlet_min_4site(P::Type{<:Sector}, S::Type{<:Sector})
    return singlet_plus_singlet_min_4site(ComplexF64, P, S)
end
function singlet_plus_singlet_min_4site(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    singp = singlet_plus(elt, particle_symmetry, spin_symmetry)
    return singp ⊗ singp'
end
function singlet_plus_singlet_min_4site(elt::Type{<:Number}, ::Type{U1Irrep}, spin_symmetry::Type{<:Sector})
    #= rewrite the operator as

    O_{ijkl}
    = ∑_σ (c†_{iσ} c†_{jσ̄} c_{kσ̄} c_{lσ} - c†_{iσ̄} c†_{jσ} c_{kσ̄} c_{lσ})
    = ∑_σ [(c†_{iσ} c_{lσ}) (c†_{jσ̄} c_{kσ̄}) + (c†_{iσ} c_{kσ}) (c†_{jσ̄} c_{lσ̄})]
    =#
    hop_up = u_plus_u_min(elt, U1Irrep, spin_symmetry)
    hop_down = d_plus_d_min(elt, U1Irrep, spin_symmetry)
    hop2 = hop_up ⊗ hop_down + hop_down ⊗ hop_up
    return permute(hop2, ((1, 3, 4, 2), (5, 7, 8, 6))) +
        permute(hop2, ((1, 3, 2, 4), (5, 7, 6, 8)))
end
function singlet_plus_singlet_min_4site(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    op1 = singlet_plus_singlet_min_4site(elt, Trivial, SU2Irrep)
    return _promote_particle_u1(op1)
end

end
