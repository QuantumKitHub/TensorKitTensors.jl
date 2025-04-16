module HubbardOperators

using TensorKit

export hubbard_space
export c_plus_c_min, u_plus_u_min, d_plus_d_min
export c_min_c_plus, u_min_u_plus, d_min_d_plus
export c_hop
export c_num, u_num, d_num, ud_num

export c⁺c⁻, u⁺u⁻, d⁺d⁻, c⁻c⁺, u⁻u⁺, d⁻d⁺
export n, nꜛ, nꜜ, nꜛꜜ

"""
    hubbard_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the local hilbert space for a Hubbard-type model with the given particle and spin symmetries.
The possible symmetries are `Trivial`, `U1Irrep`, and `SU2Irrep`, for both particle number and spin.
"""
function hubbard_space(::Type{Trivial}=Trivial, ::Type{Trivial}=Trivial)
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
    return Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1,
                                                   (1, 1, -1 // 2) => 1, (0, 2, 0) => 1)
end
function hubbard_space(::Type{U1Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1,
                                                    (0, 2, 0) => 1)
end
function hubbard_space(::Type{SU2Irrep}, ::Type{Trivial})
    return Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 2, (1, 1 // 2) => 1)
end
function hubbard_space(::Type{SU2Irrep}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ SU2Irrep ⊠ U1Irrep]((0, 0, 0) => 1, (1, 1 // 2, 1) => 1)
end
function hubbard_space(::Type{SU2Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ SU2Irrep ⊠ SU2Irrep]((1, 1 // 2, 1 // 2) => 1)
end

# Single-site operators
# ---------------------
function single_site_operator(T, particle_symmetry::Type{<:Sector},
                              spin_symmetry::Type{<:Sector})
    V = hubbard_space(particle_symmetry, spin_symmetry)
    return zeros(T, V ← V)
end

@doc """
    u_num([particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜛ([particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of spin-up particles.
""" u_num
u_num(P::Type{<:Sector}, S::Type{<:Sector}) = u_num(ComplexF64, P, S)
function u_num(T::Type{<:Number}, (::Type{Trivial})=Trivial, (::Type{Trivial})=Trivial)
    t = single_site_operator(T, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(1))][1, 1] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end
function u_num(T, ::Type{Trivial}, ::Type{U1Irrep})
    t = single_site_operator(T, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1 // 2), dual(I(1, 1 // 2)))][1, 1] = 1
    t[(I(0, 0), dual(I(0, 0)))][2, 2] = 1
    return t
end
function u_num(T, ::Type{Trivial}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_num(T, ::Type{U1Irrep}, ::Type{Trivial})
    t = single_site_operator(T, U1Irrep, Trivial)
    I = sectortype(t)
    block(t, I(1, 1))[1, 1] = 1
    block(t, I(0, 2))[1, 1] = 1
    return t
end
function u_num(T, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = single_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    block(t, I(1, 1, 1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 1
    return t
end
function u_num(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_num(T, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function u_num(T, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function u_num(T, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜛ = u_num

@doc """
    d_num([particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜜ([particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of spin-down particles.
""" d_num
d_num(P::Type{<:Sector}, S::Type{<:Sector}) = d_num(ComplexF64, P, S)
function d_num(T::Type{<:Number}, ::Type{Trivial}=Trivial, ::Type{Trivial}=Trivial)
    t = single_site_operator(T, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(1))][2, 2] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end
function d_num(T, ::Type{Trivial}, ::Type{U1Irrep})
    t = single_site_operator(T, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, -1 // 2), dual(I(1, -1 // 2)))][1, 1] = 1
    t[(I(0, 0), I(0, 0))][2, 2] = 1
    return t
end
function d_num(T, ::Type{Trivial}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_num(T, ::Type{U1Irrep}, ::Type{Trivial})
    t = single_site_operator(T, U1Irrep, Trivial)
    I = sectortype(t)
    block(t, I(1, 1))[2, 2] = 1 # expected to be [1,2]
    block(t, I(0, 2))[1, 1] = 1
    return t
end
function d_num(T, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = single_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    block(t, I(1, 1, -1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 1
    return t
end
function d_num(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_num(T, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function d_num(T, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function d_num(T, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜜ = d_num

@doc """
    c_num([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    n([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of particles.
""" c_num
c_num(P::Type{<:Sector}, S::Type{<:Sector}) = c_num(ComplexF64, P, S)
function c_num(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return u_num(T, particle_symmetry, spin_symmetry) +
           d_num(T, particle_symmetry, spin_symmetry)
end
function c_num(T, ::Type{Trivial}, ::Type{SU2Irrep})
    t = single_site_operator(T, Trivial, SU2Irrep)
    I = sectortype(t)
    block(t, I(1, 1 // 2))[1, 1] = 1
    block(t, I(0, 0))[2, 2] = 2
    return t
end
function c_num(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = single_site_operator(T, U1Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I(1, 1, 1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 2
    return t
end
const n = c_num

@doc """
    ud_num([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nꜛꜜ([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of doubly occupied sites.
""" ud_num
ud_num(P::Type{<:Sector}, S::Type{<:Sector}) = ud_num(ComplexF64, P, S)
function ud_num(T, particle_symmetry::Type{<:Sector},
                spin_symmetry::Type{<:Sector})
    return u_num(T, particle_symmetry, spin_symmetry) *
           d_num(T, particle_symmetry, spin_symmetry)
end
function ud_num(T, ::Type{Trivial}, ::Type{SU2Irrep})
    t = single_site_operator(T, Trivial, SU2Irrep)
    I = sectortype(t)
    block(t, I(0, 0))[2, 2] = 1
    return t
end
function ud_num(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = single_site_operator(T, U1Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I(0, 2, 0)) .= 1
    return t
end
const nꜛꜜ = ud_num

# Two site operators
# ------------------
function two_site_operator(T, particle_symmetry::Type{<:Sector},
                           spin_symmetry::Type{<:Sector})
    V = hubbard_space(particle_symmetry, spin_symmetry)
    return zeros(T, V ⊗ V ← V ⊗ V)
end

@doc """
    u_plus_u_min([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁺d⁻([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``c†_{1,↑}, c_{2,↑}`` that creates a spin-up particle at the first site and annihilates a spin-up particle at the second.
""" u_plus_u_min
u_plus_u_min(P::Type{<:Sector}, S::Type{<:Sector}) = u_plus_u_min(ComplexF64, P, S)
function u_plus_u_min(T, ::Type{Trivial}, ::Type{Trivial})
    t = two_site_operator(T, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 1, 1] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 1, 2] = 1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 2, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 2, 2, 2] = -1
    return t
end
function u_plus_u_min(T, ::Type{Trivial}, ::Type{U1Irrep})
    t = two_site_operator(T, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1 // 2), I(0, 0), dual(I(0, 0)), dual(I(1, 1 // 2)))][1, 1, 1, 1] = 1
    t[(I(1, 1 // 2), I(1, -1 // 2), dual(I(0, 0)), dual(I(0, 0)))][1, 1, 1, 2] = 1
    t[(I(0, 0), I(0, 0), dual(I(1, -1 // 2)), dual(I(1, 1 // 2)))][2, 1, 1, 1] = -1
    t[(I(0, 0), I(1, -1 // 2), dual(I(1, -1 // 2)), dual(I(0, 0)))][2, 1, 1, 2] = -1
    return t
end
function u_plus_u_min(T, ::Type{Trivial}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_plus_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_plus_u_min(T, ::Type{U1Irrep}, ::Type{Trivial})
    t = two_site_operator(T, U1Irrep, Trivial)
    I = sectortype(t)
    t[(I(1, 1), I(0, 0), dual(I(0, 0)), dual(I(1, 1)))][1, 1, 1, 1] = 1
    t[(I(1, 1), I(1, 1), dual(I(0, 0)), dual(I(0, 2)))][1, 2, 1, 1] = 1
    t[(I(0, 2), I(0, 0), dual(I(1, 1)), dual(I(1, 1)))][1, 1, 2, 1] = -1
    t[(I(0, 2), I(1, 1), dual(I(1, 1)), dual(I(0, 2)))][1, 2, 2, 1] = -1
    return t
end
function u_plus_u_min(T, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = two_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1, 1 // 2), I(0, 0, 0), dual(I(0, 0, 0)), dual(I(1, 1, 1 // 2)))] .= 1
    t[(I(1, 1, 1 // 2), I(1, 1, -1 // 2), dual(I(0, 0, 0)), dual(I(0, 2, 0)))] .= 1
    t[(I(0, 2, 0), I(0, 0, 0), dual(I(1, 1, -1 // 2)), dual(I(1, 1, 1 // 2)))] .= -1
    t[(I(0, 2, 0), I(1, 1, -1 // 2), dual(I(1, 1, -1 // 2)), dual(I(0, 2, 0)))] .= -1
    return t
end
function u_plus_u_min(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_plus_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_plus_u_min(T, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function u_plus_u_min(T, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function u_plus_u_min(T, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`u_plus_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
const u⁺u⁻ = u_plus_u_min

@doc """
    d_plus_d_min([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁺d⁻([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator ``c†_{1,↓}, c_{2,↓}`` that creates a spin-down particle at the first site and annihilates a spin-down particle at the second.
""" d_plus_d_min
d_plus_d_min(P::Type{<:Sector}, S::Type{<:Sector}) = d_plus_d_min(ComplexF64, P, S)
function d_plus_d_min(T, ::Type{Trivial}, ::Type{Trivial})
    t = two_site_operator(T, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 1, 2] = -1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 1, 1, 2] = -1
    return t
end
function d_plus_d_min(T, ::Type{Trivial}, ::Type{U1Irrep})
    t = two_site_operator(T, Trivial, U1Irrep)
    I = sectortype(t)
    t[(I(1, -1 // 2), I(0, 0), dual(I(0, 0)), dual(I(1, -1 // 2)))][1, 1, 1, 1] = 1
    t[(I(1, -1 // 2), I(1, 1 // 2), dual(I(0, 0)), dual(I(0, 0)))][1, 1, 1, 2] = -1
    t[(I(0, 0), I(0, 0), dual(I(1, 1 // 2)), dual(I(1, -1 // 2)))][2, 1, 1, 1] = 1
    t[(I(0, 0), I(1, 1 // 2), dual(I(1, 1 // 2)), dual(I(0, 0)))][2, 1, 1, 2] = -1
    return t
end
function d_plus_d_min(T, ::Type{Trivial}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_plus_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_plus_d_min(T, ::Type{U1Irrep}, ::Type{Trivial})
    t = two_site_operator(T, U1Irrep, Trivial)
    I = sectortype(t)
    t[(I(1, 1), I(0, 0), dual(I(0, 0)), dual(I(1, 1)))][2, 1, 1, 2] = 1
    t[(I(1, 1), I(1, 1), dual(I(0, 0)), dual(I(0, 2)))][2, 1, 1, 1] = -1
    t[(I(0, 2), I(0, 0), dual(I(1, 1)), dual(I(1, 1)))][1, 1, 1, 2] = 1
    t[(I(0, 2), I(1, 1), dual(I(1, 1)), dual(I(0, 2)))][1, 1, 1, 1] = -1
    return t
end
function d_plus_d_min(T, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = two_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1, -1 // 2), I(0, 0, 0), dual(I(0, 0, 0)), dual(I(1, 1, -1 // 2)))] .= 1
    t[(I(1, 1, -1 // 2), I(1, 1, 1 // 2), dual(I(0, 0, 0)), dual(I(0, 2, 0)))] .= -1
    t[(I(0, 2, 0), I(0, 0, 0), dual(I(1, 1, 1 // 2)), dual(I(1, 1, -1 // 2)))] .= 1
    t[(I(0, 2, 0), I(1, 1, 1 // 2), dual(I(1, 1, 1 // 2)), dual(I(0, 2, 0)))] .= -1
    return t
end
function d_plus_d_min(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_plus_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_plus_d_min(T, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function d_plus_d_min(T, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function d_plus_d_min(T, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`d_plus_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
const d⁺d⁻ = d_plus_d_min

@doc """
    u_min_u_plus([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    u⁻u⁺([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the Hermitian conjugate of `u_plus_u_min`, i.e.
``(c†_{1,↑}, c_{2,↑})† = -c_{1,↑}, c†_{2,↑}`` (note the extra minus sign). 
It annihilates a spin-up particle at the first site and creates a spin-up particle at the second.
""" u_min_u_plus
u_min_u_plus(P::Type{<:Sector}, S::Type{<:Sector}) = u_min_u_plus(ComplexF64, P, S)
function u_min_u_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return copy(adjoint(u_plus_u_min(T, particle_symmetry, spin_symmetry)))
end
const u⁻u⁺ = u_min_u_plus

@doc """
    d_min_d_plus([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    d⁻d⁺([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the Hermitian conjugate of `d_plus_d_min`, i.e.
``(c†_{1,↓}, c_{2,↓})† = -c_{1,↓}, c†_{2,↓}`` (note the extra minus sign). 
It annihilates a spin-down particle at the first site and creates a spin-down particle at the second.
""" d_min_d_plus
d_min_d_plus(P::Type{<:Sector}, S::Type{<:Sector}) = d_min_d_plus(ComplexF64, P, S)
function d_min_d_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return copy(adjoint(d_plus_d_min(T, particle_symmetry, spin_symmetry)))
end
const d⁻d⁺ = d_min_d_plus

@doc """
    c_plus_c_min([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    c⁺c⁻([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
This is the sum of `u_plus_u_min` and `d_plus_d_min`.
""" c_plus_c_min
c_plus_c_min(P::Type{<:Sector}, S::Type{<:Sector}) = c_plus_c_min(ComplexF64, P, S)
function c_plus_c_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return u_plus_u_min(T, particle_symmetry, spin_symmetry) +
           d_plus_d_min(T, particle_symmetry, spin_symmetry)
end
function c_plus_c_min(T, ::Type{Trivial}, ::Type{SU2Irrep})
    t = two_site_operator(T, Trivial, SU2Irrep)
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
function c_plus_c_min(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = two_site_operator(T, U1Irrep, SU2Irrep)
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
const c⁺c⁻ = c_plus_c_min

@doc """
    c_min_c_plus([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    c⁻c⁺([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
This is the sum of `u_min_u_plus` and `d_min_d_plus`.
""" c_min_c_plus
c_min_c_plus(P::Type{<:Sector}, S::Type{<:Sector}) = c_min_c_plus(ComplexF64, P, S)
function c_min_c_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return copy(adjoint(c_plus_c_min(T, particle_symmetry, spin_symmetry)))
end
const c⁻c⁺ = c_min_c_plus

@doc """
    c_hop([T], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that describes a particle that hops between the first and the second site.
This is the sum of `c_plus_c_min` and `c_min_c_plus`.
""" c_hop
c_hop(P::Type{<:Sector}, S::Type{<:Sector}) = c_hop(ComplexF64, P, S)
function c_hop(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return c_plus_c_min(T, particle_symmetry, spin_symmetry) +
           c_min_c_plus(T, particle_symmetry, spin_symmetry)
end

end
