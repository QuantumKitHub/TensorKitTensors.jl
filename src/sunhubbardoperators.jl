"""
    module SUNHubbardOperators

Module for the ``SU(N)`` generalizations of the [`HubbardOperators`](@ref hubbard_operators)
operators.

!!! note
    This module requires `using SUNRepresentations` in order to define the `SUNIrrep`-symmetric tensors.

"""
module SUNHubbardOperators

using TensorKit

export hubbard_space
export e_num, e_double
export e_plus_e_min, e_min_e_plus, e_hopping

export n, nn
export e⁺e⁻, e⁻e⁺, e_hop

"""
    hubbard_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; N::Integer = 3)

Return the local hilbert space for a Hubbard-type model with the given particle and spin symmetries. 
The basis is spanned by all possible fermionic color-like combinations of particles that satisfy the Pauli exclusion principle.

For example, for ``SU(3)`` (`N = 3`), we have three colors (`r`, `g` and `b`), leading to the following eight possible states:

```math
|0⟩, |r⟩, |g⟩, |b⟩, |rg⟩, |rb⟩, |gb⟩, |rgb⟩
```

Additionally, we can implement `Trivial` or `U1Irrep` symmetry for the particle number conservation.
"""
function hubbard_space(::Type{PS} = Trivial, ::Type{SS} = Trivial; N::Integer = 3) where {PS <: Sector, SS <: Sector}
    if SS === Trivial
        if PS === Trivial
            N == 2 && return Vect[FermionParity](0 => 2, 1 => 2)
            N == 3 && return Vect[FermionParity](0 => 4, 1 => 4)
        elseif PS === U1Irrep
            N == 2 && return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1)
            N == 3 && return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 3, (0, 2) => 3, (1, 3) => 1)
        end
    end

    error(lazy"combination of particle symmetry ($PS) and spin symmetry ($SS) not supported or not implemented for N = $N")
end


# Single-site operators
# ---------------------
function single_site_operator(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        kwargs...
    )
    V = hubbard_space(particle_symmetry, spin_symmetry; kwargs...)
    return zeros(elt, V ← V)
end


@doc """
    e_num([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    n([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of particles across all particle
flavors.

```math
n = ∑_{α} n_α
```
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
    t = single_site_operator(elt, Trivial, SU2Irrep)
    I = sectortype(t)
    block(t, I(1, 1 // 2))[1, 1] = 1
    block(t, I(0, 0))[2, 2] = 2
    return t
end
function e_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = single_site_operator(elt, U1Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I(1, 1, 1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 2
    return t
end
const n = e_num

@doc """
    e_double([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    nn([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of doubly occupied flavor pairs on a
given site.

```math
nn = ∑_{α < β} n_α n_β
```
""" e_double
e_double(P::Type{<:Sector}, S::Type{<:Sector}) = e_double(ComplexF64, P, S)
function e_double(
        ::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}
    )
    throw(ArgumentError("e_double is not defined for particle symmetry $particle_symmetry and spin symmetry $spin_symmetry"))
end
function e_double(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep})
    t = single_site_operator(elt, Trivial, SU2Irrep)
    I = sectortype(t)
    block(t, I(0, 0))[2, 2] = 1
    return t
end
function e_double(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = single_site_operator(elt, U1Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I(0, 2, 0)) .= 1
    return t
end

# Two site operators
# ------------------
function two_site_operator(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        kwargs...
    )
    V = hubbard_space(particle_symmetry, spin_symmetry; kwargs...)
    return zeros(elt, V ⊗ V ← V ⊗ V)
end

@doc """
    e_plus_e_min([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e⁺e⁻([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
""" e_plus_e_min

e_plus_e_min(P::Type{<:Sector}, S::Type{<:Sector}; kwargs...) = e_plus_e_min(ComplexF64, P, S; kwargs...)

const e⁺e⁻ = e_plus_e_min

@doc """
    e_min_e_plus([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e⁻e⁺([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
""" e_min_e_plus
e_min_e_plus(P::Type{<:Sector}, S::Type{<:Sector}; kwargs...) = e_min_e_plus(ComplexF64, P, S; kwargs...)
function e_min_e_plus(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}; kwargs...
    )
    return -copy(adjoint(e_plus_e_min(elt, particle_symmetry, spin_symmetry; kwargs...)))
end
const e⁻e⁺ = e_min_e_plus

@doc """
    e_hopping([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])
    e_hop([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}])

Return the two-body operator that describes a particle that hops between the first and the second site.
""" e_hopping
e_hopping(P::Type{<:Sector}, S::Type{<:Sector}; kwargs...) = e_hopping(ComplexF64, P, S; kwargs...)
function e_hopping(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        kwargs...
    )
    return e_plus_e_min(elt, particle_symmetry, spin_symmetry; kwargs...) -
        e_min_e_plus(elt, particle_symmetry, spin_symmetry; kwargs...)
end

const e_hop = e_hopping

end
